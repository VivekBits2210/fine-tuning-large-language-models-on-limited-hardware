import logging
import os
from tqdm import tqdm
from transformers import AdamW

from config import SystemConfiguration, UserConfiguration, TokenizerConfiguration, TrainerConfiguration, \
    TextGenConfiguration
from data_manager import DataManager
from model_manager import ModelManager
from profiler_utils import measure_time_taken
from tokenization_manager import TokenizationManager

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self,
                 model_name,
                 user_config: UserConfiguration,
                 system_config: SystemConfiguration,
                 tokenizer_config: TokenizerConfiguration,
                 text_gen_config: TextGenConfiguration,
                 train_config: TrainerConfiguration,
                 data_manager: DataManager,
                 model_manager: ModelManager,
                 tokenization_manager: TokenizationManager,
                 training_dataloader,
                 validation_dataloader):
        self.model_name = model_name
        self.user_config = user_config
        self.system_config = system_config
        self.tokenizer_config = tokenizer_config
        self.text_gen_config = text_gen_config
        self.train_config = train_config

        self.data_manager = data_manager
        self.model_manager = model_manager
        self.tokenization_manager = tokenization_manager

        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.num_batches = len(self.training_dataloader)

        self.log_path = None
        self.model_path = None
        self._setup_logging_and_saving()

        self.optimizer = AdamW(params=self.model_manager.model.parameters(), lr=1e-5)
        self.running_loss = 0.0

    @measure_time_taken
    def _setup_logging_and_saving(self):
        model_name = self.model_name
        dataset_name = self.data_manager.dataset_name
        tokenizer_name = self.tokenizer_config.tokenizer_name

        self.log_path = self.user_config.logs_path_generator(model_name, dataset_name, tokenizer_name)

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        with open(f"{self.log_path}/training.log", "w+") as f:
            f.write("epoch\tbatch\ttrain\tloss\tgenerated_text\n")
        with open(f"{self.log_path}/validation.log", "w+") as f:
            f.write("epoch\tbatch\tvalidation_loss\tperplexity\n")

        self.model_path = self.user_config.model_path_generator(model_name, dataset_name, tokenizer_name)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    @measure_time_taken
    def handle_batch(self, epoch, index, batch):
        self.model_manager.model.train()
        # Sample an output from the model, at each sampling interval
        if index % self.train_config.sampling_interval == 0:
            prompt = self.tokenization_manager.encode("This")
            sequence = self.model_manager.infer(prompt, self.text_gen_config)
            text = self.tokenization_manager.decode(sequence, self.text_gen_config)
            logger.info(f"Text:\n{text}")
            with open(f"{self.log_path}/training.log", "a") as f:
                f.write(f"{epoch}\t{index}\t{self.running_loss/index}\t{text}\n")

        # Save the model at each checkpointing interval
        if index % self.train_config.checkpointing_interval == 0:
            self.save_checkpoint(epoch, index)

        # Validate the model at each validation interval
        if index % self.train_config.validation_interval == 0:
            self.validate_model(epoch, index)

        self.forward_backward_pass(batch)

    @measure_time_taken
    def save_checkpoint(self, epoch, index):
        logger.info(f"Checkpointing model at epoch={epoch} and batch={index}\n")
        checkpointing_path = f"{self.model_path}_{epoch}_{index}"
        self.model_manager.model.save_pretrained(checkpointing_path)
        self.tokenization_manager.tokenizer.save_pretrained(checkpointing_path)

    @measure_time_taken
    def validate_model(self, epoch, index):
        logger.info("Running Validation...")
        avg_eval_loss, perplexity = self.model_manager.validate(self.validation_dataloader)
        logger.info(
            f"Batch {index}/{len(self.training_dataloader)}, Validation Loss: {avg_eval_loss:.4f}, "
            f"Perplexity: {perplexity:.2f}")
        with open(f"{self.log_path}/validation.log", "a") as f:
            f.write(f"{epoch}\t{index}\t{avg_eval_loss}\t{perplexity}\n")

    @measure_time_taken
    def forward_backward_pass(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        batch = {k: v.pin_memory().to(self.model_manager.device, non_blocking=True) for k, v in batch.items()}
        outputs = self.model_manager.model(**batch)
        loss = outputs.loss
        self.running_loss += loss.item()
        loss.backward()
        self.optimizer.step()

    def run(self):
        self.running_loss = 0.0
        for epoch in tqdm(range(1, self.train_config.epochs + 1)):
            logger.info(f"Starting Epoch: {epoch}/{self.train_config.epochs}")

            for index, batch in tqdm(enumerate(self.training_dataloader, 1), total=len(self.training_dataloader)):
                self.handle_batch(epoch, index, batch)

        self.model_manager.model.save_pretrained(self.model_path)
        self.tokenization_manager.tokenizer.save_pretrained(self.model_path)
