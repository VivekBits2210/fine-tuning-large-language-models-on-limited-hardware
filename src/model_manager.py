import logging
import torch
from tqdm import tqdm
from typing import Optional
from transformers import AutoModelForCausalLM, AutoConfig

from profiler_utils import measure_time_taken
from config.system_configuration import SystemConfiguration
logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, system_config: SystemConfiguration) -> None:
        self.device = system_config.device
        self.model = None
        self.model_name: Optional[str] = None

    @measure_time_taken
    def load_from_path(self, load_path: str) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(load_path)
        self.model.to(self.device)
        self.__augment_model()

    @measure_time_taken
    def load(self, model_name: str) -> None:
        self.model_name = model_name

        configuration = AutoConfig.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, config=configuration)
        self.model.to(self.device)
        self.__augment_model()

    def __augment_model(self):
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()
        self.model.config.use_cache = False

    def _generate_tokens(self, inputs, text_gen_config):
        return self.model.generate(
            **inputs,
            bos_token_id=text_gen_config.beginning_of_sentence_token_id,
            pad_token_id=text_gen_config.pad_token_id,
            do_sample=text_gen_config.do_sample,
            top_k=text_gen_config.top_k,
            min_length=text_gen_config.min_tokens_to_generate,
            max_length=text_gen_config.max_tokens_to_generate,
            top_p=text_gen_config.top_p,
            num_return_sequences=text_gen_config.num_return_sequences
        )

    def infer(self, prompt, text_gen_config):
        prompt.to_device(self.device)

        self.model.config.use_cache = True
        self.model.eval()

        with torch.no_grad():
            output_sequence = self._generate_tokens(prompt, text_gen_config)

        self.model.train()
        self.model.config.use_cache = False

        return output_sequence

    def validate(self, validation_dataloader):
        self.model.eval()
        total_eval_loss = 0.0
        avg_eval_loss = 0.0
        for index, batch in tqdm(enumerate(validation_dataloader, 1)):
            batch = {k: v.pin_memory().to(self.device, non_blocking=True) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
                loss = outputs.loss
            total_eval_loss += loss.item()
            avg_eval_loss = total_eval_loss / index
            logger.info(f"Validation: Batch {index}/{len(validation_dataloader)}, Loss: {avg_eval_loss:.4f}")

        perplexity = torch.exp(torch.as_tensor(avg_eval_loss)).item()
        self.model.train()

        return avg_eval_loss, perplexity




