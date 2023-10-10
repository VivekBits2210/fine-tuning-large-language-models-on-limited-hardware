import logging
import gc
import torch

from config import (
    UserConfiguration,
    LogConfiguration,
    TorchConfiguration,
    TokenizerConfiguration,
    TextGenConfiguration,
    SystemConfiguration, LoraConfiguration,
)

from os_environment_manager import OSEnvironmentManager
from package_path_manager import PackagePathManager
from model_manager import ModelManager
from system_monitor import SystemMonitor

from tokenization_manager import TokenizationManager
from data_manager import DataManager

from transformers import (
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    IntervalStrategy,
    TrainerCallback
)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=False,
)


NET_ID = "vgn2004"
ENV = "qlora"
NUM_WORKERS = 8
MAX_TOKENS = 64
MIN_GENERATION = 64
MODEL_NAME = "facebook/opt-125m"
DATASET_NAME = "NIH_ExPORTER_awarded_grant_text"
TOKENIZER_NAME = "speedup"
BATCH_SIZE = 64

# Constants
OS_ENV_DICT = {
    "CUDA_VISIBLE_DEVICES": 0,
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "true",
    "TORCHDYNAMO_DISABLE": 1,
    "TOKENIZERS_PARALLELISM": "false",
}

if __name__ == "__main__":
    # Clear the GPU
    torch.cuda.empty_cache()
    gc.collect()

    # Configure the logger, needed for initial utilization checks
    LogConfiguration.setup_logging()
    logger = logging.getLogger(__name__)

    # Get initial RAM and GPU utilization
    monitor = SystemMonitor()
    logger.info(f"RAM Usage: {monitor.get_ram_usage()} MB")
    logger.info(f"GPU Utilization: {monitor.get_gpu_utilization()} MB")

    # Configurations

    # Setup folder/file path related configurations
    user_config = UserConfiguration(net_id=NET_ID, env=ENV)
    system_config = SystemConfiguration(num_workers=NUM_WORKERS)
    tokenizer_config = TokenizerConfiguration(
        max_tokens=MAX_TOKENS, tokenizer_name=TOKENIZER_NAME
    )
    torch_config = TorchConfiguration()
    torch_config.commit()

    # System configurations

    # Add Python packages to sys path
    package_path_manager = PackagePathManager(user_config)
    package_path_manager.add_package_paths_to_system()

    # Add environment variables to OS env
    os_env_manager = OSEnvironmentManager()
    os_env_manager.update_from_dict(OS_ENV_DICT)

    # Tokenization
    tokenization_manager = TokenizationManager(user_config, tokenizer_config)
    tokenization_manager.load_for_model(MODEL_NAME)

    # Datasets
    data_manager = DataManager(user_config, system_config, tokenizer_config)
    data_manager.dataset_name = DATASET_NAME
    data_manager.set_data_collator(tokenization_manager.tokenizer)

    # Load from disk
    try:
        (
            training_dataset,
            validation_dataset,
        ) = data_manager.fetch_train_validation_split_from_disk()
    except FileNotFoundError as fe:
        logger.warning(f"{fe.__repr__()}")
        data_manager.create_dataset_from_jsonl_zst_file(
            name=DATASET_NAME,
            jsonl_zst_file_path="/scratch/vgn2004/fine_tuning/datasets/NIH_ExPORTER_awarded_grant_text.jsonl.zst",
        )
        data_manager.create_tokenized_dataset(tokenization_manager.tokenize)
        (
            training_dataset,
            validation_dataset,
        ) = data_manager.fetch_train_validation_split()

    # Dataloaders
    training_dataloader, validation_dataloader = data_manager.fetch_dataloaders(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        batch_size=BATCH_SIZE,
    )

    # Model
    model_manager = ModelManager(system_config)
    model_manager.load(
        MODEL_NAME,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=False,
        ),
    )

    logger.info(model_manager.model)

    # Text Generation
    text_gen_config = TextGenConfiguration(
        tokenization_manager.tokenizer, min_tokens_to_generate=MIN_GENERATION
    )
    prompt = tokenization_manager.encode("This")
    sequence = model_manager.infer(prompt, text_gen_config)
    text = tokenization_manager.decode(sequence, text_gen_config)
    logging.info(f"Generated Text Before Fine-Tuning:\n{text}")

    # Existing Trainer
    from peft import prepare_model_for_kbit_training

    #     model_manager.model.gradient_checkpointing_enable()
    model_manager.model = prepare_model_for_kbit_training(model_manager.model)
    lora_configuration = LoraConfiguration()
    model_manager.lorify(lora_configuration, "qlora")
    logger.info(model_manager.model)

    class SampleTextCallback(TrainerCallback):
        def __init__(
            self, model, tokenizer, output_dir, prompt_text="This", max_length=64
        ):
            self.model = model
            self.tokenizer = tokenizer
            self.output_dir = output_dir
            self.prompt_text = prompt_text
            self.max_length = max_length

        def on_step_begin(self, args, state, control, **kwargs):
            import os
            if state.global_step % 500 == 0 and state.global_step > 0:
                self.model.config.use_cache = True
                self.model.eval()
                input_ids = self.tokenizer.encode(
                    self.prompt_text, return_tensors="pt"
                ).to(self.model.device)
                sample_outputs = self.model.generate(
                    input_ids=input_ids,
                    max_length=self.max_length,
                    num_return_sequences=1,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                self.model.train()
                self.model.config.use_cache = False

                text = f"\n{state.global_step}: {self.tokenizer.decode(sample_outputs[0], skip_special_tokens=True)}"
                print(text)

                sample_file_path = os.path.join(
                    self.output_dir, f"training_samples.txt"
                )
                with open(sample_file_path, "a") as file:
                    file.write(text)

    trainer_callbacks = [
        SampleTextCallback(
            model_manager.model,
            tokenization_manager.tokenizer,
            "/scratch/vgn2004/fine_tuning/standard_slow_generate",
        )
    ]

    trainer = Trainer(
        model=model_manager.model,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        callbacks=trainer_callbacks,
        args=TrainingArguments(
            warmup_steps=5,
            num_train_epochs=50,
            learning_rate=2e-4,
            logging_strategy=IntervalStrategy.STEPS,
            logging_steps=500,
            evaluation_strategy=IntervalStrategy.STEPS,
            eval_steps=1000,
            save_strategy=IntervalStrategy.STEPS,
            save_steps=1000,
            lr_scheduler_type="linear",
            output_dir="/scratch/vgn2004/fine_tuning/standard_slow_generate",
            optim="paged_adamw_32bit",
        ),
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenization_manager.tokenizer,
            mlm=False,
        ),
    )
    model_manager.model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )
    trainer.train()
