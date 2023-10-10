import logging
import gc
import os
import torch

from config import (
    UserConfiguration,
    LogConfiguration,
    TorchConfiguration,
    TokenizerConfiguration,
    TextGenConfiguration,
    SystemConfiguration,
    TrainerConfiguration,
    LoraConfiguration,
    QuantizationConfiguration,
)
from managers import OSEnvironmentManager
from managers import PackagePathManager
from managers import ModelManager
from managers import SystemMonitor
from managers import TokenizationManager
from managers import DataManager
from trainer import Trainer

# TODO: Every run has a set of configurations that is "cared" for. These are stored separatly while logging
# TODO: Collate all configurations and store as 1 config. If the "cared" for config doesn't contain a config,
#  default to the configuration in the collation
# TODO: Only 2 config columns are needed - "cared" and "all"
# Note: What defines a run? A concatenation of the cared configurations and their values
# The name of the run should also be the name of the environment where the checkpoints are stored
ENV = "qlora_simplified"
MODEL_NAME = "facebook/opt-125m"
DATASET_NAME = "NIH_ExPORTER_awarded_grant_text"
TOKENIZER_NAME = "speedup"
BATCH_SIZE = 64
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

    # Logger setup
    LogConfiguration.setup_logging()
    logger = logging.getLogger(__name__)

    # Get initial RAM and GPU utilization
    monitor = SystemMonitor()
    logger.info(f"RAM Usage: {monitor.get_ram_usage()} MB")
    logger.info(f"GPU Utilization: {monitor.get_gpu_utilization()} MB")

    # Setup folder/file path related configurations
    user_config = UserConfiguration(env=ENV)
    system_config = SystemConfiguration()
    tokenizer_config = TokenizerConfiguration(tokenizer_name=TOKENIZER_NAME)

    # Setup and commit torch configurations
    torch_config = TorchConfiguration()
    torch_config.commit()

    # Add Python packages to sys path
    package_path_manager = PackagePathManager(user_config)
    package_path_manager.add_package_paths_to_system()

    # Add environment variables to OS env
    os_env_manager = OSEnvironmentManager()
    os_env_manager.update_from_dict(OS_ENV_DICT)

    # Tokenization
    tokenization_manager = TokenizationManager(user_config, tokenizer_config)
    tokenization_manager.load_for_model(MODEL_NAME)

    # Data management and config
    data_manager = DataManager(user_config, system_config, tokenizer_config)
    data_manager.dataset_name = DATASET_NAME
    data_manager.set_data_collator(tokenization_manager.tokenizer)

    # Fetch dataset
    try:
        (
            training_dataset,
            validation_dataset,
        ) = data_manager.fetch_train_validation_split_from_disk()
    except FileNotFoundError as fe:
        logger.warning(f"{fe.__repr__()}")
        data_manager.create_dataset_from_jsonl_zst_file(
            name=DATASET_NAME,
            jsonl_zst_file_path=os.path.join(
                user_config.cache_path, f"{DATASET_NAME}.jsonl.zst"
            ),
        )
        data_manager.create_tokenized_dataset(tokenization_manager.tokenize)
        (
            training_dataset,
            validation_dataset,
        ) = data_manager.fetch_train_validation_split()

    # Data loaders
    # TOASS: What do the snippets look like? Is the size of a snippet less than MAX_TOKENS?
    training_dataloader, validation_dataloader = data_manager.fetch_dataloaders(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        batch_size=BATCH_SIZE,
    )

    # Quantization
    # TOASS: Is bfloat available?
    quantization_config = QuantizationConfiguration()

    # Transformer
    # TOASS: Was the model quantized?
    model_manager = ModelManager(system_config)
    model_manager.load(MODEL_NAME, quantization_configuration=quantization_config)

    # LoRA
    # TOASS: Which modules did the lora configuration apply to?
    # TOASS: Is the rest of the model frozen
    # TOASS: Are the lora weights quantized?
    # TOASS: Are the lora weights updating during fine-tuning?
    lora_configuration = LoraConfiguration()
    model_manager.lorify(lora_configuration, module_style="qlora")
    logger.info(model_manager.model)

    # Text Generation
    text_gen_config = TextGenConfiguration(tokenization_manager.tokenizer)
    prompt = tokenization_manager.encode("This")
    sequence = model_manager.infer(prompt, text_gen_config)
    text = tokenization_manager.decode(sequence, text_gen_config)
    logging.info(f"Generated Text Before Fine-Tuning:\n{text}")

    # Training
    train_config = TrainerConfiguration()
    trainer = Trainer(
        user_config=user_config,
        system_config=system_config,
        tokenizer_config=tokenizer_config,
        text_gen_config=text_gen_config,
        train_config=train_config,
        data_manager=data_manager,
        model_manager=model_manager,
        tokenization_manager=tokenization_manager,
        training_dataloader=training_dataloader,
        validation_dataloader=validation_dataloader,
    )
    trainer.train()
