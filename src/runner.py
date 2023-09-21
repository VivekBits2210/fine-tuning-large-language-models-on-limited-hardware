import logging

from config import UserConfiguration, LogConfiguration, TorchConfiguration, TokenizerConfiguration, TextGenConfiguration, SystemConfiguration

from os_environment_manager import OSEnvironmentManager
from package_path_manager import PackagePathManager
from model_manager import ModelManager
from inference_manager import InferenceManager
from system_monitor import SystemMonitor

from tokenization_manager import TokenizationManager
from data_manager import DataManager


# TODO: These should be picked up from command line
NET_ID = "vgn2004"
ENV = "pre_prod"
NUM_WORKERS = 8
MAX_TOKENS = 512
MIN_GENERATION = 128
MODEL_NAME = "facebook/opt-125m"
DATASET_NAME = "NIH_ExPORTER_awarded_grant_text"
BATCH_SIZE = 64

# Constants
OS_ENV_DICT = {
    "CUDA_VISIBLE_DEVICES": 0,
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "true",
    "TORCHDYNAMO_DISABLE": 1
}

if __name__ == "__main__":
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
    tokenizer_config = TokenizerConfiguration(max_tokens=MAX_TOKENS)
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

    # Tokenize dataset from scratch (skipped)
#     data_manager.create_dataset_from_jsonl_zst_file(name=DATASET_NAME,
#                                                     jsonl_zst_file_path="E:\\NIH_ExPORTER_awarded_grant_text.jsonl.zst")
#     data_manager.create_tokenized_dataset()
#     training_dataset, validation_dataset = data_manager.fetch_train_validation_split()

    # Load from disk
    training_dataset, validation_dataset = data_manager.fetch_train_validation_split_from_disk()

    # Dataloaders
    training_dataloader, validation_dataloader = data_manager.fetch_dataloaders(
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
        batch_size=BATCH_SIZE
    )

    # Model
    model_manager = ModelManager(system_config)
    model_manager.load(MODEL_NAME)
    logger.info(model_manager.model)

    # Text Generation
    text_gen_config = TextGenConfiguration(tokenization_manager.tokenizer, min_tokens_to_generate=MIN_GENERATION)

    inference_manager = InferenceManager(text_gen_config)

    # TODO: The below process should involve tokenization and decode by tokenization_manager, rest by model_manager
    generated_text = inference_manager.infer(model_manager.model, tokenization_manager.tokenizer, system_config.device)
    logging.info(f"Initial Generation:\n{generated_text}")
