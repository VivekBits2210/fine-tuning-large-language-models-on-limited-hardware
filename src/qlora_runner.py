import argparse
import logging
import gc
import json
import os
import datetime
import torch
import wandb

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
    GodConfiguration,
)
from managers import (
    OSEnvironmentManager,
    PackagePathManager,
    ModelManager,
    SystemMonitor,
    TokenizationManager,
    DataManager,
)
from trainer import Trainer
from utilities.db_utils import (
    create_tables,
    store_god_configurations_if_not_exists,
    store_cared_configurations,
    generate_run_name,
)

GOD_TAG = "god1"
OS_ENV_DICT = {
    "CUDA_VISIBLE_DEVICES": 0,
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "true",
    "TORCHDYNAMO_DISABLE": 1,
    "TOKENIZERS_PARALLELISM": "false",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument(
        "--config_path",
        type=str,
        required=False,
        default="wandb",
        help="Path to the JSON file containing CARED_CONFIGURATIONS",
    )

    args = parser.parse_args()
    if args.config_path.endswith(".json"):
        with open(args.config_path, "r") as f:
            CARED_CONFIGURATIONS = json.load(f)
    elif args.config_path == "wandb":
        wandb.init(project="qlora_finetuning")
        CARED_CONFIGURATIONS = {k: v for k, v in wandb.config.as_dict().items()}
    else:
        raise Exception("Expected json configuration")

    # Clear the GPU
    torch.cuda.empty_cache()
    gc.collect()

    # Fetch god tag, used to store metrics
    god_config = GodConfiguration(god_tag=GOD_TAG)

    # User configurations
    # Setup folder/file path related configurations
    user_config = UserConfiguration(**CARED_CONFIGURATIONS.get("user_config", {}))

    # Logger setup
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    LogConfiguration.setup_logging(
        os.path.join(user_config.root_path, f"run_log_{timestamp}.log")
    )
    logger = logging.getLogger(__name__)

    # Describe configs used
    logger.info(f"Setting CARED_CONFIGURATIONS as {CARED_CONFIGURATIONS}")

    # Get initial RAM and GPU utilization
    monitor = SystemMonitor()
    logger.info(f"RAM Usage: {monitor.get_ram_usage()} MB")
    logger.info(f"GPU Utilization: {monitor.get_gpu_utilization()} MB")

    # Setup and commit torch configurations
    torch_config = TorchConfiguration(**CARED_CONFIGURATIONS.get("torch_config", {}))
    torch_config.commit()

    # Add Python packages to sys path
    package_path_manager = PackagePathManager(user_config)
    package_path_manager.add_package_paths_to_system()

    # Add environment variables to OS env
    os_env_manager = OSEnvironmentManager()
    os_env_manager.update_from_dict(OS_ENV_DICT)

    # System and tokenizer configurations
    system_config = SystemConfiguration(**CARED_CONFIGURATIONS.get("system_config", {}))
    tokenizer_config = TokenizerConfiguration(
        **CARED_CONFIGURATIONS.get("tokenizer_config", {})
    )

    # Tokenization
    tokenization_manager = TokenizationManager(user_config, tokenizer_config)
    tokenization_manager.load_for_model(CARED_CONFIGURATIONS["model_name"])

    # Setup database
    DB_PATH = os.path.join(user_config.base_dir, user_config.net_id, "metrics.sqlite3")
    create_tables(DB_PATH)
    store_god_configurations_if_not_exists(
        DB_PATH, GOD_TAG, tokenization_manager.tokenizer
    )
    run_name = generate_run_name(
        CARED_CONFIGURATIONS
    )  # using the function from db_utils
    logger.info(f"Starting run name {run_name}...")
    store_cared_configurations(DB_PATH, GOD_TAG, CARED_CONFIGURATIONS)

    # Data management and config
    data_manager = DataManager(user_config, system_config, tokenizer_config)
    data_manager.dataset_name = CARED_CONFIGURATIONS["dataset_name"]
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
            name=data_manager.dataset_name,
            jsonl_zst_file_path=os.path.join(
                user_config.cache_path, f"{data_manager.dataset_name}.jsonl.zst"
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
        batch_size=CARED_CONFIGURATIONS["batch_size"],
    )

    # Quantization
    # TOASS: Is bfloat available?
    quantization_config = QuantizationConfiguration(
        **CARED_CONFIGURATIONS.get("quantization_config", {})
    )

    # Transformer
    # TOASS: Was the model quantized?
    model_manager = ModelManager(system_config)
    model_manager.load(
        CARED_CONFIGURATIONS["model_name"],
        quantization_configuration=quantization_config,
    )

    # LoRA
    # TOASS: Is the rest of the model frozen
    # TOASS: Are the lora weights quantized?
    # TOASS: Are the lora weights updating during fine-tuning?
    lora_configuration = LoraConfiguration(
        **CARED_CONFIGURATIONS.get("lora_config", {})
    )
    model_manager.lorify(lora_configuration, module_style="qlora")
    logger.info(model_manager.model)

    # Text Generation
    text_gen_config = TextGenConfiguration(
        tokenization_manager.tokenizer,
        **CARED_CONFIGURATIONS.get("text_gen_config", {}),
    )
    prompt = tokenization_manager.encode("This")
    sequence = model_manager.infer(prompt, text_gen_config)
    text = tokenization_manager.decode(sequence, text_gen_config)
    logging.info(f"Generated Text Before Fine-Tuning:\n{text}")

    # Training
    train_config = TrainerConfiguration(**CARED_CONFIGURATIONS.get("train_config", {}))
    trainer = Trainer(
        user_config=user_config,
        system_config=system_config,
        tokenizer_config=tokenizer_config,
        text_gen_config=text_gen_config,
        train_config=train_config,
        system_monitor=monitor,
        data_manager=data_manager,
        model_manager=model_manager,
        tokenization_manager=tokenization_manager,
        training_dataloader=training_dataloader,
        validation_dataloader=validation_dataloader,
        database_path=DB_PATH,
        run_name=run_name,
        use_wandb=args.config_path == "wandb",
    )
    trainer.train()
