import sys
import logging
import gc
import json
import os
import datetime
import torch
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

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
from managers.dataset_classes import MultilabelDataset

GOD_TAG = "god1"
OS_ENV_DICT = {
    "CUDA_VISIBLE_DEVICES": 0,
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "true",
    "TORCHDYNAMO_DISABLE": 1,
    "TOKENIZERS_PARALLELISM": "false",
}


def nested_dict_from_flat(flat_dict):
    nested_dict = {}
    for key, value in flat_dict.items():
        keys = key.split(".")
        d = nested_dict
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return nested_dict


def parse_args(argv):
    args = {}
    current_key = None
    for token in argv[1:]:
        if token.startswith("--"):
            if current_key is not None:
                args[current_key] = None
            current_key = token[2:]
        else:
            args[current_key] = token
            current_key = None
    if current_key is not None:
        args[current_key] = None
    return args


if __name__ == "__main__":
    args = parse_args(sys.argv)
    print(f"ARGS={args}")

    config_path = args.get("config_path", "")
    if config_path.endswith(".json"):
        with open(config_path, "r") as f:
            CARED_CONFIGURATIONS = json.load(f)
    else:
        config_path = "wandb"
        wandb.init(project="qlora_finetuning")
        CARED_CONFIGURATIONS = nested_dict_from_flat({k: v for k, v in wandb.config.as_dict().items()})
        logging.info(f"Using CARED_CONFIGURATIONS AS: {CARED_CONFIGURATIONS}")

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

    df = pd.read_csv(os.path.join(user_config.cache_path, f"{data_manager.dataset_name}.csv")) #.sample(frac=0.2)
    train_df, val_df = train_test_split(df, test_size=0.2)
    train_dataset = MultilabelDataset(train_df, tokenization_manager.tokenizer)
    val_dataset = MultilabelDataset(val_df, tokenization_manager.tokenizer)
    training_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=CARED_CONFIGURATIONS["batch_size"]
    )
    validation_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=CARED_CONFIGURATIONS["batch_size"]
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
        style="classification",
        num_labels=6
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
    
    # Training
    model_manager.model.config.problem_type = "multi_label_classification"
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
        use_wandb=config_path == "wandb",
        task="classification"
    )
    trainer.train()
