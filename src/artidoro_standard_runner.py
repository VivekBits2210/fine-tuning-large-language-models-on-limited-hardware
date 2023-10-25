# classification
# 1. Rewrite basic script with instrumented configurations
# 2. Cut down amount of data, use only the title
# 3. 1.3b model but a modern one (Pythia-1.4b)
# 4. HF trainer
# 5. Pad left
# 6. Search how a recovery mechanism could be implemented

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
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    IntervalStrategy,
    DataCollatorWithPadding,
)

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
        CARED_CONFIGURATIONS = nested_dict_from_flat(
            {k: v for k, v in wandb.config.as_dict().items()}
        )
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

    df = pd.read_csv(
        os.path.join(user_config.cache_path, f"{data_manager.dataset_name}.csv")
    ).sample(frac=CARED_CONFIGURATIONS["keep_fraction"])
    train_df, val_df = train_test_split(df, test_size=0.2)
    train_dataset = MultilabelDataset(
        train_df, tokenization_manager.tokenizer, title_only=True
    )
    val_dataset = MultilabelDataset(
        val_df, tokenization_manager.tokenizer, title_only=True
    )
    training_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=CARED_CONFIGURATIONS["batch_size"],
    )
    validation_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
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
        style="classification",
        num_labels=6,
    )

    # Training
    model_manager.model.config.problem_type = "multi_label_classification"
    model_manager.model.config.pad_token_id = (
        tokenization_manager.tokenizer.pad_token_id
    )
    from peft import prepare_model_for_kbit_training

    model_manager.model = prepare_model_for_kbit_training(model_manager.model)
    lora_configuration = LoraConfiguration()
    model_manager.lorify(lora_configuration, "qlora")
    logger.info(model_manager.model)

    class CheckNanLossCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            self.last_checkpoint = None

        def on_step_end(self, args, state, control, **kwargs):
            # Check if loss is nan
            current_loss = state.log_history[-1].get("loss", None)
            if current_loss is not None and torch.isnan(torch.tensor(current_loss)):
                print("Loss is NaN. Enabling debugging...")
                # Enable debugging or any other required steps

                # Attempt to continue from the last saved checkpoint
                if self.last_checkpoint:
                    print(f"Reloading from checkpoint: {self.last_checkpoint}")
                    control.should_training_stop = True
                    control.should_save = True
                    control.should_evaluate = True
                    state.need_sync = True
                    trainer.train(model_path=self.last_checkpoint)
                else:
                    print("No checkpoint available. Stopping training.")
                    control.should_training_stop = True
            else:
                self.last_checkpoint = trainer.state.best_model_checkpoint

    def compute_metrics(p):
        import numpy as np
        from sklearn.metrics import f1_score, accuracy_score, hamming_loss

        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Compute metrics
        micro_f1 = f1_score(labels, predictions, average="micro")
        hamming = hamming_loss(labels, predictions)
        accuracy = accuracy_score(labels, predictions)

        # Writing metrics to file
        with open(
            "/scratch/vgn2004/fine_tuning/standard_classification/output.txt", "a"
        ) as f:
            f.write(f"Micro F1: {micro_f1}\n")
            f.write(f"Hamming Score: {hamming}\n")
            f.write(f"Accuracy: {accuracy}\n")
            f.write("---------------\n")

        return {
            "micro_f1": micro_f1,
            "hamming_score": hamming,
            "accuracy": accuracy,
        }

    trainer = Trainer(
        model=model_manager.model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        #         callbacks=[CheckNanLossCallback()],
        args=TrainingArguments(
            warmup_steps=5,
            num_train_epochs=5,
            learning_rate=2e-4,
            logging_strategy=IntervalStrategy.STEPS,
            logging_steps=500,
            evaluation_strategy=IntervalStrategy.STEPS,
            eval_steps=1000,
            save_strategy=IntervalStrategy.STEPS,
            save_steps=400,
            lr_scheduler_type="linear",
            output_dir="/scratch/vgn2004/fine_tuning/standard_classification",
            optim="paged_adamw_32bit",
        ),
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenization_manager.tokenizer,
        ),
    )
    trainer.train()
