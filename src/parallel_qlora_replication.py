import argparse
import os
import torch
import gc
import bitsandbytes
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

from transformers import set_seed
from transformers import LlamaForSequenceClassification, LlamaTokenizer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
)
from transformers import BitsAndBytesConfig
from peft import (
    get_peft_model,
    prepare_model_for_kbit_training,
    LoraConfig,
    TaskType,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from psutil import Process
from pynvml import (
    nvmlInit,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetCount,
)


class SystemMonitor:
    def __init__(self):
        # Initialize NVML for GPU monitoring
        self.nvml_initialized = SystemMonitor._initialize_nvml()

    @classmethod
    def _initialize_nvml(cls):
        try:
            nvmlInit()
            return True
        except Exception as e:
            print(f"Error initializing NVML: {e}")
            return False

    def get_ram_usage(self):
        return Process().memory_info().rss / (1024 * 1024)

    def get_gpu_memory_usage(self):
        if not self.nvml_initialized:
            print("NVML not initialized.")
            return None

        gpu_memory_usage = []
        try:
            gpu_count = nvmlDeviceGetCount()
            for i in range(gpu_count):
                handle = nvmlDeviceGetHandleByIndex(i)
                info = nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_usage.append(info.used // 1024**3)
        except Exception as e:
            print(f"Error retrieving GPU memory info: {e}")
            return None

        return gpu_memory_usage

    def get_gpu_utilization(self):
        gpu_memory_usages = self.get_gpu_memory_usage()
        return gpu_memory_usages if gpu_memory_usages is not None else None


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

set_seed(1001)
fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(
        offload_to_cpu=True, rank0_only=False
    ),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

# env_vars = {
#     "CUDA_VISIBLE_DEVICES": "0, 1",
#     "TRANSFORMERS_NO_ADVISORY_WARNINGS": "true",
#     "TORCHDYNAMO_DISABLE": "1",
# }
env_vars = {}


# Configurations
class Configuration:
    def __init__(self, **kwargs):
        self.device_count = torch.cuda.device_count()
        self.experiment_name = kwargs.get("experiment_name", "default_experiment")
        self.keep_fraction = kwargs.get("keep_fraction", 0.99)
        self.test_fraction = kwargs.get("test_fraction", 0.2)
        self.scratch_path = kwargs.get("scratch_path", "/scratch/vgn2004")
        self.dataset_path = kwargs.get(
            "dataset_path",
            os.path.join(
                self.scratch_path, "fine_tuning", "datasets", "disaster_tweets.csv"
            ),
        )
        self.num_workers = kwargs.get("num_workers", 8)
        self.batch_size = kwargs.get("batch_size", 8)
        self.lr = kwargs.get("lr", 3e-4)
        self.num_epochs = kwargs.get("num_epochs", 5)
        self.max_length = kwargs.get("max_length", 128)
        self.device = kwargs.get("device", accelerator.device)
        self.device_map = kwargs.get("device_map", "auto")
        self.max_gpu_memory = kwargs.get("max_gpu_memory", "45000MB")
        # self.device_map = kwargs.get("device_map", {"": accelerator.process_index})

        self.model_name_or_path = kwargs.get(
            "model_name_or_path", "NousResearch/Llama-2-7b-hf"
        )

        self.r = kwargs.get("r", 16)
        self.lora_alpha = kwargs.get("lora_alpha", 64)
        self.lora_dropout = kwargs.get("lora_dropout", 0.2)
        self.lora_bias = kwargs.get("lora_bias", "none")
        self.is_gradient_checkpointing_enabled = kwargs.get(
            "is_gradient_checkpointing_enabled", True
        )
        self.is_gradient_accumulation_enabled = kwargs.get(
            "is_gradient_accumulation_enabled", True
        )
        self.gradient_accumulation_steps = kwargs.get(
            "gradient_accumulation_steps", self.batch_size
        )
        self.batch_size = 1

        self.is_quantized = kwargs.get("is_quantized", True)

    def __str__(self):
        return "\n".join(f"{k}: {v}" for k, v in vars(self).items())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning configuration")
    parser.add_argument("--experiment_name", type=str, default="default_experiment")
    args, unknown = parser.parse_known_args()

    kwargs = vars(args)
    kwargs.update(
        dict((arg[0].lstrip("-"), arg[1]) for arg in zip(unknown[::2], unknown[1::2]))
    )
    print(f"KWARGS: {kwargs}")

    torch.cuda.empty_cache()
    gc.collect()

    os.environ.update(env_vars)

    config = Configuration(**kwargs)
    print(f"Configuration: \n{config}")

    monitor = SystemMonitor()
    print(f"Baseline usage: {monitor.get_gpu_utilization()} GB of GPU")

    if "llama" in config.model_name_or_path.lower():
        tokenizer = LlamaTokenizer.from_pretrained(config.model_name_or_path)
        print(f"Original tokenizer pad token: {tokenizer._pad_token}")
        tokenizer.padding_side = "right"
        tokenizer.model_max_length = config.max_length
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            token="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",
            trust_remote_code=True,
        )
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )

    if config.is_quantized:
        if "llama" in config.model_name_or_path.lower():
            model = LlamaForSequenceClassification.from_pretrained(
                config.model_name_or_path,
                device_map=config.device_map,
                quantization_config=quantization_config,
                max_memory={
                    i: config.max_gpu_memory for i in range(config.device_count)
                },
                trust_remote_code=True,
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                config.model_name_or_path,
                device_map=config.device_map,
                quantization_config=quantization_config,
                token="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",
                trust_remote_code=True,
            )
    else:
        if "llama" in config.model_name_or_path.lower():
            model = LlamaForSequenceClassification.from_pretrained(
                config.model_name_or_path, trust_remote_code=True
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                config.model_name_or_path, trust_remote_code=True
            )

    if config.is_gradient_checkpointing_enabled:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.pretraining_tp = 1
    model.config.torch_dtype = torch.float32
    setattr(model, "model_parallel", True)
    setattr(model, "is_parallelizable", True)

    def find_all_linear_names(m):
        cls = bitsandbytes.nn.Linear4bit
        lora_module_names = set()
        for name, module in m.named_modules():
            if isinstance(module, cls):
                names = name.split(".")
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if "lm_head" in lora_module_names:  # needed for 16-bit
            lora_module_names.remove("lm_head")
        return list(lora_module_names)

    peft_config = LoraConfig(
        target_modules=find_all_linear_names(model),
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=config.is_gradient_checkpointing_enabled
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    print(model.config)

    # Print dtypes
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    dataset = load_dataset("csv", data_files=config.dataset_path)
    dataset = dataset["train"].train_test_split(test_size=config.test_fraction)

    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["text"], max_length=config.max_length, truncation=True
        )
        model_inputs["labels"] = examples["target"]
        return model_inputs

    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=config.num_workers,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    training_dataloader = torch.utils.data.DataLoader(
        processed_datasets["train"],
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding="longest"),
    )
    validation_dataloader = torch.utils.data.DataLoader(
        processed_datasets["test"],
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding="longest"),
    )

    optimizer = (
        torch.optim.AdamW(model.parameters(), lr=config.lr)
        if not config.is_quantized
        else bitsandbytes.optim.AdamW(
            model.parameters(), lr=config.lr, is_paged=True, optim_bits=8
        )
    )
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(training_dataloader) * config.num_epochs),
    )

    # Function to calculate metrics
    def calculate_metrics(preds, labels):
        precision = precision_score(labels, preds, average="macro")
        recall = recall_score(labels, preds, average="macro")
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")
        return precision, recall, accuracy, f1

    # Evaluate a dataloader
    def evaluate(dataloader):
        model.eval()
        all_preds = []
        all_labels = []

        eval_loss = 0.0
        with torch.no_grad():
            for data in tqdm(dataloader):
                batch = {k: v for k, v in data.items()}
                outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()
                preds = torch.argmax(torch.softmax(outputs.logits, dim=1), dim=1)
                labels = batch["labels"]

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        precision, recall, accuracy, f1 = calculate_metrics(all_preds, all_labels)
        return precision, recall, accuracy, f1, eval_loss

    (
        model,
        optimizer,
        training_dataloader,
        validation_dataloader,
        scheduler,
    ) = accelerator.prepare(
        model, optimizer, training_dataloader, validation_dataloader, lr_scheduler
    )

    should_exit = False
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(training_dataloader)):
            if epoch == 0 and step < 5:
                print(f"Usage: {monitor.get_gpu_utilization()} GB of GPU")
            optimizer.zero_grad()
            batch = {k: v for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            if torch.isnan(loss):
                print(f"NaN loss detected at Epoch {epoch}, Step {step}")
                should_exit = True
                break
            total_loss += loss.detach().float()
            loss = loss / config.gradient_accumulation_steps
            accelerator.backward(loss)
            if (step + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        if should_exit:
            break

        model.eval()
        precision_val, recall_val, accuracy_val, f1_val, eval_loss = evaluate(
            validation_dataloader
        )
        print(
            f"Validation Data - Precision: {precision_val}, Recall: {recall_val}, Accuracy: {accuracy_val}, F1: {f1_val}"
        )
        eval_epoch_loss = eval_loss / len(validation_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(training_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(
            f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}"
        )
