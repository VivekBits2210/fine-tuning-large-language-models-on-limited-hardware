import argparse
import os
import torch
import gc
import bitsandbytes
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

from transformers import set_seed
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
)
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
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


monitor = SystemMonitor()
print(f"Baseline usage: {monitor.get_gpu_utilization()} GB of GPU")


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


class Configuration:
    def __init__(self, **kwargs):
        self.device_count = torch.cuda.device_count()
        self.experiment_name = kwargs.get("experiment_name", "default_experiment")
        self.keep_fraction = kwargs.get("keep_fraction", 0.99)
        self.test_fraction = kwargs.get("test_fraction", 0.2)
        self.scratch_path = kwargs.get("scratch_path", "/scratch/vgn2004")
        self.num_workers = kwargs.get("num_workers", 8)
        self.batch_size = kwargs.get("batch_size", 16)
        self.lr = kwargs.get("lr", 3e-4)
        self.num_epochs = kwargs.get("num_epochs", 5)
        self.seq_length = kwargs.get("seq_length", 1024)
        self.device = kwargs.get("device", accelerator.device)
        self.device_map = kwargs.get("device_map", "auto")
        self.max_gpu_memory = kwargs.get("max_gpu_memory", "45080MB")
        # self.device_map = kwargs.get("device_map", {"": accelerator.process_index})

        self.model_name_or_path = kwargs.get(
            "model_name_or_path",
            "togethercomputer/LLaMA-2-7B-32K",  # "NousResearch/Llama-2-7b-chat-hf"
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

        self.is_quantized = kwargs.get("is_quantized", True)

    def __str__(self):
        return "\n".join(f"{k}: {v}" for k, v in vars(self).items())


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

# os.environ.update(env_vars)

config = Configuration(**kwargs)
print(f"Configuration: \n{config}")

INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
INSTRUCTION_KEY = "### Instruction:"
INPUT_KEY = "Input:"
RESPONSE_KEY = "### Response:"
END_KEY = "### End"
RESPONSE_KEY_NL = f"{RESPONSE_KEY}\n"
DEFAULT_SEED = 42

# This is a training prompt that does not contain an input string.  The instruction by itself has enough information
# to respond.  For example, the instruction might ask for the year a historic figure was born.
PROMPT_NO_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{response_key}
{response}

{end_key}""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)

# This is a training prompt that contains an input string that serves as context for the instruction.  For example,
# the input might be a passage from Wikipedia and the intruction is to extract some information from it.
PROMPT_WITH_INPUT_FORMAT = """{intro}

{instruction_key}
{instruction}

{input_key}
{input}

{response_key}
{response}

{end_key}""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    input_key=INPUT_KEY,
    input="{input}",
    response_key=RESPONSE_KEY,
    response="{response}",
    end_key=END_KEY,
)

# This is the prompt that is used for generating responses using an already trained model.  It ends with the response
# key, where the job of the model is to provide the completion that follows it (i.e. the response itself).
PROMPT_FOR_GENERATION_FORMAT = """{intro}

{instruction_key}
{instruction}

{response_key}
""".format(
    intro=INTRO_BLURB,
    instruction_key=INSTRUCTION_KEY,
    instruction="{instruction}",
    response_key=RESPONSE_KEY,
)

tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
tokenizer.model_max_length = config.seq_length
tokenizer.padding_side = "right"
tokenizer.pad_token, tokenizer.eos_token
tokenizer.add_special_tokens(
    {"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL]}
)

# Find max allowed sequence length
model_config = AutoConfig.from_pretrained(config.model_name_or_path)
max_length = None
for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
    max_length = getattr(model_config, length_setting, None)
    if max_length:
        print(f"Found max lenth: {max_length}")
        break
if not max_length:
    max_length = 1024
    print(f"Using default max length: {max_length}")

model_config.max_position_embeddings = config.seq_length
model_config.bos_token_id = tokenizer.bos_token_id
model_config.eos_token_id = tokenizer.eos_token_id
model_config.pad_token_id = tokenizer.pad_token_id


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    config.model_name_or_path,
    config=model_config,
    device_map=config.device_map,
    quantization_config=quantization_config,
    max_memory={i: config.max_gpu_memory for i in range(config.device_count)},
    trust_remote_code=False,
)

model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
if config.is_gradient_checkpointing_enabled:
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

# Model settings
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

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


peft_config = LoraConfig(
    target_modules=find_all_linear_names(model),
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=config.r,
    lora_alpha=config.lora_alpha,
    lora_dropout=config.lora_dropout,
    bias=config.lora_bias,
)

model = prepare_model_for_kbit_training(
    model, use_gradient_checkpointing=config.is_gradient_checkpointing_enabled
)

from functools import partial


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        batch = super().torch_call(examples)

        # The prompt ends with the response key plus a newline.  We encode this and then try to find it in the
        # sequence of tokens.  This should just be a single token.
        response_token_ids = self.tokenizer.encode(RESPONSE_KEY_NL)

        labels = batch["labels"].clone()

        for i in range(len(examples)):
            response_token_ids_start_idx = None
            for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                response_token_ids_start_idx = idx
                break

            if response_token_ids_start_idx is None:
                raise RuntimeError(
                    f'Could not find response key {response_token_ids} in token IDs {batch["labels"][i]}'
                )

            response_token_ids_end_idx = response_token_ids_start_idx + 1
            labels[i, :response_token_ids_end_idx] = -100

        batch["labels"] = labels

        return batch


def preprocess_batch(batch, tokenizer, max_length):
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


def preprocess_dataset(tokenizer, max_length, dataset):
    _preprocessing_function = partial(
        preprocess_batch, max_length=max_length, tokenizer=tokenizer
    )
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["instruction", "context", "response", "text", "category"],
    )

    # Make sure we don't have any truncated records, as this would mean the end keyword is missing.
    print("Processed dataset has %d rows", dataset.num_rows)
    dataset = dataset.filter(lambda rec: len(rec["input_ids"]) < max_length)
    print(
        "Processed dataset has %d rows after filtering for truncated records",
        dataset.num_rows,
    )

    dataset = dataset.shuffle()
    return dataset


dataset = load_dataset("databricks/databricks-dolly-15k")


def _add_text(rec):
    instruction = rec["instruction"]
    response = rec["response"]
    context = rec.get("context")

    if not instruction:
        raise ValueError(f"Expected an instruction in: {rec}")

    if not response:
        raise ValueError(f"Expected a response in: {rec}")

    if context:
        rec["text"] = PROMPT_WITH_INPUT_FORMAT.format(
            instruction=instruction, response=response, input=context
        )
    else:
        rec["text"] = PROMPT_NO_INPUT_FORMAT.format(
            instruction=instruction, response=response
        )
    return rec


dataset = dataset.map(_add_text)
processed_dataset = preprocess_dataset(
    tokenizer=tokenizer, max_length=config.seq_length, dataset=dataset
)
split_dataset = processed_dataset["train"].train_test_split(test_size=0.2)
print("Train data size: ", split_dataset["train"].num_rows)
print("Test data size: ", split_dataset["test"].num_rows)

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

data_collator = DataCollatorForCompletionOnlyLM(
    tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
)
training_dataloader = torch.utils.data.DataLoader(
    split_dataset["train"],
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    collate_fn=data_collator,
)
validation_dataloader = torch.utils.data.DataLoader(
    split_dataset["test"],
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    collate_fn=data_collator,
)


optimizer = (
    torch.optim.AdamW(model.parameters(), lr=config.lr)
    if not config.is_quantized
    else bitsandbytes.optim.AdamW(
        model.parameters(), lr=config.lr, is_paged=True, optim_bits=32
    )
)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(training_dataloader) * config.num_epochs),
)

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
        #         loss = loss / config.gradient_accumulation_steps
        accelerator.backward(loss)
        #         if (step + 1) % config.gradient_accumulation_steps == 0:
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
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")
