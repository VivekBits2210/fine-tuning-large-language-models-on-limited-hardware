# Change net ID here to use your scratch folder
ENV = "prod"
NET_ID = "vgn2004"
DATA_PATH =  f"/scratch/{NET_ID}/fine_tuning" 
ROOT_PATH = f"/scratch/{NET_ID}/fine_tuning/{ENV}"

# Global configurations
config = {
    "DATASET_URL": "https://the-eye.eu/public/AI/pile_v2/data",
    "DATASET_NAME": "NIH_ExPORTER_awarded_grant_text",
    "NUM_WORKERS": 8,
    "DATASET_SPLIT_RATIO": 0.9,
    "PADDING_STRATEGY": "max_length",
    "MAX_TOKENS": 512,
    "MIN_GENERATION": 512,
    "MODEL_NAME": "facebook/opt-125m",
    "TOKENIZED_NAME": "opt_2700m_512",
    "BATCH_SIZE": 64,
    "NUM_EPOCHS": 8,
    "LEARNING_RATE": 5e-4,
    "MIN_LEARNING_RATE": 5e-5,
    "EPSILON": 1e-8,
    "BETAS": (0.9,0.95),
    "GRADIENT_CLIP": 1.0,
    "WEIGHT_DECAY": 0.01,
    "DECAY_STYLE": "cosine", #not used currently
    "WARMUP_RATIO": 0.003,
    "SAMPLING_INTERVAL": 20,
    "CHECKPOINTING_INTERVAL": 100,
    "VALIDATION_INTERVAL": 500,
    "GRADIENT_ACCUMULATION_STEPS": 4, #TODO: need to bring this back
    
    "DYNAMIC_LR": True,
    "PEFT": False,
}

from peft import LoraConfig, PeftConfig, get_peft_model 
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.2,
    bias="none",
    task_type="CAUSAL_LM"
)

# Ensure that packages can be found
import sys
sys.path.insert(0, f"/home/{NET_ID}/.local/lib/python3.8/site-packages")

# Ensure that GPU can be found
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
# os.environ["TORCHDYNAMO_DISABLE"] = "1"

# Setup logging
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')

# Packages for profiling
import inspect
import math
import random
import psutil
from time import time
from tqdm import tqdm
import tqdm.notebook as tq
from pynvml import *

# Packages for data loading
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Core packages
import torch
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(False)
logging.info(f"Is Flash Attention Enabled: {torch.backends.cuda.flash_sdp_enabled()}")
logging.info(f"Is Mem Efficient SDP Enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
logging.info(f"Is Math SDP Enabled: {torch.backends.cuda.math_sdp_enabled()}")


from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup, BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_skip_modules=["lm_head"],
    llm_int8_threshold=3.0
)
from transformers.optimization import Adafactor
import bitsandbytes.optim as bnb_optim


# Get GPU Utilization
def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    logging.info(f"GPU memory occupied: {info.used//1024**2} MB.")
    

# Returns RAM usage in MB
def get_ram_usage():
    return psutil.Process().memory_info().rss / (1024 * 1024)

# Returns number of trainable parameters and percentage
def print_trainable_parameters(model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        logger.info(
            f"Parameters: Trainable- {trainable_params/1e6:.2f}M|| All- {all_param/1e6:.2f}M || Trainable%- {100 * trainable_params / all_param}"
        )

#Takes a batch of inputs and runs the tokenizer on them
def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding=config["PADDING_STRATEGY"],
        truncation=True,
        max_length=config["MAX_TOKENS"],
        return_attention_mask=True
    )

# Tokenizes dataset and creates train and validation split
def preprocess_data(dataset, tokenizer):
    tokenized_dataset_path = f"{DATA_PATH}/datasets/tokenized_{config['DATASET_NAME']}_{config['TOKENIZED_NAME']}"
    train_dataset_path = f"{tokenized_dataset_path}_train"
    valid_dataset_path = f"{tokenized_dataset_path}_valid"
    if os.path.exists(train_dataset_path) and os.path.exists(valid_dataset_path):
        logger.info(f"Loading dataset from disk...")
        start_time = time()
        train_dataset = load_from_disk(train_dataset_path)
        valid_dataset = load_from_disk(valid_dataset_path)
        elapsed_time = time() - start_time
        logger.info(f"Time taken to load dataset from : {elapsed_time:.2f} seconds")
        return train_dataset, valid_dataset
        
    logger.info(f"Tokenizing the dataset...")
    start_time = time()
    try:
        tokenized_dataset = load_from_disk(tokenized_dataset_path)
    except Exception as e:
        logging.error(e)
        tokenized_dataset = dataset.map(
            tokenize_function,
            fn_kwargs={'tokenizer': tokenizer},
            batched=True,
            num_proc=8,
            remove_columns=["text", "meta"],
        )
        tokenized_dataset.save_to_disk(tokenized_dataset_path)

    elapsed_time = time() - start_time
    logger.info(f"Time taken to tokenize the dataset: {elapsed_time:.2f} seconds")

    logger.info(f"Splitting the dataset...")
    start_time = time()
    
    if os.path.exists(train_dataset_path) and os.path.exists(valid_dataset_path):
        train_dataset = load_from_disk(train_dataset_path)
        valid_dataset = load_from_disk(valid_dataset_path)
    else:
        train_size = int(config["DATASET_SPLIT_RATIO"] * len(tokenized_dataset))
        datasets = DatasetDict({
            'train': Dataset.from_dict(tokenized_dataset[:train_size]),
            'valid': Dataset.from_dict(tokenized_dataset[train_size:])
        })
        train_dataset = datasets['train']
        valid_dataset = datasets['valid']
        train_dataset.save_to_disk(train_dataset_path)
        valid_dataset.save_to_disk(valid_dataset_path)
    elapsed_time = time() - start_time
    logger.info(f"Time taken to split the datasets (or load pre-split datasets): {elapsed_time:.2f} seconds")
    
    return train_dataset, valid_dataset

# Creates data loaders
def create_dataloaders(train_dataset, valid_dataset, data_collator):
    logger.info(f"Creating data loaders...")
    start_time = time()
    train_dataloader = DataLoader(train_dataset,
                                  sampler=RandomSampler(train_dataset),
                                  batch_size=config["BATCH_SIZE"],
                                  num_workers=config["NUM_WORKERS"],
                                  collate_fn=data_collator,
                                  pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset,
                                  sampler=SequentialSampler(valid_dataset),
                                  batch_size=config["BATCH_SIZE"],
                                  num_workers=config["NUM_WORKERS"],
                                  collate_fn=data_collator,
                                  pin_memory=True)
    elapsed_time = time() - start_time
    logging.info(f"Time taken to create data loaders: {elapsed_time:.2f} seconds")
    return train_dataloader, valid_dataloader

# Fetches tokenizer relevant to the model
def create_or_load_tokenizer(checkpointed_path=None):
    if checkpointed_path:
        tokenizer = AutoTokenizer.from_pretrained(checkpointed_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config["MODEL_NAME"], cache_dir=f"{DATA_PATH}/datasets")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
    return tokenizer

# Data preparation
def run_data_pipeline(tokenizer, load_from_file=False):
    # Measure how much RAM is being used before anything runs
    ram_usage = get_ram_usage()
    logging.info(f"Baseline: RAM used: {ram_usage:.2f} MB")

    # Load data, either from url or from datasets folder
    data_file_url = f"{config['DATASET_URL']}/{config['DATASET_NAME']}.jsonl.zst"
    try:
        if load_from_file:
            raise Exception
        dataset = load_dataset("json",
                               data_files=data_file_url,
                               num_proc=config["NUM_WORKERS"],
                               split="train",
                               cache_dir=f"{DATA_PATH}/datasets")
    except Exception as e:
        logging.error(e)
        dataset = load_dataset("json",
                               data_files=f"{DATA_PATH}/datasets/{config['DATASET_NAME']}.jsonl.zst",
                               num_proc=config["NUM_WORKERS"],
                               split="train",
                               cache_dir=f"{DATA_PATH}/datasets")

    # Measurements relevant to the dataset
    ram_usage = get_ram_usage()
    logging.info(f"RAM used: {ram_usage:.2f} MB")
    logging.info(f"Dataset sample: {dataset[10]}")
    size_gb = dataset.dataset_size / (1024 ** 3)
    logging.info(f"Dataset size (cache file) : {size_gb:.2f} GB")

    # Fetch a tokenizer and tokenize + split the dataset
    train_dataset, valid_dataset = preprocess_data(dataset, tokenizer)

    # Create a data collator and use it to make data loaders
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataloader, valid_dataloader = create_dataloaders(train_dataset, valid_dataset, data_collator)

    return {
        "TRAIN_DATASET": train_dataset,
        "VALIDATION_DATASET": valid_dataset,
        "TRAIN_DATALOADER": train_dataloader,
        "VALIDATION_DATALOADER": valid_dataloader,
        "TOKENIZER": tokenizer
    }

#Get optimizer
def fetch_optimizer(model):
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
    
    head_layers = set(['lm_head.weight', '_orig_mod.lm_head.weight', 'base_model.model.lm_head.0.weight'])
    decay = set([d for d in decay if d not in head_layers])

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": config["WEIGHT_DECAY"]},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    
    if(config["PEFT"]):
        optimizer = bnb_optim.AdamW(optim_groups, lr=config["LEARNING_RATE"], betas=config["BETAS"], weight_decay=config["WEIGHT_DECAY"], optim_bits=8)
        manager = bnb_optim.GlobalOptimManager.get_instance()

        skipped = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Embedding):
                skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                manager.register_module_override(module, "weight", {"optim_bits": 32})
                logger.info(f"bitsandbytes: will optimize {module} in fp32")
        logger.info(f"Quantizing: Skipped: {skipped/2**20}M params")
    else:
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster, only works for floating point values
        use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        logger.info(f"Using fused AdamW: {use_fused}")
        fused_arg_dict = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=config["LEARNING_RATE"], betas=config["BETAS"], weight_decay=config["WEIGHT_DECAY"], **fused_arg_dict)

    return optimizer

# Get learning rate per iteration
def get_lr(it, max_iters):
    warmup_iters = int(config["WARMUP_RATIO"]*max_iters)
    if it < warmup_iters:
        return config["LEARNING_RATE"] * it / warmup_iters
    if it > max_iters:
        return config["MIN_LEARNING_RATE"]
    
    #Cosine decay after warmup phase is over
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config["MIN_LEARNING_RATE"] + coeff * (config["LEARNING_RATE"] - config["MIN_LEARNING_RATE"])


# Create model
def create_or_load_model(checkpointed_path=None, quantized=config["PEFT"], frozen=False, cast_layer_norm_to_fp32=False, cast_output_to_fp32=False):
    class CastOutputToFloat(torch.nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.float32)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpointed_path:
        model = AutoModelForCausalLM.from_pretrained(checkpointed_path)
        model.to(device)
    else:
        configuration = AutoConfig.from_pretrained(config["MODEL_NAME"])
        
        if quantized:
             model = AutoModelForCausalLM.from_pretrained(config["MODEL_NAME"], config=configuration, load_in_8bit=True, device_map='auto', quantization_config=quantization_config)
        else:
            model = AutoModelForCausalLM.from_pretrained(config["MODEL_NAME"], config=configuration)
            model.to(device)
            
        if frozen:
            for param in model.parameters():
                param.requires_grad = False
                
        if cast_layer_norm_to_fp32:
            for param in model.parameters():
                if param.ndim == 1:
                    param.data = param.data.to(torch.float32)
                
    #Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    if cast_output_to_fp32:
        model.lm_head = CastOutputToFloat(model.lm_head)
    
    # Log details
    logger.info(f"Model: {config['MODEL_NAME']}")
    print_trainable_parameters(model)
    logger.info(f"Memory Memory Footprint: {model.get_memory_footprint() / 1e6:,} MB")
    logger.info(f"Model is on device: {model.device}")
    
    model.config.use_cache = False
    return model, device

# Use the model to generate text
def generate(model, inputs):
    output_sequence = model.generate(
        **inputs,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        min_length=config["MIN_GENERATION"],
        max_length=2*config["MIN_GENERATION"],
        top_p=0.95,
        num_return_sequences=1
    )
    return output_sequence
    
def inference(model, tokenizer, device, quantized=config["PEFT"]):
    # Put the model in eval mode and enable caching
    model.config.use_cache = True
    model.eval()
    
    inputs = tokenizer(tokenizer.eos_token+"This is", return_tensors="pt").to(device)
    # Generate a sequence of text tokens
    with torch.no_grad():
        if quantized:
            with torch.cuda.amp.autocast():
                output_sequence = generate(model, inputs)
        else:
            output_sequence = generate(model, inputs)
        

    # Decode the tokens to text
    generated_text = tokenizer.decode(output_sequence[0], 
                                      skip_special_tokens=True).replace('\n', '').replace('\t', ' ')

    # Put the model back into train mode and disable caching
    model.train()
    model.config.use_cache = False
    
    return generated_text

# Evaluate the model on a data loader
def validate(model, device, valid_dataloader):
    model.eval()
    total_eval_loss = 0.0
    counter = 0
    for index, batch in tqdm(enumerate(valid_dataloader,1)):
        if counter<5:
                print_gpu_utilization()
                counter+=1
        batch = {k: v.pin_memory().to(device, non_blocking=True) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
        total_eval_loss += loss.item()
        avg_eval_loss = total_eval_loss / index
        logging.info(f"Validation: Batch {index}/{len(valid_dataloader)}, Loss: {avg_eval_loss:.4f}")

    perplexity = torch.exp(torch.as_tensor(avg_eval_loss)).item()
    model.train()
    return avg_eval_loss, perplexity

# Train the model
def train(model, device, data_dict, start_epoch=1, start_iteration_number=0):
    folder = f"fine_tuned_{config['MODEL_NAME']}_{config['DATASET_NAME']}_{config['TOKENIZED_NAME']}"
    model_save_path = f"{ROOT_PATH}/models/{folder}"
    
    # Setup logging
    log_save_path = f"{ROOT_PATH}/logs/{folder}"
    if not os.path.exists(log_save_path):
        os.makedirs(log_save_path)
    with open(f"{log_save_path}/training.log","w+") as f:
        f.write("epoch\tbatch\ttrain\tloss\tgenerated_text\n")
    with open(f"{log_save_path}/validation.log","w+") as f:
        f.write("epoch\tbatch\tvalidation_loss\tperplexity\n")

    train_dataloader = data_dict["TRAIN_DATALOADER"]
    valid_dataloader = data_dict["VALIDATION_DATALOADER"]
    tokenizer = data_dict["TOKENIZER"]

    # Scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Early stopping
    patience = 5
    min_loss = float("inf")
    epochs_since_min_loss = 0

    
    if config["DYNAMIC_LR"]:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            optimizer = fetch_optimizer(model)
    else:
        optimizer = Adafactor(model.parameters(), lr=config["LEARNING_RATE"], scale_parameter=False, relative_step=False, warmup_init=False)
    model.train()
    
    max_iters = len(train_dataloader)*config["NUM_EPOCHS"] 
    learning_rate = config["LEARNING_RATE"]
    
    # Go through each epoch
    iteration_number = start_iteration_number
    for epoch in tqdm(range(start_epoch,config["NUM_EPOCHS"]+1)):
        iteration_number_per_epoch = 0
        running_loss = 0.0
        logging.info(f"Epoch: {epoch}/{config['NUM_EPOCHS']}")

        #Go through each batch in the data loader
        for index, batch in tqdm(enumerate(train_dataloader, 1), total=len(train_dataloader)):
            iteration_number+=1
            optimizer.zero_grad(set_to_none=True)
            
            # For the initial warmup phase, keep an eye on the GPU utilization
            if iteration_number_per_epoch<5:
                print_gpu_utilization()
                iteration_number_per_epoch+=1

            # Sample an output from the model, at each sampling interval
            if index%config["SAMPLING_INTERVAL"]==0:
                generated_text = inference(model, tokenizer, device)
                logging.info(f"Text:\n{generated_text}")
                
                with open(f"{log_save_path}/training.log", "a") as f:
                    f.write(f"{epoch}\t{index}\t{avg_loss}\t{generated_text}\n")

            #Save the model at each checkpointing interval
            if index%config["CHECKPOINTING_INTERVAL"]==0:
                logging.info(f"Checkpointing model at epoch={epoch} and batch={index}\n")

                checkpointing_path = f"{model_save_path}_{epoch}_{index}"
                model.save_pretrained(checkpointing_path)
                tokenizer.save_pretrained(checkpointing_path)

            #Validate the model at each validation interval
            if index%config["VALIDATION_INTERVAL"]==0:
                logging.info("Running Validation...")
                avg_eval_loss, perplexity = validate(model, device, valid_dataloader)
                logging.info(f"Batch {index}/{len(train_dataloader)}, Validation Loss: {avg_eval_loss:.4f}, Perplexity: {perplexity:.2f}")
                with open(f"{log_save_path}/validation.log", "a") as f:
                    f.write(f"{epoch}\t{index}\t{avg_eval_loss}\t{perplexity}\n")

            #Load batches in a non-blocking manner
            batch = {k: v.pin_memory().to(device, non_blocking=True) for k, v in batch.items()}
            
            #Get dynamic learning rate using cosine decay and warmup
            if config["DYNAMIC_LR"]:                
                learning_rate = get_lr(iteration_number, max_iters)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
                    
            #Forward pass using mixed precision training
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / config["GRADIENT_ACCUMULATION_STEPS"]

            # Log the loss
            running_loss += (loss.item()*config["GRADIENT_ACCUMULATION_STEPS"])
            avg_loss = running_loss / index
            logging.info(f"Batch {index}/{len(train_dataloader)}, Loss: {avg_loss:.4f}, Learning Rate: {learning_rate}")

            # Backward pass
            scaler.scale(loss).backward()

            if index % config["GRADIENT_ACCUMULATION_STEPS"] == 0:
                # Gradient clipping mechanism
                if "GRADIENT_CLIP" in config:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["GRADIENT_CLIP"])
                    scaler.step(optimizer)
                scaler.update()

                
        # After each epoch, check if the training loss has improved
        if avg_loss < min_loss:
            min_loss = avg_loss
            epochs_since_min_loss = 0
        else:
            epochs_since_min_loss += 1

        # Early stopping mechanism
        if epochs_since_min_loss >= patience:
            logger.info("Early stopping triggered. No improvement in training loss for {} epochs.".format(patience))
            break

    #After all epochs are completed, save the final model and tokenier
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

if __name__=="__main__":
    print_gpu_utilization()
    
    checkpointed_path = None
    tokenizer = create_or_load_tokenizer(checkpointed_path=checkpointed_path)
    data_dict = run_data_pipeline(tokenizer, load_from_file=True)
    
    if config["PEFT"]:
        model, device = create_or_load_model(checkpointed_path=checkpointed_path, 
                                             frozen=True,
                                             cast_layer_norm_to_fp32=True,
                                             cast_output_to_fp32=True)
        model = get_peft_model(model, lora_config)
        model.to(device)
        logger.info(f"Peft Model: {config['MODEL_NAME']}")
        print_trainable_parameters(model)
        print_trainable_parameters(model)
        logger.info(f"Memory Memory Footprint: {model.get_memory_footprint() / 1e6:,} MB")
        logger.info(f"Model is on device: {model.device}")
    else:
        model, device = create_or_load_model(checkpointed_path=checkpointed_path)
#         torch._dynamo.config.verbose=True 
#         model = torch.compile(model)

    generated_text = inference(model, tokenizer, device)
    logging.info(f"Initial Text:\n{generated_text}")

    train(model, device, data_dict)