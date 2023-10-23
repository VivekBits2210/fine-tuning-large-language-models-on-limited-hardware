import logging
import torch
from tqdm import tqdm
from typing import Optional
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoConfig, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
import bitsandbytes as bnb
from utilities.profiler_utils import measure_time_taken
from config.system_configuration import SystemConfiguration

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, system_config: SystemConfiguration) -> None:
        self.device = system_config.device
        self.model = None
        self.model_name: Optional[str] = None
        self.is_quantized = False

    @measure_time_taken
    def load_from_path(self, load_path: str) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(load_path)
        self.__augment_model()

    def load(self, model_name: str, quantization_configuration=None, style="causal", num_labels=None) -> None:
        self.model_name = model_name


        if not quantization_configuration:
            if style == "causal":
                configuration = AutoConfig.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, config=configuration
                )
            elif style == "classification":
                configuration = AutoConfig.from_pretrained(self.model_name, num_labels = num_labels)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, config=configuration
                )
            else:
                raise Exception("Model style not recognized!")
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=quantization_configuration.load_in_4bit,
                bnb_4bit_quant_type=quantization_configuration.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=quantization_configuration.bnb_4bit_compute_dtype,
                bnb_4bit_use_double_quant=quantization_configuration.bnb_4bit_use_double_quant,
            )

            logger.info(f"Quantizing the model with config as {quantization_config}")
            self.is_quantized = True
            if style == "causal":
                configuration = AutoConfig.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, config=configuration, device_map="auto", quantization_config=quantization_config
                )
            elif style == "classification":
                configuration = AutoConfig.from_pretrained(self.model_name, num_labels = num_labels)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, config=configuration, device_map="auto", quantization_config=quantization_config,
                )
            else:
                raise Exception("Model style not recognized!")
        self.__augment_model()

    def lorify(self, lora_configuration, module_style):
        lora_config = LoraConfig(
            r=lora_configuration.r,
            lora_alpha=lora_configuration.lora_alpha,
            lora_dropout=lora_configuration.lora_dropout,
            bias=lora_configuration.bias,
            task_type=lora_configuration.task_type,
            target_modules=self._find_lora_target_modules(module_style),
        )
        self.model = get_peft_model(self.model, lora_config)
        logger.info(
            f"Addling low-rank adapters to model with config as {self.model.peft_config}"
        )

    def __augment_model(self):
        #         self.model.gradient_checkpointing_enable()
        #         self.model.enable_input_require_grads()
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

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
            num_return_sequences=text_gen_config.num_return_sequences,
        )

    def infer(self, prompt, text_gen_config):
        prompt.to(self.device)

        self.model.config.use_cache = True
        self.model.eval()

        with torch.no_grad():
            output_sequence = self._generate_tokens(prompt, text_gen_config)
        self.model.train()
        self.model.config.use_cache = False

        return output_sequence

    @measure_time_taken
    def validate(self, validation_dataloader):
        self.model.eval()
        total_eval_loss = 0.0
        avg_eval_loss = 0.0
        for index, batch in tqdm(enumerate(validation_dataloader, 1)):
            batch = {
                k: v.pin_memory().to(self.device, non_blocking=True)
                for k, v in batch.items()
            }
            with torch.no_grad():
                outputs = self.model(**batch)
                loss = outputs.loss
            total_eval_loss += loss.item()
            avg_eval_loss = total_eval_loss / index

        perplexity = torch.exp(torch.as_tensor(avg_eval_loss)).item()
        self.model.train()

        return avg_eval_loss, perplexity

    def _find_lora_target_modules(self, module_style="qlora"):
        """Find all linear layer names in the model. reference from qlora paper."""
        cls = None
        if module_style == "qlora":
            cls = bnb.nn.Linear4bit

        lora_module_names = set()
        for name, module in self.model.named_modules():
            if isinstance(module, cls):
                # last layer is not add to lora_module_names
                if "lm_head" in name or "score" in name:
                    continue
                names = name.split(".")
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        return sorted(lora_module_names)
