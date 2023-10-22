from transformers import AutoTokenizer

from utilities.profiler_utils import measure_time_taken
from config.user_configuration import UserConfiguration
from config.tokenizer_configuration import TokenizerConfiguration


class TokenizationManager:
    def __init__(
        self,
        user_config: UserConfiguration,
        tokenization_config: TokenizerConfiguration,
    ) -> None:
        self.user_config = user_config
        self.tokenization_config = tokenization_config
        self.tokenizer = None
        self.data_collator = None

    @measure_time_taken
    def load_from_path(self, load_path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)

    def load_for_model(self, model_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=self.user_config.cache_path
        )

        # TODO: Set these using configurations
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def tokenize(self, data):
        return self.tokenizer(
            data["text"],
            padding=self.tokenization_config.padding_strategy,
            truncation=self.tokenization_config.truncation,
            max_length=self.tokenization_config.max_tokens,
            return_attention_mask=self.tokenization_config.return_attention_mask,
        )

    def tokenize_for_text_classification(self, data):
        return {
            "input_ids": self.tokenizer(
                data['TITLE'] + "--" + data['ABSTRACT'],
                padding=self.tokenization_config.padding_strategy,
                truncation=self.tokenization_config.truncation,
                max_length=self.tokenization_config.max_tokens,
                return_attention_mask=self.tokenization_config.return_attention_mask,
                return_tensors="pt",
            )["input_ids"],
            "labels": data["labels"]
        }

    def encode(self, prompt: str = "This"):
        sequence = self.tokenizer(
            self.tokenizer.eos_token + prompt, return_tensors="pt"
        )
        return sequence

    def decode(self, sequence, text_gen_config):
        generated_text = self.tokenizer.decode(
            sequence[0], skip_special_tokens=text_gen_config.skip_special_tokens
        )

        if text_gen_config.should_remove_new_lines:
            generated_text = generated_text.replace("\n", "")
        if text_gen_config.should_remove_tabs:
            generated_text = generated_text.replace("\t", " ")

        return generated_text
