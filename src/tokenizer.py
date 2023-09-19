from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling


class Tokenizer:
    def __init__(self, user_config) -> None:
        self.user_config = user_config
        self.tokenizer = None
        self.data_collator = None

    def load_from_path(self, load_path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.__set_data_collator()

    def load_for_model(self, model_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.user_config.cache_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.__set_data_collator()

    def __set_data_collator(self) -> None:
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

    def run(self, data, data_prep_config):
        return self.tokenizer(
            data["text"],
            padding=data_prep_config.padding_strategy,
            truncation=True,
            max_length=data_prep_config.max_tokens,
            return_attention_mask=True
        )
