from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling


class TokenizerFactory:
    def __init__(self, user_config) -> None:
        self.user_config = user_config
        self.tokenizer = None
        self.data_collator = None

    def load_tokenizer_from_path(self, load_path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        return self.tokenizer

    def create_tokenizer(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.user_config.cache_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        return self.tokenizer

    def get_data_collator(self):
        if self.data_collator:
            return self.data_collator

        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        return self.data_collator