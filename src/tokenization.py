from transformers import AutoTokenizer


class TokenizerFactory:
    def __init__(self, save_path: str) -> None:
        self.save_path = save_path

    def load_tokenizer(self, load_path: str):
        return AutoTokenizer.from_pretrained(load_path)

    def create_tokenizer(self, model_name: str):
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.save_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        return tokenizer
