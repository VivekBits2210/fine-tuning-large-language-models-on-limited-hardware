class TokenizerConfiguration:
    def __init__(self, **kwargs):
        self.max_tokens = kwargs.get("max_tokens")
        self.tokenizer_name = (
            f"{kwargs.get('tokenizer_name', 'default')}_{str(self.max_tokens)}"
        )
        self.padding_strategy = kwargs.get("padding_strategy", "max_length")
        self.truncation = kwargs.get("truncation", True)
        self.return_attention_mask = kwargs.get("return_attention_mask", True)
