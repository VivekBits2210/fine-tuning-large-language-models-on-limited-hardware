class TextGenConfiguration:
    def __init__(self, tokenizer, **kwargs):
        self.eos_token = tokenizer.eos_token
        self.min_tokens_to_generate = kwargs.get("min_tokens_to_generate")

        self.beginning_of_sentence_token_id = kwargs.get("beginning_of_sentence_token_id", tokenizer.bos_token_id)
        self.pad_token_id = kwargs.get("pad_token_id", tokenizer.eos_token_id)
        self.max_tokens_to_generate = kwargs.get("max_tokens_to_generate", 2*self.min_tokens_to_generate)
        self.top_p = kwargs.get("top_p", 0.95)
        self.top_k = kwargs.get("top_k", 50)
        self.num_return_sequences = kwargs.get("num_return_sequences", 1)
        self.do_sample = kwargs.get("do_sample", True)
        self.should_remove_new_lines = kwargs.get("should_remove_new_lines", True)
        self.should_remove_tabs = kwargs.get("should_remove_tabs", True)
        self.skip_special_tokens = kwargs.get("skip_special_tokens", True)