class QuantizationConfiguration:
    def __init__(self, **kwargs):
        self.load_in_4bit = kwargs.get("load_in_4bit", True)
        self.bnb_4bit_quant_type = kwargs.get("bnb_4bit_quant_type", "nf4")
        self.bnb_4bit_compute_dtype = kwargs.get("bnb_4bit_compute_dtype", "bfloat16")
        self.bnb_4bit_use_double_quant = kwargs.get("bnb_4bit_use_double_quant", False)
        self.use_flash_attention_2 = kwargs.get("use_flash_attention_2", False)
