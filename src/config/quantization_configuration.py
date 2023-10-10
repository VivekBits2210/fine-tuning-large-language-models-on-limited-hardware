class QuantizationConfiguration:
    def __init__(self, **kwargs):
        self.load_in_4bit = kwargs.get("load_in_4bit", True)
        self.bnb_4bit_quant_type = kwargs.get("bnb_4bit_quant_type", "nf4")
        self.bnb_4bit_compute_dtype = kwargs.get("bnb_4bit_compute_dtype", "float16")
        self.bnb_4bit_use_double_quant = kwargs.get("bnb_4bit_use_double_quant", False)