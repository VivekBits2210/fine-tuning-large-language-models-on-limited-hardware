class LoraConfiguration:
    def __init__(self, **kwargs):
        self.r = kwargs.get("r", 64)
        self.lora_alpha = kwargs.get("lora_alpha", 16)
        self.lora_dropout = kwargs.get("lora_dropout", 0.1)
        self.bias = kwargs.get("bias", "none")
        self.task_type = kwargs.get("task_type", "CAUSAL_LM")
