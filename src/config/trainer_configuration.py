class TrainerConfiguration:
    def __init__(self, **kwargs):
        self.epochs = kwargs.get("epochs", 50)
        self.lr = kwargs.get("lr", 2e-4)
        self.num_warmup_steps = kwargs.get("num_warmup_steps", 5)
        self.sampling_interval = kwargs.get("sampling_interval", 50)
        self.validation_interval = kwargs.get("validation_interval", 500)
        self.checkpointing_interval = kwargs.get("checkpointing_interval", 1000)
