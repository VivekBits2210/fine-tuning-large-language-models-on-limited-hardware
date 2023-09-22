class TrainerConfiguration:
    def __init__(self, **kwargs):
        self.epochs = kwargs.get("epochs", 2)
        self.sampling_interval = kwargs.get("sampling_interval", 50)
        self.validation_interval = kwargs.get("validation_interval", 5)
        self.checkpointing_interval = kwargs.get("checkpointing_interval", 1000)