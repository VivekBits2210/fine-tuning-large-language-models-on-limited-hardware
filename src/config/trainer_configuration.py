class TrainerConfiguration:
    def __init__(self, **kwargs):
        self.epochs = kwargs.get("epochs")
        self.sampling_interval = kwargs.get("sampling_interval")
        self.validation_interval = kwargs.get("validation_interval")
        self.checkpointing_interval = kwargs.get("checkpointing_interval")