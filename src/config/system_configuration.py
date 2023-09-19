class SystemConfiguration:
    def __init__(self, **kwargs):
        self.num_workers = kwargs.get("num_workers")
        self.device = kwargs.get("device", "cuda")
