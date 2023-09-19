from typing import Optional
from transformers import AutoModelForCausalLM, AutoConfig


class ModelManager:
    def __init__(self, device: str) -> None:
        self.device = device
        self.model = None
        self.model_name: Optional[str] = None

    def load_from_path(self, load_path: str) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(load_path)
        self.model.to(self.device)
        self.__augment_model()

    def load(self, model_name: str) -> None:
        self.model_name = model_name
        configuration = AutoConfig.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, config=configuration)
        self.model.to(self.device)
        self.__augment_model()

    def __augment_model(self):
        self.model.config.use_cache = False




