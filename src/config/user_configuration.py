import os
import logging

logger = logging.getLogger(__name__)


class UserConfiguration:
    def __init__(self, **kwargs):
        self.net_id = kwargs.get("net_id")
        self.env = kwargs.get("env")

        # Set the base directory depending on the OS
        if os.name == "posix":  # UNIX-like systems
            base_dir = "/scratch"
        else:  # Windows (and potentially other OSs)
            base_dir = "E:\\"
            # base_dir = os.environ.get('USERPROFILE', 'D:\\')  # Ideally, default to D:\\ if USERPROFILE is not set
        logger.info(f"The base directory is set to {base_dir}.")

        default_root_path = os.path.join(base_dir, self.net_id, "fine_tuning", self.env)
        default_data_path = os.path.join(base_dir, self.net_id, "fine_tuning")

        self.root_path = kwargs.get("root_path", default_root_path)
        self.data_path = kwargs.get("data_path", default_data_path)
        self.cache_path = kwargs.get(
            "tokenized_dataset_path", os.path.join(self.data_path, "datasets")
        )

        self.tokenized_dataset_path_generator = (
            lambda dataset, tokenizer_name: os.path.join(
                self.cache_path, f"tokenized_{dataset}_{tokenizer_name}"
            )
        )
        self.train_dataset_path_generator = (
            lambda dataset, tokenizer_name: f"{self.tokenized_dataset_path_generator(dataset, tokenizer_name)}_train"
        )
        self.validation_dataset_path_generator = (
            lambda dataset, tokenizer_name: f"{self.tokenized_dataset_path_generator(dataset, tokenizer_name)}_valid"
        )

        self.model_path_generator = lambda model, dataset, tokenizer_name: os.path.join(
            self.root_path, "models", f"fine_tuned_{model}_{dataset}_{tokenizer_name}"
        )
        self.logs_path_generator = lambda model, dataset, tokenizer_name: os.path.join(
            self.root_path, "logs", f"fine_tuned_{model}_{dataset}_{tokenizer_name}"
        )

    def commit(self):
        # TODO: Create root, data and cache folders if they are missing
        pass
