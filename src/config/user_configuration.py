class UserConfiguration:
    def __init__(self, **kwargs):
        self.net_id = kwargs.get("net_id")
        self.env = kwargs.get("env")

        self.root_path = kwargs.get("root_path", f"/scratch/{self.net_id}/fine_tuning/{self.env}")
        self.data_path = kwargs.get("data_path", f"/scratch/{self.net_id}/fine_tuning")
        self.cache_path = kwargs.get("tokenized_dataset_path", f"{self.data_path}/datasets")

        self.tokenized_dataset_path_generator = lambda dataset, tokenizer_name: f"{self.cache_path}/tokenized_{dataset}_{tokenizer_name}"
        self.train_dataset_path_generator = lambda dataset, tokenizer_name: self.tokenized_dataset_path_generator(dataset, tokenizer_name) + "_train"
        self.validation_dataset_path_generator = lambda dataset, tokenizer_name: self.tokenized_dataset_path_generator(dataset, tokenizer_name) + "_valid"

        self.model_path_generator = lambda model, dataset, tokenizer_name: f"{self.root_path}/models/fine_tuned_{model}_{dataset}_{tokenizer_name}"
        self.logs_path_generator = lambda model, dataset, tokenizer_name: f"{self.root_path}/logs/fine_tuned_{model}_{dataset}_{tokenizer_name}"
