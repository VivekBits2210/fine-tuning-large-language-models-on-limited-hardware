import os
from typing import Optional
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from tokenizer import Tokenizer


class DataManager:
    def __init__(self,
                 tokenizer: Tokenizer,
                 system_config: "SystemConfig",
                 user_config: "UserConfig",
                 data_prep_config: "DataPrepConfig"
                 ) -> None:
        self.tokenizer = tokenizer
        self.system_config = system_config
        self.user_config = user_config
        self.data_prep_config = data_prep_config
        self.dataset = None
        self.tokenized_dataset = None

    def create_dataset_from_jsonl_zst_file(self, jsonl_zst_file_path: Optional[str] = None) -> Dataset:
        self.dataset = load_dataset("json",
                                    data_files=jsonl_zst_file_path,
                                    num_proc=self.system_config.num_workers,
                                    split="train",
                                    cache_dir=self.user_config.cache_path)
        return self.dataset

    def create_tokenized_dataset(self, save_to_disk: bool = True) -> None:
        self.tokenized_dataset = self.dataset.map(
            self.tokenizer.run,
            fn_kwargs={'data_prep_config': self.data_prep_config},
            batched=True,
            num_proc=self.system_config.num_workers,
            remove_columns=["text", "meta"],
        )
        if save_to_disk:
            self.tokenized_dataset.save_to_disk(self.user_config.tokenized_dataset_path)

    def train_test_split(self, tokenized_dataset: Dataset = None, split_ratio: float = 0.9):
        if not tokenized_dataset:
            if not self.tokenized_dataset:
                raise ValueError("You need to tokenize the dataset first!")

            tokenized_dataset = self.tokenized_dataset

        train_size = int(split_ratio * len(tokenized_dataset))
        datasets = DatasetDict({
            'train': Dataset.from_dict(tokenized_dataset[:train_size]),
            'valid': Dataset.from_dict(tokenized_dataset[train_size:])
        })
        return datasets['train'], datasets['valid']

    def train_test_split_from_disk(self):
        if not os.path.exists(self.user_config.train_dataset_path):
            raise FileNotFoundError("The train_dataset_path does not exist!")

        train_dataset = load_from_disk(self.user_config.train_dataset_path)

        valid_dataset = None
        if os.path.exists(self.user_config.valid_dataset_path):
            valid_dataset = load_from_disk(self.user_config.valid_dataset_path)
        return train_dataset, valid_dataset

    def create_dataloader(self, train_dataset, batch_size, valid_dataset=None):
        train_dataloader = DataLoader(train_dataset,
                                      sampler=RandomSampler(train_dataset),
                                      batch_size=batch_size,
                                      num_workers=self.system_config.num_workers,
                                      collate_fn=self.tokenizer.data_collator,
                                      pin_memory=True)

        valid_dataloader = None
        if valid_dataset:
            valid_dataloader = DataLoader(valid_dataset,
                                          sampler=SequentialSampler(valid_dataset),
                                          batch_size=batch_size,
                                          num_workers=self.system_config.num_workers,
                                          collate_fn=self.tokenizer.data_collator,
                                          pin_memory=True)

        return train_dataloader, valid_dataloader