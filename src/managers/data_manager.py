import os
import random
import logging
import torch
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding

from utilities.profiler_utils import measure_time_taken
from config import UserConfiguration, TokenizerConfiguration, SystemConfiguration

logger = logging.getLogger(__name__)


class DataManager:
    def __init__(
        self,
        user_config: UserConfiguration,
        system_config: SystemConfiguration,
        tokenizer_config: TokenizerConfiguration,
        dataset_name=None,
    ) -> None:
        self.user_config = user_config
        self.system_config = system_config
        self.tokenizer_config = tokenizer_config
        self.dataset_name = dataset_name

        self.data_collator = None
        self.dataset = None
        self.tokenized_dataset = None

    @measure_time_taken
    def create_dataset_from_csv_for_text_classification(self, name: str, csv_file_path: str, save_to_disk=True, topics=[]) -> None:
        self.dataset_name = name
        self.dataset = load_dataset(
            "csv",
            data_files=csv_file_path,
            num_proc=self.system_config.num_workers,
            cache_dir=self.user_config.cache_path,
        )
        
        # Handling multi-label
        def format_multilabel(example):
            example['labels'] = torch.tensor([example[topic] for topic in topics])
            return example

        self.dataset = self.dataset.map(format_multilabel)

        if save_to_disk:
            self.dataset.save_to_disk(self.user_config.data_path)

    @measure_time_taken
    def create_dataset_from_jsonl_zst_file(
        self, name: str, jsonl_zst_file_path: str, save_to_disk=True
    ) -> None:
        self.dataset_name = name
        self.dataset = load_dataset(
            "json",
            data_files=jsonl_zst_file_path,
            num_proc=self.system_config.num_workers,
            split="train",
            cache_dir=self.user_config.cache_path,
        )
        if save_to_disk:
            self.dataset.save_to_disk(self.user_config.data_path)

    @measure_time_taken
    def create_tokenized_dataset(self, tokenizer, save_to_disk: bool = True, is_classification: bool = False) -> None:
        print(self.dataset.column_names)
        raise Exception
        self.tokenized_dataset = self.dataset.map(
            tokenizer,
            batched=True,
            num_proc=self.system_config.num_workers,
            remove_columns=["text", "meta"] if not is_classification else ["ID"], 
        )
        if save_to_disk:
            if not self.dataset_name:
                raise Exception(
                    "You need to set dataset_name in order to save to disk!"
                )
            self.tokenized_dataset.save_to_disk(
                self.user_config.tokenized_dataset_path_generator(
                    self.dataset_name, self.tokenizer_config.tokenizer_name
                )
            )

    @measure_time_taken
    def fetch_train_validation_split(
        self, split_ratio: float = 0.95, keep_fraction: float = 0.1, save_to_disk=True
    ):
        if not self.tokenized_dataset:
            raise ValueError("You need to tokenize the dataset first!")

        random_seed = random.randint(0, 2**32 - 1)
        keep_size = int(keep_fraction * len(self.tokenized_dataset))
        sampled_dataset = self.tokenized_dataset.shuffle(seed=random_seed).select(
            range(keep_size)
        )

        train_size = int(split_ratio * len(sampled_dataset))
        datasets = DatasetDict(
            {
                "train": Dataset.from_dict(sampled_dataset[:train_size]),
                "valid": Dataset.from_dict(sampled_dataset[train_size:]),
            }
        )

        if save_to_disk:
            if not self.dataset_name:
                raise Exception(
                    "You need to set dataset_name in order to save to disk!"
                )
            datasets["train"].save_to_disk(
                self.user_config.train_dataset_path_generator(
                    self.dataset_name, self.tokenizer_config.tokenizer_name
                )
            )
            datasets["valid"].save_to_disk(
                self.user_config.validation_dataset_path_generator(
                    self.dataset_name, self.tokenizer_config.tokenizer_name
                )
            )

        return datasets["train"], datasets["valid"]

    def set_data_collator(self, tokenizer) -> None:
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        )

    def set_data_collator_for_text_classification(self, tokenizer) -> None:
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def fetch_train_validation_split_from_disk(self):
        if not self.dataset_name:
            raise Exception("You need to set dataset_name in order to fetch from disk!")

        train_path = self.user_config.train_dataset_path_generator(
            self.dataset_name, self.tokenizer_config.tokenizer_name
        )
        validation_path = self.user_config.validation_dataset_path_generator(
            self.dataset_name, self.tokenizer_config.tokenizer_name
        )

        if not os.path.exists(train_path):
            raise FileNotFoundError("The training dataset path does not exist!")

        training_dataset = load_from_disk(train_path)

        validation_dataset = None
        if os.path.exists(validation_path):
            validation_dataset = load_from_disk(validation_path)
        else:
            logger.warning(f"The validation dataset path does not exist!")
        return training_dataset, validation_dataset

    @measure_time_taken
    def fetch_dataloaders(self, training_dataset, batch_size, validation_dataset=None):
        if not self.data_collator:
            raise Exception(
                "The data collator needs to be set before data loaders can be created!"
            )
        logger.info(f"Batch size is set to {batch_size}.")

        training_dataloader = DataLoader(
            training_dataset,
            sampler=RandomSampler(training_dataset),
            batch_size=batch_size,
            num_workers=self.system_config.num_workers,
            collate_fn=self.data_collator,
            pin_memory=True,
        )

        validation_dataloader = None
        if validation_dataset:
            validation_dataloader = DataLoader(
                validation_dataset,
                sampler=SequentialSampler(validation_dataset),
                batch_size=batch_size,
                num_workers=self.system_config.num_workers,
                collate_fn=self.data_collator,
                pin_memory=True,
            )

        return training_dataloader, validation_dataloader
