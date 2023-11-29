import time
import logging
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
import torch
import wandb
from tqdm import tqdm
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from utilities.db_utils import store_metric, store_checkpoint

from config import (
    SystemConfiguration,
    UserConfiguration,
    TokenizerConfiguration,
    TrainerConfiguration,
    TextGenConfiguration,
)
from managers import DataManager, ModelManager, TokenizationManager, SystemMonitor
from utilities.profiler_utils import measure_time_taken

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        user_config: UserConfiguration,
        system_config: SystemConfiguration,
        tokenizer_config: TokenizerConfiguration,
        text_gen_config: TextGenConfiguration,
        train_config: TrainerConfiguration,
        system_monitor: SystemMonitor,
        data_manager: DataManager,
        model_manager: ModelManager,
        tokenization_manager: TokenizationManager,
        training_dataloader,
        validation_dataloader,
        database_path,
        run_name,
        use_wandb=False,
        task="generation",
    ):
        self.task = task
        self.use_wandb = use_wandb
        self.model_name = model_manager.model_name
        self.user_config = user_config
        self.system_config = system_config
        self.tokenizer_config = tokenizer_config
        self.text_gen_config = text_gen_config
        self.train_config = train_config

        self.system_monitor = system_monitor
        self.data_manager = data_manager
        self.model_manager = model_manager
        self.tokenization_manager = tokenization_manager

        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.num_batches = len(self.training_dataloader)

        self.database_path = database_path
        self.run_name = run_name

        self.log_path = None
        self.model_path = None
        self._setup_logging_and_saving()

        self.optimizer = self._fetch_optimizer()

        logger.info(f"Using optimizer: {type(self.optimizer).__name__}")
        if self.train_config.scheduler_type == "linear":
            self.lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.train_config.num_warmup_steps,
                num_training_steps=(
                    len(self.training_dataloader) * self.train_config.epochs
                ),
            )
        elif self.train_config.scheduler_type == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.train_config.num_warmup_steps,
                num_training_steps=(
                    len(self.training_dataloader) * self.train_config.epochs
                ),
            )
        lr_scheduler_details = {
            "num_warmup_steps": self.train_config.num_warmup_steps,
            "num_training_steps": len(self.training_dataloader)
            * self.train_config.epochs,
        }

        self.running_loss = 0.0
        store_metric(
            self.database_path,
            "lr_scheduler_details",
            self.run_name,
            lr_scheduler_details,
        )
        if self.use_wandb:
            wandb.log(lr_scheduler_details)
            wandb.log(
                {"log_message": f"Using optimizer: {type(self.optimizer).__name__}"}
            )

    def _fetch_optimizer(self):
        if self.model_manager.is_quantized:
            from bitsandbytes.optim import AdamW

            optimizer = AdamW(
                params=self.model_manager.model.parameters(),
                lr=self.train_config.lr,
                is_paged=self.train_config.is_optimizer_paged,
                optim_bits=self.train_config.optim_bits,
            )
        else:
            from transformers import AdamW

            optimizer = AdamW(
                params=self.model_manager.model.parameters(), lr=self.train_config.lr
            )
        optimizer_detail = {
            "optimizer_type": type(optimizer).__name__,
        }
        store_metric(
            self.database_path, "optimizer_details", self.run_name, optimizer_detail
        )
        if self.use_wandb:
            wandb.log(optimizer_detail)
        return optimizer

    def _setup_logging_and_saving(self):
        model_name = self.model_name
        dataset_name = self.data_manager.dataset_name
        tokenizer_name = self.tokenizer_config.tokenizer_name

        self.log_path = self.user_config.logs_path_generator(
            model_name, dataset_name, tokenizer_name
        )

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        with open(f"{self.log_path}/training.log", "w+") as f:
            f.write("epoch\tbatch\ttrain\tloss\tgenerated_text\n")
        with open(f"{self.log_path}/validation.log", "w+") as f:
            f.write("epoch\tbatch\tvalidation_loss\tperplexity\n")

        self.model_path = self.user_config.model_path_generator(
            model_name, dataset_name, tokenizer_name
        )

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def handle_batch(self, epoch, index, batch):
        self.model_manager.model.train()

        current_lr = self.optimizer.param_groups[0]["lr"]

        if index % 10 == 0:
            training_loss_details = {
                "epoch": epoch + (index / len(self.training_dataloader)),
                "running_loss": self.running_loss / index,
            }
            learning_rate_details = {
                "epoch": epoch + (index / len(self.training_dataloader)),
                "learning_rate": current_lr,
            }
            gpu_details = {
                "epoch": epoch + (index / len(self.training_dataloader)),
                "gpu_util": self.system_monitor.get_gpu_utilization(),
                "ram_usage": self.system_monitor.get_ram_usage(),
            }
            logger.info(
                f"Running Loss at epoch {training_loss_details['epoch']}, {training_loss_details['running_loss']}"
            )
            logger.info(
                f"GPU usage at epoch {gpu_details['epoch']}, {gpu_details['gpu_util']} GB"
            )
            if self.use_wandb:
                wandb.log(
                    {
                        "log_message": f"Running Loss at epoch {training_loss_details['epoch']}, "
                        f"{training_loss_details['running_loss']}"
                    }
                )
                wandb.log(
                    {
                        "log_message": f"GPU usage at epoch {gpu_details['epoch']}, {gpu_details['gpu_util']} GB"
                    }
                )
            store_metric(
                self.database_path,
                "training_loss_details",
                self.run_name,
                training_loss_details,
            )
            store_metric(
                self.database_path,
                "learning_rate",
                self.run_name,
                learning_rate_details,
            )
            store_metric(
                self.database_path, "gpu_utilization", self.run_name, gpu_details
            )

            if self.use_wandb:
                wandb.log(training_loss_details)
                wandb.log(learning_rate_details)
                wandb.log(gpu_details)

        # Sample an output from the model, at each sampling interval
        if (
            index % self.train_config.sampling_interval == 0
            and self.task == "generation"
        ):
            prompt = self.tokenization_manager.encode("This")
            sequence = self.model_manager.infer(prompt, self.text_gen_config)
            text = self.tokenization_manager.decode(sequence, self.text_gen_config)
            logger.info(
                f"Training: Epoch-{epoch} Index-{index} Loss-{self.running_loss / index}"
            )
            logger.info(f"Text:\n{text}")
            if self.use_wandb:
                wandb.log(
                    {
                        "log_message": f"Training: Epoch-{epoch} Index-{index} Loss-{self.running_loss / index}"
                    }
                )
                wandb.log({"log_message": f"Text:\n{text}"})

            text_gen_details = {
                "epoch": epoch + (index / len(self.training_dataloader)),
                "text": text,
            }
            store_metric(
                self.database_path, "generated_text", self.run_name, text_gen_details
            )
            if self.use_wandb:
                wandb.log(text_gen_details)

        # Save the model at each checkpointing interval
        if index % self.train_config.checkpointing_interval == 0:
            self.save_checkpoint(epoch, index)

        # Validate the model at each validation interval
        if index % self.train_config.validation_interval == 0:
            if self.task == "generation":
                self.validate_model(epoch, index)
            elif self.task == "classification":
                self.validate_model_for_classification(epoch, index)

        self.forward_backward_pass(batch)

    @measure_time_taken
    def save_checkpoint(self, epoch, index):
        logger.info(f"Checkpointing model at epoch={epoch} and batch={index}\n")
        if self.use_wandb:
            wandb.log(
                {
                    "log_message": f"Checkpointing model at epoch={epoch} and batch={index}\n"
                }
            )
        checkpointing_path = f"{self.model_path}_{epoch}_{index}"
        store_checkpoint(
            self.database_path,
            epoch + (index / len(self.training_dataloader)),
            self.run_name,
            checkpointing_path,
        )
        self.model_manager.model.save_pretrained(checkpointing_path)
        self.tokenization_manager.tokenizer.save_pretrained(checkpointing_path)

    @measure_time_taken
    def validate_model(self, epoch, index):
        logger.info("Running Validation...")
        avg_eval_loss, perplexity = self.model_manager.validate(
            self.validation_dataloader
        )
        logger.info(
            f"Batch {index}/{len(self.training_dataloader)}, Validation Loss: {avg_eval_loss:.4f}, "
            f"Perplexity: {perplexity:.2f}"
        )
        if self.use_wandb:
            wandb.log({"log_message": "Running Validation..."})
            wandb.log(
                {
                    "log_message": f"Epoch: {epoch + (index / len(self.training_dataloader))}, Validation Loss: {avg_eval_loss:.4f}, Perplexity: {perplexity:.2f}"
                }
            )

        metric_details = {
            "epoch": epoch + (index / len(self.training_dataloader)),
            "eval_loss": avg_eval_loss,
            "perplexity": perplexity,
        }
        store_metric(
            self.database_path, "validation_metrics", self.run_name, metric_details
        )
        if self.use_wandb:
            wandb.log(metric_details)

    @measure_time_taken
    def validate_model_for_classification(self, epoch, index):
        logger.info("Running Validation...")
        if self.use_wandb:
            wandb.log({"log_message": "Running Validation..."})

        loss_fn = torch.nn.BCEWithLogitsLoss()
        total_loss = 0
        self.model_manager.model.eval()  # Ensure model is in evaluation mode

        all_preds = []
        all_labels = []

        for batch in tqdm(self.validation_dataloader):
            with torch.no_grad():
                batch = {k: v.to(self.model_manager.device) for k, v in batch.items()}
                outputs = self.model_manager.model(
                    batch["input_ids"], attention_mask=batch["attention_mask"]
                )
                logits = outputs.logits
                loss = loss_fn(logits, batch["labels"])
                preds = torch.sigmoid(logits) > 0.5
                total_loss += loss.item()

                for p, l in zip(
                    list(1 * preds.detach().cpu()), list(batch["labels"].detach().cpu())
                ):
                    pred_indices = [i for i, val in enumerate(p) if val > 0]
                    label_indices = [i for i, val in enumerate(l) if val > 0]
                    logger.info(f"Prediction: {pred_indices}, Actual: {label_indices}")
                    if self.use_wandb:
                        wandb.log(
                            {
                                "log_message": f"Prediction: {pred_indices}, Actual: {label_indices}"
                            }
                        )

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())

        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(
            all_labels, all_preds, average="micro"
        )  # using micro average for multi-label
        hamming = hamming_loss(all_labels, all_preds)

        avg_eval_loss = total_loss / len(self.validation_dataloader)
        logger.info(
            f"Batch {index}/{len(self.training_dataloader)}\n "
            f"Validation Loss: {avg_eval_loss:.4f}\n "
            f"Accuracy: {100 * accuracy:.2f}, F1: {f1:.2f}\n "
            f"Hamming Loss: {hamming:.4f}\n"
        )
        if self.use_wandb:
            wandb.log(
                {
                    "log_message": f"Batch {index}/{len(self.training_dataloader)}\n "
                    f"Validation Loss: {avg_eval_loss:.4f}\n "
                    f"Accuracy: {100 * accuracy:.2f}, F1: {f1:.2f}\n "
                    f"Hamming Loss: {hamming:.4f}\n"
                }
            )

        metric_details = {
            "epoch": epoch + (index / len(self.training_dataloader)),
            "eval_loss": avg_eval_loss,
            "accuracy": accuracy,
            "hamming loss": hamming,
            "f1": f1,
        }
        store_metric(
            self.database_path, "validation_metrics", self.run_name, metric_details
        )
        if self.use_wandb:
            wandb.log(metric_details)

        LABEL_NAMES = [
            "Computer Science",
            "Physics",
            "Mathematics",
            "Statistics",
            "Quantitative Biology",
            "Quantitative Finance",
        ]
        for i in range(len(all_preds)):
            pred_i = [
                LABEL_NAMES[j] for j in range(len(LABEL_NAMES)) if all_preds[i][j]
            ]
            label_i = [
                LABEL_NAMES[j] for j in range(len(LABEL_NAMES)) if all_labels[i][j]
            ]
            logger.info(f"{i}: Predicted labels: {pred_i}")
            logger.info(f"{i}: Actual labels: {label_i}")
            if self.use_wandb:
                wandb.log({"log_message": f"{i}: Predicted labels: {pred_i}"})
                wandb.log({"log_message": f"{i}: Actual labels: {label_i}"})
            if (i + 1) % 7 == 0:
                break

    def forward_backward_pass(self, batch):
        batch = {
            k: v.pin_memory().to(self.model_manager.device, non_blocking=True)
            for k, v in batch.items()
        }
        outputs = self.model_manager.model(
            batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        if self.task == "classification":
            logits = outputs.logits
            loss = torch.nn.BCEWithLogitsLoss()(logits, batch["labels"])
            for p, l in zip(
                list(1 * (torch.sigmoid(logits) > 0.5).detach().cpu()),
                list(batch["labels"].detach().cpu()),
            ):
                pred_indices = [i for i, val in enumerate(p) if val > 0]
                label_indices = [i for i, val in enumerate(l) if val > 0]
                #                 logger.info(f"Prediction: {pred_indices}, Actual: {label_indices}")
                if self.use_wandb:
                    wandb.log(
                        {
                            "log_message": f"Prediction: {pred_indices}, Actual: {label_indices}"
                        }
                    )
        else:
            loss = outputs.loss

        #         logger.info(f"Batch Loss: {loss.item()}")
        if self.use_wandb:
            wandb.log({"log_message": f"Batch Loss: {loss.item()}"})

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logger.error("Nan or inf found in loss!")
            if self.use_wandb:
                wandb.log({"log_message": "Nan or inf found in loss!"})
            for name, param in self.model_manager.model.named_parameters():
                if torch.isnan(param.data).any():
                    logger.error(f"Nan found in {name} data!")
                    if self.use_wandb:
                        wandb.log({"log_message": f"Nan found in {name} data!"})
                if torch.isnan(param.grad).any():
                    logger.error(f"Nan found in {name} gradient!")
                    if self.use_wandb:
                        wandb.log({"log_message": f"Nan found in {name} gradient!"})
                if torch.isinf(param.data).any():
                    logger.error(f"Inf found in {name} data!")
                    if self.use_wandb:
                        wandb.log({"log_message": f"Inf found in {name} data!"})
                if torch.isinf(param.grad).any():
                    logger.error(f"Inf found in {name} gradient!")
                    if self.use_wandb:
                        wandb.log({"log_message": f"Inf found in {name} gradient!"})
        else:
            self.running_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

    def train(self):
        start_time = time.time()

        for epoch in tqdm(range(self.train_config.epochs)):
            self.running_loss = 0.0
            logger.info(f"Starting Epoch: {epoch}/{self.train_config.epochs}")
            if self.use_wandb:
                wandb.log(
                    {
                        "log_message": f"Starting Epoch: {epoch}/{self.train_config.epochs}"
                    }
                )

            epoch_start_time = time.time()
            for index, batch in tqdm(
                enumerate(self.training_dataloader, 1),
                total=len(self.training_dataloader),
            ):
                if index < 10:
                    logger.info(
                        f"Epoch: {epoch}, Index: {index}, RAM Usage: {self.system_monitor.get_ram_usage()} MB, "
                        f"GPU Utilization: {self.system_monitor.get_gpu_utilization()} GB"
                    )
                self.handle_batch(epoch, index, batch)

            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            epoch_dict = {"epoch": epoch, "time": epoch_time}
            store_metric(self.database_path, "epoch_time", self.run_name, epoch_dict)
            if self.use_wandb:
                wandb.log(epoch_dict)

            logger.info(
                f"Training Loss after Epoch {epoch}: {self.running_loss / self.num_batches}"
            )
            if self.use_wandb:
                wandb.log(
                    {
                        "log_message": f"Training Loss after Epoch {epoch}: {self.running_loss / self.num_batches}"
                    }
                )

        end_time = time.time()
        total_time = end_time - start_time
        total_time_dict = {"total_time": total_time}
        store_metric(self.database_path, "total_time", self.run_name, total_time_dict)
        if self.use_wandb:
            wandb.log(total_time_dict)

        logger.info(
            f"Final Training Loss after {self.train_config.epochs} epochs: {self.running_loss / self.num_batches}"
        )
        if self.use_wandb:
            wandb.log(
                {
                    "log_message": f"Final Training Loss after {self.train_config.epochs} epochs: {self.running_loss / self.num_batches}"
                }
            )

        store_checkpoint(
            self.database_path,
            self.train_config.epochs,
            self.run_name,
            self.model_path,
        )
        self.model_manager.model.save_pretrained(self.model_path)
        self.tokenization_manager.tokenizer.save_pretrained(self.model_path)
