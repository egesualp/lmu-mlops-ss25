import torch
import random
import numpy as np
import sys
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.data.data_collator import DataCollatorWithPadding
import evaluate
import hydra
from omegaconf import DictConfig
from pathlib import Path
from loguru import logger as log
import os
import wandb

from data import create_hf_datasets
from model import create_hf_model

# Load accuracy metric once
accuracy_metric = evaluate.load("accuracy")

def setup_logging(cfg: DictConfig, logging_choice: str):
    """
    Setup logging based on configuration.
    """
    if logging_choice in ["loguru", "both"]:
        # Get Hydra output directory
        hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

        # Use sensible defaults for loguru
        log_file = os.path.join(hydra_path, "train_hf.log")

        # Remove existing handlers to avoid duplicates
        log.remove()

        # Add console handler
        log.add(
            sys.stdout,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level="INFO",
            colorize=True
        )

        # Add file handler with rotation
        log.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
            level="INFO",
            rotation="10 MB",
            retention="7 days",
            compression="gz"
        )

        log.info("Loguru logging configured")

    if logging_choice in ["wandb", "both"]:
        # Use sensible defaults for wandb
        wandb.init(
            project="financial-sentiment-bert",
            entity=None,  # Use default entity
            tags=["bert", "sentiment", "financial"],
            notes="BERT fine-tuning for financial sentiment analysis",
            resume="allow",
            config=dict(cfg)  # Log the entire config
        )

        log.info("Wandb logging configured")

def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return accuracy_metric.compute(predictions=predictions, references=labels)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg: DictConfig):
    # Config values
    experiment_name = cfg.experiment_name
    data_dir = Path(cfg.data.data_dir)
    max_rows = cfg.data.get("max_rows", None)
    batch_size = cfg.batch_size
    epochs = cfg.epochs
    lr = cfg.model.lr
    seed = cfg.seed
    pretrained_model = cfg.model.pretrained_model
    eval_strategy = cfg.get("eval_strategy", "none")  # epoch/steps/none
    eval_steps = int(cfg.get("eval_steps", 100))
    logging_choice = cfg.get("logging", None)  # loguru or wandb
    save_strategy = cfg.get("save_strategy", "end")  # end/checkpoint/none

    # Model training parameters
    weight_decay = cfg.model.get("weight_decay", 0.01)
    warmup_steps = cfg.model.get("warmup_steps", 100)
    scheduler = cfg.model.get("scheduler", "linear")
    max_grad_norm = cfg.model.get("max_grad_norm", 1.0)
    label_smoothing = cfg.model.get("label_smoothing", 0.0)
    optim = cfg.model.get("optim", "adamw_torch")
    adam_beta1 = cfg.model.get("adam_beta1", 0.9)
    adam_beta2 = cfg.model.get("adam_beta2", 0.999)
    adam_epsilon = cfg.model.get("adam_epsilon", 1e-8)

    # Additional training parameters
    dataloader_num_workers = cfg.model.get("dataloader_num_workers", 0)
    dataloader_pin_memory = cfg.model.get("dataloader_pin_memory", True)
    remove_unused_columns = cfg.model.get("remove_unused_columns", True)
    group_by_length = cfg.model.get("group_by_length", False)
    fp16 = cfg.model.get("fp16", False)
    bf16 = cfg.model.get("bf16", False)
    dataloader_drop_last = cfg.model.get("dataloader_drop_last", False)
    logging_steps = cfg.model.get("logging_steps", 500)
    save_total_limit = cfg.model.get("save_total_limit", 3)
    full_determinism = cfg.model.get("full_determinism", False)

    # Seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Setup logging
    setup_logging(cfg, logging_choice)

    # Log initial information
    if logging_choice in ["loguru", "both"]:
        log.info("=" * 60)
        log.info("Starting BERT Fine-tuning Training")
        log.info("=" * 60)
        log.info("Experiment: {}", experiment_name)
        log.info("Model: {}", pretrained_model)
        log.info("Learning Rate: {}", lr)
        log.info("Batch Size: {}", batch_size)
        log.info("Epochs: {}", epochs)
        log.info("Seed: {}", seed)
        log.info("Device: {}", "CUDA" if torch.cuda.is_available() else "CPU")
        if torch.cuda.is_available():
            log.info("GPU: {}", torch.cuda.get_device_name(0))
        log.info("=" * 60)

    # Create datasets and model using imported functions
    if logging_choice in ["loguru", "both"]:
        log.info("Loading datasets...")

    train_ds, eval_ds, tokenizer, num_labels = create_hf_datasets(
        data_dir, pretrained_model, max_rows
    )

    if logging_choice in ["loguru", "both"]:
        log.info("Dataset loaded successfully")
        log.info("Train dataset size: {}", len(train_ds))
        log.info("Eval dataset size: {}", len(eval_ds))
        log.info("Number of labels: {}", num_labels)

    if logging_choice in ["loguru", "both"]:
        log.info("Creating model...")

    model = create_hf_model(pretrained_model, num_labels)

    if logging_choice in ["loguru", "both"]:
        log.info("Model created successfully")
        log.info("Model parameters: {:,}", sum(p.numel() for p in model.parameters()))
        log.info("Trainable parameters: {:,}", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Convert strategies
    hf_eval_strategy = {
        "none": "no",
        "epoch": "epoch",
        "steps": "steps",
    }[eval_strategy]
    hf_save_strategy = {
        "none": "no",
        "end": "epoch",  # save at each epoch end but we'll copy best later
        "checkpoint": "steps",
    }[save_strategy]

    # Wandb toggle
    if logging_choice not in ["wandb", "both"]:
        os.environ["WANDB_MODE"] = "disabled"

    if logging_choice in ["loguru", "both"]:
        log.info("Configuring training arguments...")
        log.info("Evaluation strategy: {}", eval_strategy)
        log.info("Save strategy: {}", save_strategy)
        log.info("Logging choice: {}", logging_choice)

    args = TrainingArguments(
        output_dir="models/checkpoints",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type=scheduler,
        max_grad_norm=max_grad_norm,
        label_smoothing_factor=label_smoothing,
        optim=optim,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=dataloader_pin_memory,
        remove_unused_columns=remove_unused_columns,
        group_by_length=group_by_length,
        fp16=fp16,
        bf16=bf16,
        dataloader_drop_last=dataloader_drop_last,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        full_determinism=full_determinism,
        eval_strategy=hf_eval_strategy,
        eval_steps=eval_steps if eval_strategy == "steps" else None,
        save_strategy=hf_save_strategy,
        save_steps=eval_steps if save_strategy == "checkpoint" else 500,
        logging_strategy="steps",
        load_best_model_at_end=(save_strategy != "none"),
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        report_to=["wandb"] if logging_choice == "wandb" else [],
        seed=seed
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    if logging_choice in ["loguru", "both"]:
        log.info("Initializing trainer...")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if eval_strategy != "none" else None,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if logging_choice in ["loguru", "both"]:
        log.info("Starting training...")
        log.info("=" * 60)

    trainer.train()

    if logging_choice in ["loguru", "both"]:
        log.info("Training completed!")
        log.info("=" * 60)

    # Final evaluation and save
    if eval_strategy != "none" and eval_ds is not None:
        if logging_choice in ["loguru", "both"]:
            log.info("Running final evaluation...")

        final_metrics = trainer.evaluate()

        if logging_choice in ["loguru", "both"]:
            log.info("Final evaluation results:")
            for key, value in final_metrics.items():
                log.info("  {}: {:.4f}", key, value)

        if logging_choice in ["wandb", "both"]:
            wandb.log({"final_eval": final_metrics})

    # Final save if needed
    if save_strategy == "end":
        if logging_choice in ["loguru", "both"]:
            log.info("Saving final model...")

        Path("models").mkdir(exist_ok=True, parents=True)
        model.save_pretrained("models/final")
        tokenizer.save_pretrained("models/final")

        if logging_choice in ["loguru", "both"]:
            log.info("Model saved to models/final")

        if logging_choice in ["wandb", "both"]:
            # Log model to wandb (optional)
            try:
                artifact = wandb.Artifact(
                    name=f"model-{experiment_name}",
                    type="model",
                    description="Fine-tuned BERT model for financial sentiment analysis"
                )
                artifact.add_dir("models/final")
                wandb.log_artifact(artifact)
            except Exception as e:
                log.warning("Failed to log model to wandb: {}", e)

    if logging_choice in ["loguru", "both"]:
        log.info("Training run completed successfully!")
        log.info("=" * 60)

    if logging_choice in ["wandb", "both"]:
        wandb.finish()

if __name__ == "__main__":
    train()
