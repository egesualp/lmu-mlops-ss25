import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from model import Classifier
from data import MyDataset
import random
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
import hydra
from omegaconf import DictConfig
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
from loguru import logger as log
import os

class SentimentModule(pl.LightningModule):
    def __init__(self, pretrained_model_name: str, lr: float, dropout: float):
        super().__init__()
        self.save_hyperparameters()
        self.model = Classifier(pretrained_model_name=pretrained_model_name, dropout=dropout)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, texts):
        return self.model(texts)

    def training_step(self, batch, batch_idx):
        texts = batch["text"]
        labels = batch["label"]
        logits = self(texts)
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        texts = batch["text"]
        labels = batch["label"]
        logits = self(texts)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg: DictConfig):
    # Basic config values
    experiment_name = cfg.experiment_name
    data_dir = cfg.data.data_dir
    max_rows = cfg.data.get("max_rows", None)  # Optional: use None to load all data
    batch_size = cfg.batch_size
    epochs = cfg.epochs
    lr = cfg.model.lr
    seed = cfg.seed
    pretrained_model = cfg.model.pretrained_model
    dropout = cfg.model.dropout
    eval_strategy = cfg.get("eval_strategy", "none")
    eval_steps = int(cfg.get("eval_steps", 100))
    logging_choice = cfg.get("logging", None)
    save_strategy = cfg.get("save_strategy", "end")  # 'end', 'checkpoint', or 'none'

    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Configure loguru to write to file in current (Hydra) work dir if needed
    if logging_choice == "loguru":
        hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        log.add(os.path.join(hydra_path, "train_config.log"))
        log.info(cfg)
        log.info("Experiment: {} | Strategy: {} | Save: {}", experiment_name, eval_strategy, save_strategy)

    # Build datasets and dataloaders
    train_file = Path(data_dir) / "train.csv"
    eval_file = Path(data_dir) / "eval.csv"
    train_ds = MyDataset(train_file, max_rows)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=19)

    eval_dl = None
    if eval_strategy != "none":
        eval_ds = MyDataset(eval_file, max_rows)
        eval_dl = DataLoader(eval_ds, batch_size=batch_size, num_workers=19)

    # Logger setup for PyTorch Lightning
    wandb_logger = WandbLogger(project=experiment_name) if logging_choice == "wandb" else False

    # Checkpoint callback (only if save_strategy is 'checkpoint')
    callbacks = []
    if save_strategy == "checkpoint":
        checkpoint_callback = ModelCheckpoint(
            dirpath="models/checkpoints",
            filename="checkpoint-{epoch:02d}-{val_loss:.2f}",
            save_top_k=1,
            monitor="val_loss" if eval_strategy != "none" else None,
        )
        callbacks.append(checkpoint_callback)

    # Trainer arguments
    trainer_kwargs = {
        "max_epochs": epochs,
        "logger": wandb_logger if logging_choice == "wandb" else False,
        "default_root_dir": "models" if save_strategy != "none" else None,
        "callbacks": callbacks,
        "enable_checkpointing": save_strategy == "checkpoint"
    }
    if eval_strategy == "steps":
        trainer_kwargs["val_check_interval"] = eval_steps
    elif eval_strategy == "epoch":
        trainer_kwargs["check_val_every_n_epoch"] = 1
    else:
        trainer_kwargs["check_val_every_n_epoch"] = 0

    trainer = pl.Trainer(**trainer_kwargs)

    # Lightning module
    lit_model = SentimentModule(pretrained_model_name=pretrained_model, lr=lr, dropout=dropout)

    # Train
    trainer.fit(lit_model, train_dl, eval_dl)

    # Save model weights (only if save_strategy is 'end')
    if save_strategy == "end":
        Path("models").mkdir(parents=True, exist_ok=True)
        torch.save(lit_model.model.state_dict(), "models/model.pt")
        if logging_choice == "loguru":
            log.info("Training finished. Model saved to models/model.pt")
    elif save_strategy == "none":
        if logging_choice == "loguru":
            log.info("Training finished. No model saved.")

    if logging_choice == "wandb":
        wandb.finish()

if __name__ == "__main__":
    train()
