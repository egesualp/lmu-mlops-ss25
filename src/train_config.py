import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from model import Classifier
from data import MyDataset
import random
import numpy as np
import torch
import typer
from omegaconf import OmegaConf
from torch import nn, optim
from torch.utils.data import DataLoader
import wandb
import hydra
import os
from omegaconf import DictConfig, OmegaConf

from data import MyDataset
from model import Classifier

@hydra.main(config_path="../conf", config_name="config")
def train(cfg: DictConfig):
    """
    Train a text classification model.
    """
    experiment_name = cfg.experiment_name
    data_dir = cfg.data.data_dir
    max_rows = cfg.data.max_rows
    batch_size = cfg.batch_size
    epochs = cfg.epochs
    lr = cfg.model.lr
    seed = cfg.seed
    pretrained_model = cfg.model.pretrained_model
    dropout = cfg.model.dropout



    wandb.init(
        project=experiment_name,
        config={
            "data_dir": data_dir,
            "max_rows": max_rows,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "seed": seed,
            "pretrained_model": pretrained_model,
            "dropout": dropout,
        },
    )

    # Seed everything for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset = MyDataset(data_dir, max_rows)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Pass pretrained_model and dropout to your Classifier
    model = Classifier(pretrained_model_name=pretrained_model, dropout=dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    wandb.watch(model, log="all", log_freq=10)

    print("Training started...")
    model.train()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        running_loss = 0.0

        for batch in dataloader:
            texts = batch["text"]
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(texts).to(device)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "loss": avg_loss})

    print("Training complete.")

    torch.save(model.state_dict(), "models/model.pt")
    wandb.save("model.pt", policy="now")


if __name__ == "__main__":
    config = OmegaConf.load("conf/config.yaml")
    train()
