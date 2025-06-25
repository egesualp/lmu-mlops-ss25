import random
import numpy as np
import torch
import typer
from omegaconf import OmegaConf
from torch import nn, optim
from torch.utils.data import DataLoader
import wandb

from data import MyDataset
from model import Classifier

app = typer.Typer()

def train(
    data_dir: str,
    max_rows: int,
    batch_size: int,
    epochs: int,
    lr: float,
    seed: int,
    experiment_name: str
) -> None:
    """
    Train a text classification model.
    """

    wandb.init(
        project=experiment_name,
        config={
            "data_dir": data_dir,
            "max_rows": max_rows,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "seed": seed,
        },
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset = MyDataset(data_dir, max_rows)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Classifier()
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

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader):.4f}")
        wandb.log({"epoch": epoch + 1, "loss": running_loss})

    print("Training complete.")

    torch.save(model.state_dict(), "models/model.pt")
    wandb.save("model.pt", policy="now")

@app.command()
def train_config(
    config_file: str = typer.Argument("conf/config.yaml", help="Path to YAML config")
) -> None:
    """
    Run training using a config YAML file.
    """
    config = OmegaConf.load(config_file)
    train(
        config.data.data_dir,
        config.data.max_rows,
        config.batch_size,
        config.epochs,
        config.model.lr,
        config.seed,
        config.experiment_name,
    )

if __name__ == "__main__":
    app()
