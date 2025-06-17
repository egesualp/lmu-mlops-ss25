import typer
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from model import Classifier
from data import MyDataset
from omegaconf import OmegaConf
import random
import numpy as np



def train(data_dir, max_rows, batch_size, epochs, lr, seed):
    """
    Train a text classification model.
    """
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

    print("Training started...")
    model.train()

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
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

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader):.4f}")

    print("Training complete.")

if __name__ == "__main__":
    config = OmegaConf.load('conf/config.yaml')
    train(config.data.data_dir, config.data.max_rows, config.batch_size, config.epochs, config.model.lr, config.seed)
