import torch
import typer
from torch import nn, optim
from torch.utils.data import DataLoader

from data import MyDataset
from model import Classifier

app = typer.Typer()


@app.command()
def train(
    data_dir: str = typer.Option("data/processed", help="Path to the processed dataset"),
    max_rows: int = typer.Option(200, help="Maximum number of rows to load from dataset"),
    batch_size: int = typer.Option(16, help="Batch size for training"),
    epochs: int = typer.Option(5, help="Number of training epochs"),
    lr: float = typer.Option(1e-4, help="Learning rate"),
) -> None:
    """
    Train a text classification model.
    """
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

    print("Training complete.")


if __name__ == "__main__":
    app()
