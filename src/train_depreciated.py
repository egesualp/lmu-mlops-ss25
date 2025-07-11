import torch
import typer
from torch import nn, optim
from torch.utils.data import DataLoader
from pathlib import Path

from data import MyDataset
from model import Classifier

app = typer.Typer()


@app.command()
def train(
    data_dir: str = typer.Option("data/processed", help="Path to the processed dataset"),
    max_rows: int = typer.Option(None, help="Maximum number of rows to load from dataset"),
    batch_size: int = typer.Option(16, help="Batch size for training"),
    epochs: int = typer.Option(5, help="Number of training epochs"),
    lr: float = typer.Option(1e-4, help="Learning rate"),
    eval_strategy: str = typer.Option("none", help="Evaluation strategy: 'epoch', 'steps', or 'none'", show_choices=True),
    eval_steps: int = typer.Option(100, help="Evaluate every N steps if eval_strategy is 'steps'"),
) -> None:
    """
    Train a text classification model.
    """
    train_file = Path(data_dir) / "train.csv"
    eval_file = Path(data_dir) / "eval.csv"
    dataset = MyDataset(train_file, max_rows)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Prepare eval dataloader if needed
    if eval_strategy != "none":
        eval_dataset = MyDataset(eval_file, max_rows)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size)

    model = Classifier()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    def evaluate_model():
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in eval_loader:
                texts = batch["text"]
                labels = batch["label"].to(device)
                outputs = model(texts)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
        print(f"Eval Accuracy: {acc:.4f}")
        model.train()

    print("Training started...")
    model.train()

    global_step = 0
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
            global_step += 1

            if eval_strategy == "steps" and global_step % eval_steps == 0:
                print(f"Step {global_step}: Running evaluation...")
                evaluate_model()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader):.4f}")
        if eval_strategy == "epoch":
            print(f"Epoch {epoch + 1}: Running evaluation...")
            evaluate_model()

    print("Training complete.")


if __name__ == "__main__":
    app()
