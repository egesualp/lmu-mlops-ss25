import torch
from torch.utils.data import DataLoader
from pathlib import Path
from data import MyDataset
from model import Classifier
import typer

def accuracy_score_torch(y_true, y_pred):
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    return (y_true == y_pred).float().mean().item()

def classification_report_torch(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    report = {}
    for label in labels:
        tp = sum((yt == label and yp == label) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != label and yp == label) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == label and yp != label) for yt, yp in zip(y_true, y_pred))
        support = sum(yt == label for yt in y_true)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        report[label] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support
        }
    # Print report
    print("label  precision  recall  f1-score  support")
    for label, metrics in report.items():
        print(f"{label:5}  {metrics['precision']:.2f}     {metrics['recall']:.2f}   {metrics['f1-score']:.2f}    {metrics['support']}")
    return report

def evaluate(
    model_path: str = typer.Option("models/model.pt", help="Path to the trained model file"),
    data_dir: str = typer.Option("data/processed", help="Path to the processed dataset"),
    batch_size: int = typer.Option(16, help="Batch size for evaluation"),
    max_rows: int = typer.Option(None, help="Maximum number of rows to load from test set"),
):
    """
    Evaluate the trained model on the test set and print metrics.
    """
    # Load test set (MyDataset expects a file path)
    test_dataset = MyDataset(Path(f"{data_dir}/test.csv"), max_rows)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Load model
    model = Classifier()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            texts = batch["text"]
            labels = batch["label"].to(device)
            outputs = model(texts)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score_torch(all_labels, all_preds)
    print(f"Test Accuracy: {acc:.4f}")
    classification_report_torch(all_labels, all_preds)

if __name__ == "__main__":
    typer.run(evaluate)
