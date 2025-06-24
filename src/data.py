import os
from pathlib import Path

import pandas as pd
import torch
import typer
from torch.utils.data import Dataset

app = typer.Typer()


class MyDataset(Dataset):
    """Custom dataset for financial sentiment analysis."""

    def __init__(self, data_path: Path, max_rows: int = None) -> None:
        file_path = f"{data_path}/sentiment_data.csv"

        self.data = pd.read_csv(file_path)
        if max_rows is not None:
            self.data = self.data.head(max_rows)

        # Create label mapping based on the 'label' column (fixed here)
        label_set = sorted(self.data["label"].unique())
        self.label2id = {label: i for i, label in enumerate(label_set)}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        row = self.data.iloc[index]
        label_id = self.label2id[row["label"]]
        label_tensor = torch.tensor(label_id, dtype=torch.long)
        return {"text": row["text"], "label": label_tensor}


def preprocess_data(input_path: Path, output_folder: Path) -> None:
    """Preprocess raw CSV and save cleaned version in output_folder/processed/"""
    raw_file = input_path / "sentiment_data.csv"
    if not raw_file.exists():
        raise FileNotFoundError(f"'sentiment_data.csv' not found at {raw_file}")

    df = pd.read_csv(raw_file)
    df = df.dropna(subset=["Sentence", "Sentiment"])
    df = df[["Sentence", "Sentiment"]].rename(columns={"Sentence": "text", "Sentiment": "label"})

    processed_dir = output_folder / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_file = processed_dir / "sentiment_data.csv"
    df.to_csv(output_file, index=False)

    print(f"Preprocessed data saved to {output_file}")


@app.command()
def preprocess(input_path: Path, output_folder: Path) -> None:
    """
    Preprocess raw data located at `input_path` and save cleaned data to `output_folder/processed`.
    """
    print("Preprocessing data...")
    preprocess_data(input_path, output_folder)


@app.command()
def load(data_path: Path, max_rows: int = typer.Option(None, help="Limit to first N rows")) -> None:
    """
    Load dataset from processed data path and print basic info.
    """
    dataset = MyDataset(data_path, max_rows)
    typer.echo(f"Loaded dataset with {len(dataset)} samples")
    typer.echo(f"Labels mapping: {dataset.label2id}")


if __name__ == "__main__":
    app()
