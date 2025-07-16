import os
from pathlib import Path

import kagglehub
from kagglehub import dataset_load, KaggleDatasetAdapter
import pandas as pd
import torch
import typer
from torch.utils.data import Dataset
from loguru import logger
from typing import Optional
from transformers import AutoTokenizer

app = typer.Typer()


class MyDataset(Dataset):
    """Custom dataset for financial sentiment analysis."""

    def __init__(self, data_path: Path, max_rows: Optional[int] = None, label2id: Optional[dict] = None) -> None:
        # Read CSV and handle empty files
        try:
            self.data = pd.read_csv(data_path)
            if self.data.empty:
                raise pd.errors.EmptyDataError("No data found in CSV file")
        except pd.errors.EmptyDataError:
            raise pd.errors.EmptyDataError("No data found in CSV file")

        # Handle max_rows parameter
        if max_rows is not None:
            if max_rows <= 0:
                self.data = pd.DataFrame(columns=self.data.columns)  # Empty dataframe
            else:
                self.data = self.data.head(max_rows)

        # Create or use provided label mapping
        if label2id is not None:
            self.label2id = label2id
        else:
            label_set = sorted(self.data["label"].unique())
            self.label2id = {label: i for i, label in enumerate(label_set)}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        row = self.data.iloc[index]
        label_id = self.label2id[row["label"]]
        label_tensor = torch.tensor(label_id, dtype=torch.long)
        return {"text": row["text"], "label": label_tensor}

class HFDataset(Dataset):
    """A Dataset that tokenizes on-the-fly using a HuggingFace tokenizer."""

    def __init__(self, base_ds: MyDataset, tokenizer: AutoTokenizer):
        self.base_ds = base_ds
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        item = self.base_ds[idx]
        encoding = self.tokenizer(item["text"], truncation=True, padding=False)
        encoding["labels"] = item["label"]
        return encoding

def preprocess_data(
    input_path: Path,
    output_folder: Path = Path("data/processed"),
    train_ratio: float = 0.8,
    test_ratio: float = 0.1,
    eval_ratio: float = 0.1,
    random_state: int = 42,
) -> None:
    """Preprocess raw CSV and save train, test, eval splits in output_folder/"""
    logger.info("Starting preprocessing with train_ratio={}, test_ratio={}, eval_ratio={}, random_state={}", train_ratio, test_ratio, eval_ratio, random_state)
    # Validate ratios
    total = train_ratio + test_ratio + eval_ratio
    if not abs(total - 1.0) < 1e-6:
        logger.error("Ratios must sum to 1.0 (got {:.4f})", total)
        raise ValueError(f"Ratios must sum to 1.0 (got {total:.4f})")
    logger.info("Ratios validated successfully.")
    raw_file = input_path / "data.csv"
    if not raw_file.exists():
        logger.error("'data.csv' not found at {}", raw_file)
        raise FileNotFoundError(f"'data.csv' not found at {raw_file}")

    df = pd.read_csv(raw_file)
    df = df.dropna(subset=["Sentence", "Sentiment"])
    df = df[["Sentence", "Sentiment"]].rename(columns={"Sentence": "text", "Sentiment": "label"})  # type: ignore

    # Shuffle the data for random splitting
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Calculate split indices
    n = len(df)
    train_end = int(train_ratio * n)
    test_end = train_end + int(test_ratio * n)

    train_df = df.iloc[:train_end]
    test_df = df.iloc[train_end:test_end]
    eval_df = df.iloc[test_end:]

    output_folder.mkdir(parents=True, exist_ok=True)

    train_file = output_folder / "train.csv"
    test_file = output_folder / "test.csv"
    eval_file = output_folder / "eval.csv"

    train_df.to_csv(train_file, index=False)
    logger.info("Train set saved to {} ({} rows)", train_file, len(train_df))
    test_df.to_csv(test_file, index=False)
    logger.info("Test set saved to {} ({} rows)", test_file, len(test_df))
    eval_df.to_csv(eval_file, index=False)
    logger.info("Eval set saved to {} ({} rows)", eval_file, len(eval_df))

    # Print label distribution for sanity check
    for name, split in zip(["Train", "Test", "Eval"], [train_df, test_df, eval_df]):
        label_counts = split["label"].value_counts().to_dict()
        label_props = (split["label"].value_counts(normalize=True).round(3)).to_dict()
        logger.info(f"{name} label distribution (counts): {label_counts}")
        logger.info(f"{name} label distribution (proportions): {label_props}")


@app.command()
def preprocess(
    input_path: Path = typer.Option(Path("data/raw"), help="Folder to extract raw data"),
    output_folder: Path = typer.Option(Path("data/processed"), help="Folder to save processed splits"),
    train_ratio: float = typer.Option(0.8, help="Proportion of data for training"),
    test_ratio: float = typer.Option(0.1, help="Proportion of data for testing"),
    eval_ratio: float = typer.Option(0.1, help="Proportion of data for evaluation"),
    random_state: int = typer.Option(42, help="Random seed for shuffling and splitting"),
) -> None:
    """
    Preprocess raw data located at `input_path` and save cleaned data to `output_folder`.
    """
    try:
        preprocess_data(input_path, output_folder, train_ratio, test_ratio, eval_ratio, random_state)
    except ValueError as e:
        logger.error("Error: {}", e)
        raise typer.Exit(code=1)


@app.command()
def load(
    data_path: Path = typer.Option(Path("data/processed"), help="Path to processed data directory"),
    split: str = typer.Option("train", help="Which split to load", show_choices=True, case_sensitive=False),
    max_rows: int = typer.Option(None, help="Limit to first N rows"),
) -> None:
    """
    Load dataset from processed data path and print basic info.
    """
    split = split.lower()
    assert split in {"train", "test", "eval"}, "split must be one of: train, test, eval"
    file_path = data_path / f"{split}.csv"
    dataset = MyDataset(file_path, max_rows)
    typer.echo(f"Loaded {split} dataset with {len(dataset)} samples")
    typer.echo(f"Labels mapping: {dataset.label2id}")

@app.command()
def download() -> None:
    """
    Download the financial sentiment dataset from KaggleHub and copy CSVs to data/raw.
    """
    try:
        typer.echo("Downloading dataset from KaggleHub...")
        df = dataset_load(
            KaggleDatasetAdapter.PANDAS,
            "sbhatti/financial-sentiment-analysis",
            "data.csv"
        )

    except Exception as e:
        typer.echo(f"Failed to download dataset: {e}", err=True)
        return

    dest = Path("data/raw")
    dest.mkdir(parents=True, exist_ok=True)

    target = dest / "data.csv"
    df.to_csv(target, index=False)
    typer.echo(f"Saved DataFrame to {target}")


    """try:
        typer.echo("Downloading dataset from KaggleHub...")
        path_raw = kagglehub.dataset_download("sbhatti/financial-sentiment-analysis")
        path = Path(path_raw)
    except Exception as e:
        typer.echo(f"Failed to download dataset: {e}", err=True)
        return

    dest = Path("data/raw")
    dest.mkdir(parents=True, exist_ok=True)

    for file in path.glob("*.csv"):
        target = dest / file.name
        file.replace(target)
        typer.echo(f"Copied: {file.name} → {target}")

    typer.echo("Download and copy complete.")"""


def create_hf_datasets(data_dir: Path, pretrained_model: str, max_rows: Optional[int] = None):
    """Create HuggingFace datasets for training and evaluation."""
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    # Create base datasets with consistent label mapping
    train_base = MyDataset(data_dir / "train.csv", max_rows)
    eval_base = MyDataset(data_dir / "eval.csv", max_rows, label2id=train_base.label2id)

    # Create HF datasets
    train_ds = HFDataset(train_base, tokenizer)
    eval_ds = HFDataset(eval_base, tokenizer)

    return train_ds, eval_ds, tokenizer, len(train_base.label2id)


if __name__ == "__main__":
    download()
    app()
