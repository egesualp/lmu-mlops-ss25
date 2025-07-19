import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import typer
from loguru import logger

app = typer.Typer()


def dataset_statistics(datadir: Path = Path("data/processed")) -> None:
    """Compute and log dataset statistics and plots for text-based sentiment data."""

    for split in ["train", "test", "eval"]:
        file_path = datadir / f"{split}.csv"

        if not file_path.exists():
            logger.warning(f"{split}.csv not found in {datadir}")
            continue

        df = pd.read_csv(file_path)

        print(f"\n📄 {split.capitalize()} Dataset")
        print(f"Samples: {len(df)}")
        print(f"Columns: {list(df.columns)}")

        # Check for label column
        label_col = "label" if "label" in df.columns else "Sentiment"
        if label_col not in df.columns:
            logger.error(f"Label column '{label_col}' not found in {split}")
            continue

        # Text length distribution
        df["text_len"] = df["text"].str.len()
        plt.hist(df["text_len"], bins=30)
        plt.title(f"{split.capitalize()} Text Length Distribution")
        plt.xlabel("Text length")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(f"./reports/figures/{split}_text_length.png")
        plt.close()

        # Label distribution
        label_counts = df[label_col].value_counts()
        label_counts.plot(kind="bar")
        plt.title(f"{split.capitalize()} Label Distribution")
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"./reports/figures/{split}_label_distribution.png")
        plt.close()

        print(f"Label counts:\n{label_counts.to_string()}")
        print(f"Saved: {split}_text_length.png, {split}_label_distribution.png")


if __name__ == "__main__":
    app.command()(dataset_statistics)
    app()
