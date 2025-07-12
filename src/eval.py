import torch
import numpy as np
from pathlib import Path
import typer
from typing import Optional
from loguru import logger as log
import sys
import os
import wandb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt

# Import for Hugging Face models
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers.data.data_collator import DataCollatorWithPadding
import evaluate

# Import for old PyTorch models
from data import MyDataset, create_hf_datasets
from model import Classifier

# Load metrics
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def setup_logging(logging_choice: str):
    """Setup logging based on choice."""
    if logging_choice in ["loguru", "both"]:
        # Remove existing handlers
        log.remove()

        # Add console handler
        log.add(
            sys.stdout,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level="INFO",
            colorize=True
        )

        log.info("Loguru logging configured")

    if logging_choice in ["wandb", "both"]:
        wandb.init(
            project="financial-sentiment-eval",
            entity=None,
            tags=["bert", "sentiment", "financial", "evaluation"],
            notes="BERT evaluation for financial sentiment analysis",
            resume="allow"
        )
        log.info("Wandb logging configured")

def evaluate_hf_model(
    model_path: str,
    data_dir: Path,
    pretrained_model: Optional[str] = None,
    max_rows: Optional[int] = None,
    batch_size: int = 16,
    logging_choice: str = "loguru"
):
    """Evaluate Hugging Face model."""
    log.info("Loading Hugging Face model from: {}", model_path)

    # Load model first to get the config
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # If pretrained_model not provided, try to get it from model config
    if pretrained_model is None:
        if hasattr(model.config, 'architectures') and model.config.architectures:
            # Try to infer from model architecture
            if 'BertForSequenceClassification' in model.config.architectures:
                pretrained_model = "bert-base-uncased"
            elif 'DistilBertForSequenceClassification' in model.config.architectures:
                pretrained_model = "distilbert-base-uncased"
            else:
                # Default fallback
                pretrained_model = "bert-base-uncased"
                log.warning("Could not determine pretrained model, using bert-base-uncased as default")
        else:
            pretrained_model = "bert-base-uncased"
            log.warning("Could not determine pretrained model, using bert-base-uncased as default")

    log.info("Using pretrained model: {}", pretrained_model)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    # Create datasets
    train_ds, eval_ds, _, num_labels = create_hf_datasets(
        data_dir, pretrained_model, max_rows
    )

    # Setup evaluation arguments
    eval_args = TrainingArguments(
        output_dir="./temp_eval",
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        report_to=[]  # No wandb for evaluation
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Evaluate
    log.info("Running evaluation...")
    results = trainer.evaluate()

    # Get predictions for detailed analysis
    predictions = trainer.predict(eval_ds)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids

    # Calculate additional metrics
    detailed_metrics = calculate_detailed_metrics(true_labels, pred_labels)

    # Log results
    log.info("Evaluation Results:")
    log.info("Accuracy: {:.4f}", results['eval_accuracy'])
    log.info("Loss: {:.4f}", results['eval_loss'])

    for metric, value in detailed_metrics.items():
        log.info("{}: {:.4f}", metric, value)

    # Log to wandb if enabled
    if logging_choice in ["wandb", "both"]:
        wandb.log({
            "eval_accuracy": results['eval_accuracy'],
            "eval_loss": results['eval_loss'],
            **detailed_metrics
        })

    return results, detailed_metrics, true_labels, pred_labels

def evaluate_pytorch_model(
    model_path: str,
    data_dir: Path,
    max_rows: Optional[int] = None,
    batch_size: int = 16,
    logging_choice: str = "loguru"
):
    """Evaluate PyTorch model."""
    log.info("Loading PyTorch model from: {}", model_path)

    # Load test dataset
    test_dataset = MyDataset(Path(f"{data_dir}/test.csv"), max_rows)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # Load model
    model = Classifier()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    log.info("Running evaluation...")
    with torch.no_grad():
        for batch in test_loader:
            texts = batch["text"]
            labels = batch["label"].to(device)
            outputs = model(texts)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    detailed_metrics = calculate_detailed_metrics(all_labels, all_preds)

    # Log results
    log.info("Evaluation Results:")
    log.info("Accuracy: {:.4f}", accuracy)

    for metric, value in detailed_metrics.items():
        log.info("{}: {:.4f}", metric, value)

    # Log to wandb if enabled
    if logging_choice in ["wandb", "both"]:
        wandb.log({
            "eval_accuracy": accuracy,
            **detailed_metrics
        })

    return {"accuracy": accuracy}, detailed_metrics, all_labels, all_preds

def compute_metrics(eval_pred):
    """Compute metrics for Hugging Face evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "precision": precision_metric.compute(predictions=predictions, references=labels, average="weighted")["precision"],
        "recall": recall_metric.compute(predictions=predictions, references=labels, average="weighted")["recall"],
        "f1": f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    }

def calculate_detailed_metrics(y_true, y_pred):
    """Calculate detailed classification metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def print_classification_report(y_true, y_pred, labels=None):
    """Print detailed classification report."""
    log.info("Detailed Classification Report:")
    report = classification_report(y_true, y_pred, target_names=labels, digits=4)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    log.info("Confusion Matrix:")
    print(cm)

def save_confusion_matrix(y_true, y_pred, save_path: str = "confusion_matrix.png"):
    """Save confusion matrix as a plot."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    log.info("Confusion matrix saved to: {}", save_path)

def evaluate_model(
    model_path: str = typer.Option("models/final", help="Path to the trained model"),
    model_type: str = typer.Option("hf", help="Model type: hf (Hugging Face) or pytorch"),
    data_dir: str = typer.Option("data/processed", help="Path to the processed dataset"),
    batch_size: int = typer.Option(16, help="Batch size for evaluation"),
    max_rows: Optional[int] = typer.Option(None, help="Maximum number of rows to load"),
    logging: str = typer.Option("loguru", help="Logging choice: loguru, wandb, both, none"),
    save_plots: bool = typer.Option(True, help="Save confusion matrix plot"),
    pretrained_model: Optional[str] = typer.Option(None, help="Pretrained model name for HF models (auto-detected if not provided)"),
):
    """
    Evaluate the trained model on the test set and print comprehensive metrics.
    """
    # Setup logging
    if logging != "none":
        setup_logging(logging)

    log.info("=" * 60)
    log.info("Starting Model Evaluation")
    log.info("=" * 60)
    log.info("Model path: {}", model_path)
    log.info("Model type: {}", model_type)
    log.info("Data directory: {}", data_dir)
    log.info("Batch size: {}", batch_size)
    log.info("Max rows: {}", max_rows if max_rows else "All")
    log.info("=" * 60)

    data_path = Path(data_dir)

    # Validate model_type
    if model_type not in ["hf", "pytorch"]:
        log.error("Invalid model_type: {}. Must be 'hf' or 'pytorch'", model_type)
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'hf' or 'pytorch'")

    try:
        if model_type == "hf":
            # Evaluate Hugging Face model
            results, detailed_metrics, true_labels, pred_labels = evaluate_hf_model(
                model_path=model_path,
                data_dir=data_path,
                pretrained_model=pretrained_model,
                max_rows=max_rows,
                batch_size=batch_size,
                logging_choice=logging
            )
        else:
            # Evaluate PyTorch model
            results, detailed_metrics, true_labels, pred_labels = evaluate_pytorch_model(
                model_path=model_path,
                data_dir=data_path,
                max_rows=max_rows,
                batch_size=batch_size,
                logging_choice=logging
            )

        # Print detailed classification report
        print_classification_report(true_labels, pred_labels)

        # Save confusion matrix if requested
        if save_plots:
            save_confusion_matrix(true_labels, pred_labels)

        log.info("=" * 60)
        log.info("Evaluation completed successfully!")
        log.info("=" * 60)

    except Exception as e:
        log.error("Evaluation failed: {}", e)
        raise

    finally:
        if logging in ["wandb", "both"]:
            wandb.finish()

if __name__ == "__main__":
    typer.run(evaluate_model)
