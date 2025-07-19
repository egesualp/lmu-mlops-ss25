import json
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Dict, Any

import anyio
import nltk
import pandas as pd
from evidently.metric_preset import TargetDriftPreset, TextEvals
from evidently.report import Report
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from google.cloud import storage

# Global variables to store data
training_data = None
class_names = None
bucket_name = None
storage_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan function to load data and class names before the application starts."""
    global training_data, class_names, bucket_name, storage_client

    print("Starting up sentiment monitoring service...")

    # Download NLTK data if needed
    try:
        nltk.download("words")
        nltk.download("wordnet")
        nltk.download("omw-1.4")
    except Exception as e:
        print(f"Warning: Could not download NLTK data: {e}")

    # Initialize Google Cloud Storage client
    storage_client = storage.Client()
    bucket_name = os.getenv("GCS_BUCKET_NAME", "sentiment-prediction-data")

    # Load training data
    try:
        # Training data bucket (different from predictions bucket)
        training_bucket_name = "mlops-project-bucket1"
        training_blob_path = "data/processed/train.csv"

        print(f"Downloading training data from gs://{training_bucket_name}/{training_blob_path}")

        # Get the training data bucket
        training_bucket = storage_client.bucket(training_bucket_name)
        training_blob = training_bucket.blob(training_blob_path)

        # Check if the blob exists
        if training_blob.exists():
            # Download the training data as string
            csv_content = training_blob.download_as_text()

            # Create DataFrame from CSV content
            from io import StringIO

            training_data = pd.read_csv(StringIO(csv_content))
            print(f"Successfully loaded training data from GCS: {len(training_data)} samples")

        else:
            print(f"Warning: Training data blob does not exist: {training_blob_path}")
            training_data = pd.DataFrame(columns=["text", "label"])

    except Exception as e:
        print(f"Error loading training data: {e}")
        training_data = pd.DataFrame(columns=["text", "label"])

    # Set class names based on training data
    if not training_data.empty:
        class_names = sorted(training_data["label"].unique().tolist())
    else:
        class_names = ["negative", "neutral", "positive"]  # Default sentiment classes

    print(f"Class names: {class_names}")

    yield

    print("Shutting down sentiment monitoring service...")


app = FastAPI(
    title="Sentiment Analysis Monitoring API",
    description="API for monitoring data drift in sentiment predictions using Evidently",
    version="1.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {"message": "Welcome to Sentiment Analysis of Financial Text Monitoring API!"}


async def download_files(n: int) -> List[Dict[str, Any]]:
    """
    Download the N latest prediction files from the GCP bucket.

    Args:
        n: Number of latest files to download

    Returns:
        List of prediction data dictionaries
    """
    try:
        bucket = storage_client.bucket(bucket_name)

        # List all blobs in the predictions folder
        blobs = list(bucket.list_blobs(prefix="predictions/"))

        if not blobs:
            print("No prediction files found in bucket")
            return []

        # Sort blobs by creation time (newest first)
        blobs.sort(key=lambda x: x.time_created, reverse=True)

        # Take the latest n files
        latest_blobs = blobs[:n]

        predictions = []
        for blob in latest_blobs:
            try:
                # Download blob content
                content = blob.download_as_text()
                prediction_data = json.loads(content)
                predictions.append(prediction_data)
            except Exception as e:
                print(f"Error downloading blob {blob.name}: {e}")
                continue

        print(f"Downloaded {len(predictions)} prediction files from GCS")
        return predictions

    except Exception as e:
        print(f"Error downloading files from GCS: {e}")
        return []


def load_latest_files(predictions_path: Path, n: int) -> List[Dict[str, Any]]:
    """
    Load the latest n prediction files from local directory.

    Args:
        predictions_path: Path to the predictions directory
        n: Number of latest files to load

    Returns:
        List of prediction data dictionaries
    """
    try:
        if not predictions_path.exists():
            print(f"Predictions path does not exist: {predictions_path}")
            return []

        # Get all JSON files in the directory
        json_files = list(predictions_path.glob("*.json"))

        if not json_files:
            print("No JSON files found in predictions directory")
            return []

        # Sort by modification time (newest first)
        json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Take the latest n files
        latest_files = json_files[:n]

        predictions = []
        for file_path in latest_files:
            try:
                with open(file_path, "r") as f:
                    prediction_data = json.load(f)
                    predictions.append(prediction_data)
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
                continue
        print(f"Loaded {len(predictions)} prediction files from local directory")
        return predictions

    except Exception as e:
        print(f"Error loading files from local directory: {e}")
        return []


async def run_analysis(n_predictions: int = 100) -> Report:
    """
    Run data drift analysis comparing training data with latest predictions.

    Args:
        n_predictions: Number of latest predictions to analyze

    Returns:
        Evidently Report object
    """
    global training_data, class_names

    # Download latest predictions from GCS

    predictions = await download_files(n_predictions)

    if not predictions:
        print("No predictions found, cannot run analysis")
        # Create empty dataframe for analysis
        prediction_df = pd.DataFrame(columns=["text", "predicted_label", "confidence_score"])
    else:
        # Convert predictions to DataFrame
        prediction_df = pd.DataFrame(
            [
                {
                    "text": pred.get("text", ""),
                    "predicted_label": pred.get("predicted_label", ""),
                    "confidence_score": pred.get("confidence_score", 0.0),
                    "timestamp": pred.get("timestamp", ""),
                }
                for pred in predictions
            ]
        )

    # Prepare reference data (training data)
    reference_df = training_data.copy()
    if "predicted_label" not in reference_df.columns:
        reference_df["predicted_label"] = reference_df["label"]  # Use true labels as reference
    if "confidence_score" not in reference_df.columns:
        reference_df["confidence_score"] = 1.0  # Assume perfect confidence for training data

    # Ensure we have some data for analysis
    if reference_df.empty:
        print("Warning: No reference data available for analysis")
        reference_df = pd.DataFrame(
            {"text": ["Sample text"], "predicted_label": ["neutral"], "confidence_score": [1.0]}
        )

    if prediction_df.empty:
        print("Warning: No prediction data available for analysis")
        prediction_df = pd.DataFrame(
            {"text": ["Sample prediction text"], "predicted_label": ["neutral"], "confidence_score": [0.5]}
        )

    # Create Evidently report
    report = Report(metrics=[TargetDriftPreset(columns=["predicted_label"]), TextEvals(column_name="text")])

    try:
        # Run the report
        report.run(reference_data=reference_df, current_data=prediction_df)
        print("Data drift analysis completed successfully")
    except Exception as e:
        print(f"Error running analysis: {e}")
        # Create a minimal report if analysis fails
        from evidently.metrics import DataDriftTable

        report = Report(metrics=[DataDriftTable()])
        report.run(reference_data=reference_df.head(1), current_data=prediction_df.head(1))

    return report


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global training_data, class_names, storage_client

    return {
        "status": "healthy",
        "training_data_loaded": training_data is not None and not training_data.empty,
        "training_samples": len(training_data) if training_data is not None else 0,
        "class_names": class_names,
        "gcs_client_ready": storage_client is not None,
        "bucket_name": bucket_name,
    }


@app.get("/stats")
async def get_stats():
    """Get current statistics about the monitoring service."""
    global training_data

    stats = {
        "training_data_size": len(training_data) if training_data is not None else 0,
        "class_names": class_names,
        "bucket_name": bucket_name,
    }

    if training_data is not None and not training_data.empty:
        stats["label_distribution"] = training_data["label"].value_counts().to_dict()

    return stats


@app.get("/report", response_class=HTMLResponse)
async def get_report(n_predictions: int = 100):
    """
    Generate and return data drift analysis report.

    Args:
        n_predictions: Number of latest predictions to analyze (default: 100)
    """
    try:
        # Run the analysis
        report = await run_analysis(n_predictions)

        # Get HTML representation of the report
        html_report = report.get_html()

        return HTMLResponse(content=html_report)

    except Exception as e:
        error_html = f"""
        <html>
            <body>
                <h1>Error generating report</h1>
                <p>An error occurred while generating the data drift report:</p>
                <pre>{str(e)}</pre>
                <p>Please check the logs for more details.</p>
            </body>
        </html>
        """
        return HTMLResponse(content=error_html, status_code=500)


@app.get("/report/json")
async def get_report_json(n_predictions: int = 100):
    """
    Generate and return data drift analysis report as JSON.

    Args:
        n_predictions: Number of latest predictions to analyze (default: 100)
    """
    try:
        # Run the analysis
        report = await run_analysis(n_predictions)

        # Get JSON representation of the report
        json_report = report.json()

        return {"report": json_report, "status": "success"}

    except Exception as e:
        return {"error": str(e), "status": "error", "message": "Failed to generate drift analysis report"}
