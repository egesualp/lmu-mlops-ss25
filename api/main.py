from fastapi import FastAPI, BackgroundTasks, HTTPException
from google.cloud import storage
from pydantic import BaseModel
from contextlib import asynccontextmanager
from transformers import AutoTokenizer
from prometheus_client import CollectorRegistry, Counter, Histogram, Summary, make_asgi_app
import wandb
import onnxruntime as ort
import numpy as np
import os
import time
import datetime
import json

MY_REGISTRY = CollectorRegistry()
# Define Prometheus metrics
error_counter = Counter("prediction_error_total", "Number of prediction errors", registry=MY_REGISTRY)
request_counter = Counter("prediction_requests_total", "Number of prediction requests", registry=MY_REGISTRY)
request_latency = Histogram("prediction_latency_seconds", "Prediction latency in seconds", registry=MY_REGISTRY)
review_summary = Summary("review_length_summary", "Length of input reviews", registry=MY_REGISTRY)


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    score: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    run = wandb.init(
        project="financial-sentiment-bert",
        entity="cbrkcan90-ludwig-maximilianuniversity-of-munich",
        job_type="inference",
    )

    artifact = run.use_artifact(
        "cbrkcan90-ludwig-maximilianuniversity-of-munich/financial-sentiment-bert/financial-bert-onnx:latest",
        type="model",
    )
    artifact_dir = artifact.download()
    print(f"ONNX model downloaded to: {artifact_dir}")

    tokenizer = AutoTokenizer.from_pretrained(artifact_dir)

    session = ort.InferenceSession(os.path.join(artifact_dir, "model.onnx"))
    input_names = [inp.name for inp in session.get_inputs()]
    output_name = session.get_outputs()[0].name

    app.state.tokenizer = tokenizer
    app.state.session = session
    app.state.input_names = input_names
    app.state.output_name = output_name
    app.state.label_map = {0: "negative", 1: "neutral", 2: "positive"}

    yield

    run.finish()
    print("W&B run closed, model cleanup complete.")


app = FastAPI(
    title="Financial Sentiment Analysis API",
    description="API for predicting sentiment of financial text",
    version="2.0",
    lifespan=lifespan,
)

# Mount Prometheus metrics endpoint
app.mount("/metrics", make_asgi_app(registry=MY_REGISTRY))


def save_prediction_gcloud(text: str, label: str, score: float, timestamp: str = None):
    """
    Save predictions to Google Cloud Storage
    """
    try:
        # Get bucket name use default
        bucket_name = "sentiment-prediction-data"

        # Initialize Google Cloud Storage client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # Generate timestamp if not provided
        if timestamp is None:
            timestamp = datetime.datetime.now().isoformat()

        # Prepare prediction data
        prediction_data = {
            "text": text,
            "predicted_label": label,
            "confidence_score": score,
            "timestamp": timestamp,
            "model_version": "financial-bert-onnx:latest",
        }

        # Create filename with timestamp
        filename = f"predictions/{timestamp.replace(':', '-').replace('.', '-')}.json"

        # Create blob and upload data
        blob = bucket.blob(filename)
        blob.upload_from_string(json.dumps(prediction_data, indent=2), content_type="application/json")

        print(f"Prediction saved to gs://{bucket_name}/{filename}")

    except Exception as e:
        print(f"Error saving prediction to Google Cloud Storage: {e}")


@app.get("/")
async def root():
    return {"message": "Welcome to Sentiment Analysis of Financial Text API!"}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, background_tasks: BackgroundTasks):
    request_counter.inc()  # Count total prediction requests
    review_summary.observe(len(request.text))  # Track review size
    tokenizer = app.state.tokenizer
    session = app.state.session
    input_names = app.state.input_names
    output_name = app.state.output_name
    label_map = app.state.label_map

    start_time = time.time()  # Start latency timer
    try:
        tokenizer = app.state.tokenizer
        session = app.state.session
        input_names = app.state.input_names
        output_name = app.state.output_name
        label_map = app.state.label_map

        inputs = tokenizer(request.text, return_tensors="np", truncation=True, padding=True)

        ort_inputs = {name: inputs[name].astype(np.int64) for name in input_names}

        outputs = session.run([output_name], ort_inputs)
        logits = outputs[0][0]
        probs = np.exp(logits) / np.sum(np.exp(logits))  # Softmax
        label_id = int(np.argmax(probs))
        score = float(np.max(probs))
        label = label_map.get(label_id, str(label_id))

        # Add background task to save prediction
        background_tasks.add_task(save_prediction_gcloud, request.text, label, score)

        return PredictResponse(label=label, score=score)

    except Exception as e:
        error_counter.inc()
        raise HTTPException(status_code=500, detail=str(e)) from e

    finally:
        duration = time.time() - start_time
        request_latency.observe(duration)  # Record latency
