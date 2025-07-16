from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import wandb
import torch

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    score: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Download model artifact from wandb
    run = wandb.init(project="financial-sentiment-bert", job_type="inference")
    artifact = run.use_artifact('model-run_financial_bert:latest', type='model')
    artifact_dir = artifact.download()
    # Load model and tokenizer using HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
    model = AutoModelForSequenceClassification.from_pretrained(artifact_dir)

    app.state.tokenizer = tokenizer
    app.state.model = model

    yield

    run.finish()
    # Clean up the model
    del model
    del tokenizer
    print("Model cleaned up! Goodbye!")


app = FastAPI(
    title="Financial Sentiment Analysis API",
    description="API for predicting sentiment of financial text",
    version="1.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    return {"message": "Welcome to Sentiment Analysis of Financial Text API!"}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    tokenizer = app.state.tokenizer
    model = app.state.model
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        score, label_id = torch.max(probs, dim=1)
        label = model.config.id2label[label_id.item()] if hasattr(model.config, "id2label") else str(label_id.item())
    return PredictResponse(label=label, score=score.item())


