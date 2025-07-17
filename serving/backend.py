import os
import json
import numpy as np
import onnxruntime as ort
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
import anyio
import transformers

# Suppress HuggingFace warnings
transformers.logging.set_verbosity_error()

MODEL_DIR = "onnx"
MODEL_FILE = os.path.join(MODEL_DIR, "model.onnx")
LABEL_FILE = os.path.join(MODEL_DIR, "label_map.json")

class TextRequest(BaseModel):
    text: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ONNX model and tokenizer at startup."""
    global tokenizer, session, input_names, output_name, label_map

    # Check model file exists
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"❌ ONNX model not found at {MODEL_FILE}")

    # Load ONNX model
    session = ort.InferenceSession(MODEL_FILE)
    input_names = [inp.name for inp in session.get_inputs()]
    output_name = session.get_outputs()[0].name

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    # Load label map or fallback
    if os.path.exists(LABEL_FILE):
        async with await anyio.open_file(LABEL_FILE, "r") as f:
            label_map = json.loads(await f.read())
    else:
        label_map = {"0": "negative", "1": "neutral", "2": "positive"}

    yield

    # Cleanup
    del tokenizer, session, input_names, output_name, label_map


app = FastAPI(
    title="Financial Sentiment Classifier",
    description="ONNX + FastAPI backend for sentiment classification",
    version="1.0",
    lifespan=lifespan
)


def predict_text(text: str) -> tuple[np.ndarray, str]:
    """Run text through ONNX model and return probabilities and label."""
    inputs = tokenizer(text, return_tensors="np", truncation=True, padding=True)
    ort_inputs = {name: inputs[name].astype(np.int64) for name in input_names}
    logits = session.run([output_name], ort_inputs)[0][0]

    # Numerically stable softmax
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()

    label_id = int(np.argmax(probs))
    label = label_map.get(str(label_id), str(label_id))
    return probs, label


@app.get("/")
async def root():
    return {"message": "Hello from the ONNX backend!"}


@app.post("/predict/")
async def classify_text(request: TextRequest):
    try:
        probs, label = predict_text(request.text)
        return {
            "text": request.text,
            "prediction": label,
            "probabilities": probs.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
