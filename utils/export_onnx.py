import os
import torch
import wandb
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# === Configuration ===
PROJECT_NAME = "financial-sentiment-bert"
ENTITY = "cbrkcan90-ludwig-maximilianuniversity-of-munich"
MODEL_ARTIFACT = "cbrkcan90-ludwig-maximilianuniversity-of-munich/financial-sentiment-bert/model-run_financial_bert:v0"
ONNX_ARTIFACT_NAME = "financial-bert-onnx"
ONNX_DIR = "models/onnx"
ONNX_PATH = os.path.join(ONNX_DIR, "model.onnx")

# === Step 1: Download model from W&B ===
run = wandb.init(
    project=PROJECT_NAME,
    entity=ENTITY,
    job_type="download"
)

artifact = run.use_artifact(MODEL_ARTIFACT, type="model")
artifact_dir = artifact.download()
print(f"Artifact downloaded to: {artifact_dir}")

# === Step 2: Load model and tokenizer ===
model = AutoModelForSequenceClassification.from_pretrained(artifact_dir)
tokenizer = AutoTokenizer.from_pretrained(artifact_dir)

# === Step 3: Export to ONNX ===
os.makedirs(ONNX_DIR, exist_ok=True)
dummy_text = "This is a dummy input for ONNX export."
inputs = tokenizer(dummy_text, return_tensors="pt")

torch.onnx.export(
    model,
    (inputs["input_ids"], inputs["attention_mask"]),
    ONNX_PATH,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"}
    },
    opset_version=17
)

# Save tokenizer along with ONNX model
tokenizer.save_pretrained(ONNX_DIR)

print(f"Model exported to: {ONNX_PATH}")

# === Step 4: Upload ONNX model to W&B ===
run = wandb.init(
    project=PROJECT_NAME,
    entity=ENTITY,
    job_type="export-onnx"
)

onnx_artifact = wandb.Artifact(ONNX_ARTIFACT_NAME, type="model")
onnx_artifact.add_dir(ONNX_DIR)
run.log_artifact(onnx_artifact)
run.finish()

print(f"ONNX model uploaded to W&B as: {ONNX_ARTIFACT_NAME}")
