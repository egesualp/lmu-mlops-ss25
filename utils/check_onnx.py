import time
import numpy as np
import torch
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# === CONFIG ===
TEXT = "The market outlook is stable and promising."
HF_MODEL_PATH = "artifacts/model-run_financial_bert:v0"  # Local path or W&B-downloaded dir
ONNX_MODEL_PATH = "models/onnx/model.onnx"  # Path to exported ONNX model
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}  # Adjust based on training

# === Load Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH)
inputs_pt = tokenizer(TEXT, return_tensors="pt", truncation=True, padding=True)
inputs_np = {k: v.numpy().astype(np.int64) for k, v in inputs_pt.items()}

# === Inference with HuggingFace ===
hf_model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_PATH)
hf_model.eval()

start = time.time()
with torch.no_grad():
    logits_hf = hf_model(**inputs_pt).logits.numpy()
hf_time = time.time() - start

# === Inference with ONNX ===
session = ort.InferenceSession(ONNX_MODEL_PATH)
onnx_inputs = {
    "input_ids": inputs_np["input_ids"],
    "attention_mask": inputs_np["attention_mask"],
}

start = time.time()
logits_onnx = session.run(["logits"], onnx_inputs)[0]
onnx_time = time.time() - start


# === Compare Predictions ===
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


probs_hf = softmax(logits_hf[0])
probs_onnx = softmax(logits_onnx[0])

label_id_hf = int(np.argmax(probs_hf))
label_id_onnx = int(np.argmax(probs_onnx))

print("\n=== PREDICTIONS ===")
print(f"Input Text: {TEXT}")
print(f"HuggingFace: {LABEL_MAP[label_id_hf]} ({probs_hf[label_id_hf]:.4f})")
print(f"ONNX:        {LABEL_MAP[label_id_onnx]} ({probs_onnx[label_id_onnx]:.4f})")

print("\n=== TIMING ===")
print(f"HuggingFace inference time: {hf_time * 1000:.2f} ms")
print(f"ONNX Runtime inference time: {onnx_time * 1000:.2f} ms")

print("\n=== LOGIT DIFFERENCE ===")
print("L2 norm of logits diff:", np.linalg.norm(logits_hf - logits_onnx))
