import os
import secrets
from typing import List

import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = os.getenv("MODEL_ID", "model")  # local folder, or "username/sentiment-binary" on the Hub
API_KEY = os.getenv("SENTIMENT_API_KEY", "")

# Both values confirmed from train_binary.py:
#   - tokenizer(..., truncation=True, max_length=128)
#   - LabelEncoder sorts classes alphabetically: Negative -> 0, Positive -> 1
MAX_LENGTH = 128
LABELS = {0: "Negative", 1: "Positive"}

BATCH_SIZE = 32
MAX_TEXTS = 500  # safety cap per request

# Training used no text cleaning (no lowercasing, no regex), so none is applied here.
HF_TOKEN = os.getenv("HF_TOKEN", "")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN or None,
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN or None,
)
model = model.float()
model.eval()

app = FastAPI()


class Batch(BaseModel):
    texts: List[str]


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict_batch")
def predict_batch(batch: Batch, x_api_key: str = Header(default="")):
    if API_KEY and not secrets.compare_digest(x_api_key.encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Invalid API key")

    texts = batch.texts
    if len(texts) > MAX_TEXTS:
        raise HTTPException(status_code=413, detail=f"Max {MAX_TEXTS} texts per request")

    results = [None] * len(texts)

    # Sort by length so each batch has similar-length texts (less padding, faster on CPU).
    # Results are written back to their original positions.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))

    for start in range(0, len(order), BATCH_SIZE):
        idxs = order[start:start + BATCH_SIZE]
        chunk = [texts[i] for i in idxs]
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        with torch.inference_mode():
            probs = torch.softmax(model(**inputs).logits, dim=-1)
        for i, p in zip(idxs, probs):
            idx = int(p.argmax())
            results[i] = {"label": LABELS[idx], "confidence": float(p[idx])}

    return {"results": results}
