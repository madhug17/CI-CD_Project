from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# =========================================================
# APP
# =========================================================
app = FastAPI(
    title="FinBERT Sentiment API",
    description="Financial news sentiment — Bullish / Bearish / Neutral",
    version="1.0.0"
)

# =========================================================
# REQUEST SCHEMA
# =========================================================
class NewsInput(BaseModel):
    text: str

# =========================================================
# LABEL MAPPING
# ProsusAI/finbert label order: 0=positive, 1=negative, 2=neutral
# We rename to financial terms for the response.
# =========================================================
LABELS = {0: "Bearish", 1: "Neutral", 2: "Bullish"}
# =========================================================
# LOAD MODEL ON STARTUP
# =========================================================
MODEL_PATH = "./app/saved_model"

try:
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()   # set to inference mode — disables dropout
    print("Model loaded successfully! Ready for inference.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model. Did you run train.py first?\nError: {e}")
    tokenizer = None
    model = None

LABELS = model.config.id2label
# =========================================================
# HEALTH CHECK
# =========================================================
@app.get("/health")
def health_check():
    """
    Used by Docker HEALTHCHECK and CI/CD pipeline to verify
    the server is up and the model is loaded.
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {"status": "ok", "model_path": MODEL_PATH}


# =========================================================
# PREDICT
# =========================================================
@app.post("/predict")
async def predict_sentiment(news: NewsInput):

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=500,
            detail="Model not initialized. Run train.py first."
        )

    # --------------------------------------------------
    # 1. Tokenize
    # --------------------------------------------------
    inputs = tokenizer(
        news.text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    # --------------------------------------------------
    # 2. Forward pass
    # --------------------------------------------------
    with torch.no_grad():
        outputs = model(**inputs)

    # --------------------------------------------------
    # 3. Softmax → probabilities
    # outputs.logits shape: [batch=1, num_classes=3]
    # probs shape:          [batch=1, num_classes=3]
    # --------------------------------------------------
    probs = F.softmax(outputs.logits, dim=-1)

    # --------------------------------------------------
    # 4. Argmax → predicted class
    # predicted_class shape: [batch=1]
    # --------------------------------------------------
    confidence, predicted_class = torch.max(probs, dim=1)

    # --------------------------------------------------
    # 5. Build response
    # probs[0] = first (only) item in the batch → shape [3]
    # probs[0][0] = Bullish prob (scalar) → .item() works
    # probs[0][1] = Bearish prob
    # probs[0][2] = Neutral prob
    # --------------------------------------------------
    return {
        "headline":        news.text,
        "prediction":      LABELS[predicted_class.item()],
        "confidence":      round(confidence.item(), 4),
        "probabilities": {
            "Bullish": round(probs[0][2].item(), 4),
            "Bearish": round(probs[0][0].item(), 4),
            "Neutral": round(probs[0][1].item(), 4),
        }
    }