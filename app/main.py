from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import time
import os
import dotenv

dotenv.load_dotenv()


# =========================================================
# APP
# =========================================================
app = FastAPI(
    title="FinBERT Sentiment API (Serverless)",
    description="Lightweight API Gateway pointing to Hugging Face Inference API",
    version="2.0.0"
)


# =========================================================
# REQUEST SCHEMA
# =========================================================
class NewsInput(BaseModel):
    text: str


# =========================================================
# CONFIG
# =========================================================
HF_API_URL = "https://api-inference.huggingface.co/models/yiyanghkust/finbert-tone"
HF_TOKEN   = os.getenv("HUGGING_FACE")

# finbert-tone returns "Positive" / "Negative" / "Neutral"
# we map them to financial terms for the response
LABEL_MAP = {
    "Positive": "Bullish",
    "Negative": "Bearish",
    "Neutral":  "Neutral"
}


# =========================================================
# HEALTH CHECK
# =========================================================
@app.get("/health")
def health_check():
    """
    Used by CI/CD pipeline and Render to verify server is alive.
    Does NOT call HuggingFace — just confirms the API is running.
    """
    return {
        "status":       "ok",
        "architecture": "serverless",
        "model":        HF_API_URL,
        "token_set":    HF_TOKEN is not None   # confirms env var is loaded
    }


# =========================================================
# PREDICT
# =========================================================
@app.post("/predict")
async def predict_sentiment(news: NewsInput):

    # --------------------------------------------------
    # 1. Guard — no token, no call
    # --------------------------------------------------
    if not HF_TOKEN:
        raise HTTPException(
            status_code=500,
            detail=(
                "HUGGING_FACE token missing. "
                "Add it to Render → Environment Variables."
            )
        )

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": news.text,
        "options": {"wait_for_model": True} 
    }

    # --------------------------------------------------
    # 2. Call HuggingFace — with cold start retry
    # Free tier cold starts the model if unused for a while.
    # First request returns 503 "model is loading" → wait 20s → retry once.
    # --------------------------------------------------
    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json=payload,
            timeout=30        # don't hang forever
        )

        if response.status_code == 503:
            print("HuggingFace model is cold starting — waiting 20s and retrying...")
            time.sleep(20)
            response = requests.post(
                HF_API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )

    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="HuggingFace API timed out. Try again in a few seconds."
        )

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=502,
            detail="Could not reach HuggingFace API. Check your network."
        )

    # --------------------------------------------------
    # 3. Handle HuggingFace errors
    # --------------------------------------------------
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"HuggingFace API Error: {response.text}"
        )

    # --------------------------------------------------
    # 4. Parse response
    # HuggingFace returns a NESTED list:
    #   [[{'label': 'Positive', 'score': 0.94}, ...]]
    #    ^^outer list  ^^inner list = actual predictions
    # Must index [0] to get the inner list before calling max()
    # --------------------------------------------------
    raw     = response.json()
    hf_data = raw[0]           # inner list — list of {label, score} dicts

    top_pred = max(hf_data, key=lambda x: x["score"])

    # --------------------------------------------------
    # 5. Build clean response
    # --------------------------------------------------
    probs = {
        LABEL_MAP.get(item["label"], item["label"]): round(item["score"], 4)
        for item in hf_data
    }

    return {
        "headline":      news.text,
        "prediction":    LABEL_MAP.get(top_pred["label"], top_pred["label"]),
        "confidence":    round(top_pred["score"], 4),
        "probabilities": probs
    }