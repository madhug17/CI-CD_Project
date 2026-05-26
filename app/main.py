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
    # 3. Handle explicit HuggingFace errors
    # --------------------------------------------------
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"HuggingFace API Error: {response.text}"
        )

    # --------------------------------------------------
    # 4. Parse response safely (The Bulletproof Block)
    # --------------------------------------------------
    try:
        raw = response.json()

        # Guard 1: Did HF send a hidden error inside a dictionary?
        if isinstance(raw, dict) and "error" in raw:
            raise ValueError(f"HuggingFace threw an error: {raw['error']}")

        # Guard 2: Is it a nested list [[{...}]] or a flat list [{...}]?
        if isinstance(raw, list) and len(raw) > 0:
            if isinstance(raw, list):
                hf_data = raw  # It's nested, grab the inner list
            elif isinstance(raw, dict):
                hf_data = raw     # It's flat, use it directly
            else:
                raise ValueError(f"Unknown list format: {raw}")
        else:
            raise ValueError(f"Empty or weird response: {raw}")

        # Now it is safe to grab the max score
        top_pred = max(hf_data, key=lambda x: x.get("score", 0))

        # --------------------------------------------------
        # 5. Build clean response
        # --------------------------------------------------
        probs = {
            LABEL_MAP.get(item.get("label"), item.get("label")): round(item.get("score", 0), 4)
            for item in hf_data
        }

        return {
            "headline":      news.text,
            "prediction":    LABEL_MAP.get(top_pred.get("label"), top_pred.get("label")),
            "confidence":    round(top_pred.get("score", 0), 4),
            "probabilities": probs
        }

    except Exception as e:
        # If ANYTHING fails, don't crash silently. 
        # Send the exact data straight to Streamlit so we can read it.
        raise HTTPException(
            status_code=500,
            detail=f"Parsing Bug: {str(e)} | Raw HF Data: {response.text}"
        )