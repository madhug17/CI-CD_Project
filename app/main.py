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
        "token_set":    HF_TOKEN is not None
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
        "inputs":  news.text,
        "options": {"wait_for_model": True}  # handles cold start on HF side
    }

    # --------------------------------------------------
    # 2. Call HuggingFace — with cold start retry as backup
    # wait_for_model handles most cold starts automatically,
    # but we keep the 503 retry as a safety net
    # --------------------------------------------------
    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json=payload,
            timeout=60        # longer timeout since wait_for_model can take ~30s
        )

        if response.status_code == 503:
            print("HuggingFace still loading — waiting 20s and retrying...")
            time.sleep(20)
            response = requests.post(
                HF_API_URL,
                headers=headers,
                json=payload,
                timeout=60
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
    # 3. Handle explicit HuggingFace HTTP errors
    # --------------------------------------------------
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"HuggingFace API Error: {response.text}"
        )

    # --------------------------------------------------
    # 4. Parse response safely
    # --------------------------------------------------
    try:
        raw = response.json()

        # Guard 1: HF sometimes returns {"error": "..."} with a 200 status
        if isinstance(raw, dict) and "error" in raw:
            raise ValueError(f"HuggingFace returned an error: {raw['error']}")

        # Guard 2: unwrap the response correctly
        # Check raw[0] — the FIRST ELEMENT — to know the structure:
        #   raw = [[{label, score}, ...]]  → raw[0] is a list  → nested, use raw[0]
        #   raw = [{label, score}, ...]    → raw[0] is a dict  → flat, use raw directly
        if isinstance(raw, list) and len(raw) > 0:
            if isinstance(raw[0], list):
                hf_data = raw[0]    # nested [[{...}]] — unwrap outer list
            elif isinstance(raw[0], dict):
                hf_data = raw       # flat [{...}] — already correct
            else:
                raise ValueError(f"Unknown list element format: {type(raw[0])} | raw: {raw}")
        elif isinstance(raw, dict):
            hf_data = [raw]         # single dict response — wrap in list
        else:
            raise ValueError(f"Empty or unexpected response format: {raw}")

        # --------------------------------------------------
        # 5. Get top prediction
        # --------------------------------------------------
        top_pred = max(hf_data, key=lambda x: x.get("score", 0))

        # --------------------------------------------------
        # 6. Build clean response
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
        # Surface the raw HF response so you can debug exactly what came back
        raise HTTPException(
            status_code=500,
            detail=f"Parsing error: {str(e)} | Raw HF response: {response.text}"
        )