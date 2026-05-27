from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import time
import os
import dotenv
import requests

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
HF_API_URL = "https://router.huggingface.co/hf-inference/models/ProsusAI/finbert"
HF_TOKEN   = os.getenv("HUGGING_FACE")

LABEL_MAP = {
    "positive": "Bullish",
    "negative": "Bearish",
    "neutral":  "Neutral"
}


# =========================================================
# ROOT
# =========================================================
@app.get("/")
def root():
    return {
        "message": "FinBERT Sentiment API is running.",
        "docs":    "/docs",
        "health":  "/health",
        "predict": "POST /predict"
    }


# =========================================================
# HEALTH CHECK
# =========================================================
@app.get("/health")
def health_check():
    return {
        "status":       "ok",
        "architecture": "serverless",
        "model":        HF_API_URL,
        "token_set":    HF_TOKEN is not None
    }


# =========================================================
# DEBUG — remove once everything works
# =========================================================
@app.get("/debug")
async def debug_hf():
    """
    Step 1: checks if Render can reach huggingface.co at all
    Step 2: fires a real inference request and returns raw response
    """
    result = {}

    # --- connectivity test ---
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            ping = await client.get("https://huggingface.co")
        result["hf_reachable"]    = ping.status_code == 200
        result["hf_ping_status"]  = ping.status_code
    except Exception as e:
        result["hf_reachable"] = False
        result["ping_error"]   = str(e)
        return result

    if not HF_TOKEN:
        result["token_set"] = False
        result["error"]     = "No HUGGING_FACE token set in environment"
        return result

    result["token_set"] = True

    # --- real inference call ---
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs":  "Apple stock hits all time high after record earnings",
        "options": {"wait_for_model": True}
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(HF_API_URL, headers=headers, json=payload)

        result["status_code"]  = response.status_code
        result["raw_text"]     = response.text
        result["parsed_json"]  = response.json()

    except Exception as e:
        result["inference_error"] = str(e)

    return result


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
        "options": {"wait_for_model": True}
    }

    # --------------------------------------------------
    # 2. Call HuggingFace — async httpx with cold start retry
    # --------------------------------------------------
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(HF_API_URL, headers=headers, json=payload)

            # cold start fallback — wait_for_model handles most cases
            # but keep 503 retry as safety net
            if response.status_code == 503:
                print("HuggingFace still loading — waiting 20s and retrying...")
                time.sleep(20)
                response = await client.post(HF_API_URL, headers=headers, json=payload)

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="HuggingFace API timed out. Try again in a few seconds."
        )

    except httpx.ConnectError:
        raise HTTPException(
            status_code=502,
            detail="Could not reach HuggingFace API. Check your network."
        )

    # --------------------------------------------------
    # 3. Handle HuggingFace HTTP errors
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

        # Guard 1: HF sometimes hides errors inside a 200 response
        if isinstance(raw, dict) and "error" in raw:
            raise ValueError(f"HuggingFace returned an error: {raw['error']}")

        # Guard 2: detect response shape by checking the first element
        #   [[{label, score}]]  → raw[0] is a list → nested, unwrap with raw[0]
        #   [{label, score}]    → raw[0] is a dict → flat,   use raw directly
        if isinstance(raw, list) and len(raw) > 0:
            if isinstance(raw[0], list):
                hf_data = raw[0]        # nested — unwrap
            elif isinstance(raw[0], dict):
                hf_data = raw           # flat — use directly
            else:
                raise ValueError(f"Unexpected element type: {type(raw[0])} | raw: {raw}")
        elif isinstance(raw, dict):
            hf_data = [raw]             # single dict — wrap in list
        else:
            raise ValueError(f"Unexpected response format: {raw}")

        # --------------------------------------------------
        # 5. Get top prediction
        # --------------------------------------------------
        top_pred = max(hf_data, key=lambda x: x.get("score", 0))

        # --------------------------------------------------
        # 6. Build response
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
        raise HTTPException(
            status_code=500,
            detail=f"Parsing error: {str(e)} | Raw HF response: {response.text}"
        )