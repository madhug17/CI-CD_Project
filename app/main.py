from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
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
# SERVERLESS CONFIGURATION
# =========================================================
# The URL to the Hugging Face supercomputer running FinBERT
HF_API_URL = "https://api-inference.huggingface.co/models/yiyanghkust/finbert-tone"
# Your secret VIP pass (injected from GitHub/Render environments)
HF_TOKEN = os.getenv("HUGGING_FACE")

# =========================================================
# HEALTH CHECK
# =========================================================
@app.get("/health")
def health_check():
    """
    Used by CI/CD and Render to verify the server is alive.
    """
    return {"status": "ok", "architecture": "serverless"}

# =========================================================
# PREDICT (Forwarded to Hugging Face)
# =========================================================
@app.post("/predict")
async def predict_sentiment(news: NewsInput):
    # 1. Check if we have the VIP pass
    if not HF_TOKEN:
        raise HTTPException(
            status_code=500, 
            detail="HF_TOKEN environment variable is missing. Check your Render/GitHub secrets."
        )

    # 2. Package the text for Hugging Face
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": news.text}

    # 3. Ping the supercomputer
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    
    # 4. Handle any errors from Hugging Face
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code, 
            detail=f"Hugging Face API Error: {response.text}"
        )

    # 5. Parse the results
    # Hugging Face returns data in this format: [[{'label': 'Neutral', 'score': 0.8}, ...]]
    hf_data = response.json() 
    
    # Find the label with the highest score
    top_pred = max(hf_data, key=lambda x: x['score'])
    
    # Format probabilities cleanly for the API response
    probs = {item['label']: round(item['score'], 4) for item in hf_data}

    # 6. Return the exact same JSON schema your old PyTorch code used!
    return {
        "headline": news.text,
        "prediction": top_pred['label'],
        "confidence": round(top_pred['score'], 4),
        "probabilities": probs
    }