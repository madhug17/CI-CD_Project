# tests/test_api.py

import sys
import os
from unittest.mock import patch, AsyncMock, MagicMock
import dotenv

dotenv.load_dotenv()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

os.environ["HF_TOKEN"] = "fake_test_token"

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@patch("httpx.AsyncClient.post", new_callable=AsyncMock)
def test_predict_endpoint(mock_post):

    # Build a fake httpx response
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.text = "OK"
    fake_response.json.return_value = [[
        {"label": "Bullish", "score": 0.95},
        {"label": "Neutral", "score": 0.04},
        {"label": "Bearish", "score": 0.01}
    ]]
    mock_post.return_value = fake_response

    response = client.post("/predict", json={"text": "Nvidia is great!"})

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] == "Bullish"  # 0.95 is the top score