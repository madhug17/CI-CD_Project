import sys 
import os 
from unittest.mock import patch
import dotenv

dotenv.load_dotenv()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# We can safely use a fake token again because the internet request is intercepted!
os.environ["HUGGING_FACE"] = "fake_test_token"

import pytest 
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# 1. The @patch decorator intercepts 'requests.post' inside your main.py
@patch("app.main.requests.post")
def test_predict_endpoint(mock_post):
    
    # 2. Build the "Wind Tunnel" (A fake Hugging Face response)
    class FakeResponse:
        status_code = 200
        text = "OK"
        def json(self):
            # This perfectly mimics Hugging Face's math format
            return [[
                {"label": "Bullish", "score": 0.95},
                {"label": "Neutral", "score": 0.04},
                {"label": "Bearish", "score": 0.01}
            ]]
    
    # 3. Force the intercepted request to return our fake data
    mock_post.return_value = FakeResponse()

    # 4. Run the test! (It will run in 0.001 seconds)
    response = client.post("/predict", json={"text": "Nvidia is great!"})
    
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] == "Bullish"