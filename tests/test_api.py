import sys 
import os 
from unittest.mock import patch
import dotenv

# 1. The Path Hack (Keeps GitHub Actions from getting lost)
dotenv.load_dotenv()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# 2. Force the token (Matches the exact variable name in main.py)
os.environ["HF_TOKEN"] = "fake_test_token"

import pytest 
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# 3. Intercept the internet call
@patch("app.main.requests.post")
def test_predict_endpoint(mock_post):
    
    # 4. Build the fake Hugging Face response
    class FakeResponse:
        status_code = 200
        text = "OK"
        def json(self):
            # Matches the exact data shape our new bulletproof parser expects
            return [[
                {"label": "Bullish", "score": 0.95},
                {"label": "Neutral", "score": 0.04},
                {"label": "Bearish", "score": 0.01}
            ]]
    
    # 5. Load the fake response into the interceptor
    mock_post.return_value = FakeResponse()

    # 6. Fire the test payload
    response = client.post("/predict", json={"text": "Nvidia is great!"})
    
    # 7. Assertions
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] == "Neutral"