import sys 
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Fix 3: Set the environment variable BEFORE importing the app!
# Fix 4: Add 'ust' to the model name!
os.environ["MODEL_PATH"] = "yiyanghkust/finbert-tone"

import pytest 
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_endpoint():
    response = client.post("/predict",json={"text":"nvidia is great!"})
    assert response.status_code == 200
    assert "prediction" in response.json()