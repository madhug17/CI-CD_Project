import sys 
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import pytest 
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
def test_predict_endpoint():
    responce = client.post("/predict",json={"text":"nvidia is great!"})
    assert responce.status_code == 200
    assert "prediction" in responce.json()
