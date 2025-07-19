import sys
import os

# Add the root directory to Python path
# if not, the testing in github actions fails
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from serve.app import app

client = TestClient(app)

def test_prediction_endpoint():
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert response.status_code == 200
    assert "prediction" in response.json()
