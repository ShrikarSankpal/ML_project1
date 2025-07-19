from fastapi.testclient import TestClient
from serve.app import app

client = TestClient(app)

def test_predict_valid_input():
    response = client.post("/predict", json={"features": [8.3252, 41.0, 6.984127, 1.02381, 322.0, 2.555556, 37.88, -122.23]})
    assert response.status_code == 200
    json_data = response.json()
    assert "prediction" in json_data
    assert isinstance(json_data["prediction"], float)
