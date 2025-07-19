from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_read_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to Sentiment Analysis of Financial Text API!"}


def test_predict():
    with TestClient(app) as client:
        data = {"text": "The company's profits increased this quarter."}
        response = client.post("/predict", json=data)
        assert response.status_code == 200
        result = response.json()
        assert "label" in result
        assert "score" in result
        assert isinstance(result["label"], str)
        assert isinstance(result["score"], float)
