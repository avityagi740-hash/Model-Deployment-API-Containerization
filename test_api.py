from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
client.__enter__()  # trigger lifespan startup for module-level client


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "endpoints" in resp.json()


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_setosa():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_class"] == "setosa"
    assert body["predicted_class_index"] == 0
    assert "probabilities" in body


def test_predict_virginica():
    payload = {
        "sepal_length": 6.7,
        "sepal_width": 3.0,
        "petal_length": 5.2,
        "petal_width": 2.3,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    assert resp.json()["predicted_class"] == "virginica"


def test_predict_batch():
    payload = {
        "instances": [
            {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
            {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3},
        ]
    }
    resp = client.post("/predict/batch", json=payload)
    assert resp.status_code == 200
    preds = resp.json()["predictions"]
    assert len(preds) == 2


def test_predict_batch_empty():
    resp = client.post("/predict/batch", json={"instances": []})
    assert resp.status_code == 400


def test_predict_invalid_payload():
    resp = client.post("/predict", json={"sepal_length": "not_a_number"})
    assert resp.status_code == 422
