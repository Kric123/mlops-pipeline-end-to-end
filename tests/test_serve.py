"""API tests — exercise the FastAPI service with an in-process test client."""

from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

from src.serve import N_FEATURES, app

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"] == "breast-cancer-classifier"
    assert "model_version" in body


def test_example_returns_full_feature_vector():
    r = client.get("/example")
    assert r.status_code == 200
    assert len(r.json()["features"]) == N_FEATURES


def test_predict_flags_malignant():
    sample = client.get("/example").json()["features"]
    r = client.post("/predict", json={"features": sample})
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == "malignant"
    assert body["malignant_probability"] >= body["threshold"]


def test_predict_clears_benign():
    ds = load_breast_cancer(as_frame=True)
    benign = ds.data[ds.target == 1].iloc[0].tolist()  # raw target 1 == benign
    r = client.post("/predict", json={"features": benign})
    assert r.status_code == 200
    assert r.json()["prediction"] == "benign"


def test_predict_rejects_wrong_feature_count():
    r = client.post("/predict", json={"features": [1, 2, 3]})
    assert r.status_code == 422  # pydantic validation error
