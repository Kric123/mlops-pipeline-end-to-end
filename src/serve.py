"""
M3 — Serve the trained model behind a FastAPI REST endpoint.

This turns the registered MLflow model into a live service. It:
  * loads the LATEST version of the registered model from the MLflow registry,
  * exposes /predict which returns the malignant probability AND a decision,
  * applies the 0.40 decision threshold we justified in docs/evaluation.md
    (tuned for recall on malignant — catch more cancers), not a blind 0.5.

Run:
    python src/serve.py
    # or: uvicorn src.serve:app --reload --port 8000

Then:
    GET  http://127.0.0.1:8000/health
    GET  http://127.0.0.1:8000/example     -> a real sample you can paste into /predict
    POST http://127.0.0.1:8000/predict     -> {"features": [...30 floats...]}
    Interactive docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field, field_validator
from sklearn.datasets import load_breast_cancer

# Import the monitoring module whether serve.py is loaded as a package
# (uvicorn src.serve:app, pytest) or run directly (python src/serve.py).
try:
    from src import monitoring
except ImportError:  # pragma: no cover - script-mode fallback
    import monitoring

# ---- Configuration -------------------------------------------------------
REGISTERED_MODEL_NAME = "breast-cancer-classifier"
# Operating threshold chosen in docs/evaluation.md: trade a little precision for
# higher recall on malignant (a missed cancer costs far more than a false alarm).
DECISION_THRESHOLD = 0.40
# Minimum logged predictions before a drift report is statistically meaningful.
# PSI bins inputs into deciles, so small samples leave empty bins and produce
# noisy, inflated scores — require a reasonable sample before reporting.
MIN_DRIFT_SAMPLES = 100

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Two ways to load the model:
#   * MODEL_PATH set  -> load a self-contained model dir baked into the image
#     (this is how the container runs — see Dockerfile / export_model.py).
#   * MODEL_PATH unset -> local dev: pull the latest version from the MLflow
#     registry (SQLite backend that train.py writes to).
MODEL_PATH = os.environ.get("MODEL_PATH")


# ---- Load the model once at startup -------------------------------------
def _load_model():
    """Load the sklearn-flavored model from a baked path or the registry.

    We load the sklearn flavor (not pyfunc) because we need predict_proba to
    return a probability, which the generic pyfunc wrapper hides.
    """
    if MODEL_PATH:
        model = mlflow.sklearn.load_model(MODEL_PATH)
        return model, os.environ.get("MODEL_VERSION", "packaged")

    # Local dev path: point at the SQLite store and grab the highest version.
    mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        raise RuntimeError(
            f"No registered model '{REGISTERED_MODEL_NAME}'. Run src/train.py first."
        )
    latest = max(versions, key=lambda v: int(v.version))
    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{latest.version}")
    return model, latest.version


model, MODEL_VERSION = _load_model()
# In training we relabeled malignant -> 1, so the column for P(malignant) is the
# index of class label 1 in the model's class ordering.
MALIGNANT_IDX = list(model.classes_).index(1)

# Derive the expected feature schema FROM THE MODEL so the API contract always
# matches what the model was trained on (and the input DataFrame's columns line
# up exactly — no "X does not have valid feature names" warnings). Fall back to
# the dataset names if the model didn't record them.
if hasattr(model, "feature_names_in_"):
    FEATURE_NAMES = list(model.feature_names_in_)
else:
    FEATURE_NAMES = list(load_breast_cancer().feature_names)
N_FEATURES = len(FEATURE_NAMES)


# ---- Request / response schemas -----------------------------------------
class PredictRequest(BaseModel):
    features: list[float] = Field(
        ...,
        description=f"Exactly {N_FEATURES} feature values in the canonical dataset order. "
                    f"Call GET /example for a ready-to-use sample.",
    )

    @field_validator("features")
    @classmethod
    def _check_length(cls, v: list[float]) -> list[float]:
        if len(v) != N_FEATURES:
            raise ValueError(f"Expected {N_FEATURES} features, got {len(v)}.")
        return v


class PredictResponse(BaseModel):
    malignant_probability: float
    threshold: float
    prediction: str  # "malignant" | "benign"
    model_version: str


# ---- App ----------------------------------------------------------------
app = FastAPI(
    title="Breast Cancer Classifier API",
    description="Serves the MLflow-registered model. Positive class = malignant.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    """Liveness + which model version is loaded."""
    return {
        "status": "ok",
        "model": REGISTERED_MODEL_NAME,
        "model_version": MODEL_VERSION,
        "decision_threshold": DECISION_THRESHOLD,
    }


@app.get("/example")
def example() -> dict:
    """Return one real malignant sample so you can immediately test /predict."""
    data = load_breast_cancer(as_frame=True)
    # raw target 0 == malignant; grab the first such row
    malignant_row = data.data[data.target == 0].iloc[0]
    return {
        "features": malignant_row.tolist(),
        "note": "This is a known MALIGNANT case — /predict should flag it.",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Score one sample and return probability + thresholded decision."""
    try:
        # Build a single-row DataFrame with the right column names so the model
        # sees the same schema it was trained on.
        row = pd.DataFrame([req.features], columns=FEATURE_NAMES)
        proba = float(model.predict_proba(row)[0, MALIGNANT_IDX])
    except Exception as exc:  # surface model errors as a clean 400
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    label = "malignant" if proba >= DECISION_THRESHOLD else "benign"

    # Log the prediction for monitoring (M5). Never let logging break serving.
    try:
        monitoring.log_prediction(req.features, round(proba, 4), label)
    except Exception:  # pragma: no cover - logging is best-effort
        pass

    return PredictResponse(
        malignant_probability=round(proba, 4),
        threshold=DECISION_THRESHOLD,
        prediction=label,
        model_version=str(MODEL_VERSION),
    )


@app.get("/monitoring/drift")
def drift(window: int | None = None) -> dict:
    """Compare recently logged inputs against the training data (drift check).

    `window` limits the check to the last N logged predictions (default: all).
    """
    current = monitoring.load_recent_predictions(window)
    if len(current) < MIN_DRIFT_SAMPLES:
        return {
            "status": "insufficient_data",
            "n_current": len(current),
            "min_required": MIN_DRIFT_SAMPLES,
            "hint": "Send more predictions, then re-check.",
        }
    reference = monitoring.load_reference()
    report = monitoring.compute_drift_report(reference, current)
    report["n_current"] = len(current)
    report["status"] = "ok"
    return report


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
