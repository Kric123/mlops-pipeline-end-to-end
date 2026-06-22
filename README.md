# ML Pipeline Project — End-to-End MLOps

An end-to-end machine learning pipeline demonstrating production MLOps practices:
training with experiment tracking, containerized serving, CI/CD, and drift monitoring.

The pipeline is **dataset-agnostic** — it ships with a built-in scikit-learn dataset
(Breast Cancer Wisconsin, binary classification) so it runs with zero external data,
but any tabular dataset can be swapped in.

## Status (milestone tracker)

- [x] **M1 — Train + experiment tracking** (`src/train.py`, MLflow + full eval suite)
- [x] **M3 — Serve** (`src/serve.py`, FastAPI `/predict` endpoint)
- [x] **M2 — Containerize** (`Dockerfile`, baked model artifact)
- [ ] M4 — CI/CD (GitHub Actions)
- [ ] M5 — Drift detection + monitoring
- [ ] M6 — Deploy to cloud (SageMaker / Vertex AI)

## Quickstart

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
python src/train.py
```

Then view the tracked experiments:

```bash
mlflow ui
# open http://127.0.0.1:5000
```

## Project layout

```
ml-pipeline-project/
  src/
    train.py        # M1 — train + full evaluation, logged to MLflow
    serve.py        # M3 — FastAPI service: loads registered model, /predict
  docs/
    evaluation.md   # what each metric means & which to optimize (read this)
  requirements.txt
  README.md
  .gitignore
```

## Serving (M3)

```bash
python src/serve.py        # starts FastAPI on http://127.0.0.1:8000
```

| Endpoint | What it does |
|---|---|
| `GET /health` | liveness + which model version is loaded |
| `GET /example` | returns a real malignant sample to paste into `/predict` |
| `POST /predict` | body `{"features": [..30 floats..]}` → `{malignant_probability, prediction}` |
| `GET /docs` | interactive Swagger UI |

The endpoint applies the **0.40 decision threshold** chosen in `docs/evaluation.md`
(tuned for recall on malignant), so the API reflects the cost-of-error reasoning rather
than a default 0.5 cutoff.

## Container (M2)

The image bakes in an **immutable, exported model artifact** (`model_artifact/`) rather
than the MLflow tracking DB — so it's fully portable and doesn't depend on host paths.

```bash
# 1. export the latest registered model to ./model_artifact (once, after training)
python src/export_model.py

# 2. build the image
docker build -t breast-cancer-api:latest .

# 3. run it
docker run -d --name bc-api -p 8000:8000 breast-cancer-api:latest
curl http://127.0.0.1:8000/health
```

`serve.py` loads the baked model when `MODEL_PATH` is set (the container case) and falls
back to the MLflow registry for local dev. *Future optimization: the image is ~1.3GB
because MLflow pulls heavy deps; loading the model with `skops`/`joblib` directly would
slim it substantially.*

## Evaluation

`train.py` treats **malignant as the positive class** and logs a full, interpretable
evaluation suite to MLflow: headline metrics, a confusion matrix, a per-class report,
ROC + precision-recall curves, and a decision-threshold sweep. The reasoning behind every
metric — and why recall on malignant is the number that matters here — is documented in
[`docs/evaluation.md`](docs/evaluation.md).

## Stack

Python · scikit-learn · MLflow · (Docker, FastAPI, GitHub Actions to come)
