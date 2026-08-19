# ML Pipeline Project — End-to-End MLOps

An end-to-end machine learning pipeline demonstrating production MLOps practices:
training with experiment tracking, containerized serving, CI/CD, and drift monitoring.

The pipeline is **dataset-agnostic** — it ships with a built-in scikit-learn dataset
(Breast Cancer Wisconsin, binary classification) so it runs with zero external data,
but any tabular dataset can be swapped in.

## Results

| | |
|---|---|
| Model | ROC-AUC **0.994**, registered and versioned in MLflow |
| Serving | containerized FastAPI `/predict`; container output verified identical to local inference |
| CI | install → train → export → test → image build, gated on **14 tests** |
| Drift detection | PSI + KS per feature: **0/30** flagged on stable data, **30/30** under a known shift |
| Sample-size floor | measured, not guessed — see below |

**The drift check once lied to me.** On clean data it reported 28 of 30 features drifted.
Nothing failed; the pipeline was green. Causes: a non-random sample, and too few rows for
decile binning. Rather than patch it and move on, I wrote
[`src/sample_size_study.py`](src/sample_size_study.py), which scores no-drift samples with the
same estimator the endpoint uses, so any alarm it raises comes from sample size alone:

| n | mean PSI | false alarms |
|---|---|---|
| 30 | 0.737 | **68.2%** |
| 50 | 0.247 | 26.9% |
| **100** | 0.096 | **0.8%** |

That is where `MIN_DRIFT_SAMPLES = 100` comes from, and
[`tests/test_sample_size.py`](tests/test_sample_size.py) fails if anyone lowers it.
Known weakness, documented in [`docs/monitoring.md`](docs/monitoring.md): the 1e-6 epsilon floor
makes PSI's magnitude depend on a constant I chose. Deleting it sends PSI to infinity;
collapsing sparse bins is the real fix, and it isn't done yet.

## Status (milestone tracker)

- [x] **M1 — Train + experiment tracking** (`src/train.py`, MLflow + full eval suite)
- [x] **M3 — Serve** (`src/serve.py`, FastAPI `/predict` endpoint)
- [x] **M2 — Containerize** (`Dockerfile`, baked model artifact)
- [x] **M4 — CI/CD** (`.github/workflows/ci.yml`, `pytest` suite)
- [x] **M5 — Monitoring** (`src/monitoring.py`, drift detection + prediction logging)
- [x] **M6 — Cloud deploy** (runbook in `docs/deployment.md`, SageMaker BYOC)

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
    export_model.py # M2 — export the registered model to a portable artifact
    monitoring.py   # M5 — prediction logging + PSI/KS drift detection
    check_drift.py  # M5 — CLI drift check (exits non-zero on drift, for cron/CI)
  tests/            # M4 — pytest suite (API, metrics, drift)
  docs/
    evaluation.md   # what each metric means & which to optimize (read this)
    monitoring.md   # how drift detection works (PSI, KS, pitfalls)
    deployment.md   # SageMaker deploy runbook (M6)
  Dockerfile        # M2 — containerized serving
  .github/workflows/ci.yml   # M4 — CI pipeline
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
| `GET /monitoring/drift` | data-drift report on recent inputs (M5) |
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

## Tests & CI (M4)

```bash
pip install -r requirements-dev.txt
pytest                     # 7 tests: API behavior + metric logic
```

`.github/workflows/ci.yml` runs on every push / PR to `main`:
**install deps → train → export model → `pytest` → `docker build`**. This guarantees the
model trains, the API passes its tests, and the image still builds before anything merges.

## Monitoring (M5)

Every prediction is logged, and inputs are checked for **data drift** against the
training distribution using **PSI** and the **KS test**. Demonstrated end-to-end:
normal traffic reads as `0/30` features drifted; inputs scaled ×1.4 read as `30/30`.

```bash
# live, on the running API
curl "http://127.0.0.1:8000/monitoring/drift?window=200"

# scheduled / CI check — exits non-zero on drift
python src/check_drift.py --window 500
```

How it works, the two metrics, and a real small-sample pitfall are documented in
[`docs/monitoring.md`](docs/monitoring.md).

## Cloud deploy (M6)

The container deploys to **AWS SageMaker** (bring-your-own-container). The full
runbook — ECR push, model/endpoint creation, the `/ping` + `/invocations` contract,
invocation, teardown, and cheaper serverless/Vertex AI paths — is in
[`docs/deployment.md`](docs/deployment.md).

## Evaluation

`train.py` treats **malignant as the positive class** and logs a full, interpretable
evaluation suite to MLflow: headline metrics, a confusion matrix, a per-class report,
ROC + precision-recall curves, and a decision-threshold sweep. The reasoning behind every
metric — and why recall on malignant is the number that matters here — is documented in
[`docs/evaluation.md`](docs/evaluation.md).

## Stack

Python · scikit-learn · MLflow · (Docker, FastAPI, GitHub Actions to come)
