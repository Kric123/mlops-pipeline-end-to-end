# M2 — Containerize the serving app.
# Packages serve.py + a baked, immutable model artifact so the API runs
# identically on any machine. Build context excludes the MLflow tracking store
# (see .dockerignore) — the model is loaded from MODEL_PATH, not the registry.

FROM python:3.14-slim

# - PYTHONUNBUFFERED: logs stream out immediately (don't sit in a buffer)
# - PYTHONDONTWRITEBYTECODE: no .pyc clutter in the image
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first (own layer) so code changes don't bust the pip cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code + the exported model artifact.
COPY src/ ./src/
COPY model_artifact/ ./model_artifact/

# Tell serve.py to load the baked model instead of the MLflow registry.
ENV MODEL_PATH=/app/model_artifact \
    MODEL_VERSION=2

EXPOSE 8000

# 0.0.0.0 so the port is reachable from outside the container.
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
