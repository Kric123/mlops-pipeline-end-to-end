"""
Export the latest registered model to a self-contained directory for packaging.

The MLflow tracking DB records artifact paths as absolute *host* paths, which
don't exist inside a container. So before building the Docker image we export an
immutable, portable copy of the model that the image can load via MODEL_PATH.

Run:
    python src/export_model.py
    # produces ./model_artifact/  (and writes the version it exported)
"""

from __future__ import annotations

import shutil
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

REGISTERED_MODEL_NAME = "breast-cancer-classifier"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "model_artifact"


def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        raise RuntimeError(
            f"No registered model '{REGISTERED_MODEL_NAME}'. Run src/train.py first."
        )
    latest = max(versions, key=lambda v: int(v.version))

    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{latest.version}")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    mlflow.sklearn.save_model(sk_model=model, path=OUT_DIR.as_posix())

    (PROJECT_ROOT / "model_artifact" / "EXPORTED_VERSION.txt").write_text(
        str(latest.version), encoding="utf-8"
    )
    print(f"Exported '{REGISTERED_MODEL_NAME}' v{latest.version} -> {OUT_DIR}")


if __name__ == "__main__":
    main()
