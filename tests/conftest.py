"""Shared test setup.

serve.py loads its model at import time. We point it at the exported, baked model
artifact (MODEL_PATH) so tests don't depend on the MLflow registry/DB being present
— the same way the container loads it. CI runs `export_model.py` before pytest so
this directory exists.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# setdefault: respect an externally-provided MODEL_PATH if one is already set.
os.environ.setdefault("MODEL_PATH", str(PROJECT_ROOT / "model_artifact"))
os.environ.setdefault("MODEL_VERSION", "test")
