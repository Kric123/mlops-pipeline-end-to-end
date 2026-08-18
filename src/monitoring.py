"""
M5 — Monitoring: prediction logging + data-drift detection.

Two production concerns live here:

1. **Prediction logging** — every served prediction is appended to a JSONL log,
   so you have a record of what the model actually saw and said in production.

2. **Data drift detection** — comparing the distribution of recent inputs against
   the training data. A model is only valid while production data looks like the
   data it was trained on; when inputs drift, accuracy silently rots. We quantify
   drift two complementary ways:
     * **PSI (Population Stability Index)** — industry-standard single number per
       feature. Rule of thumb: <0.1 stable, 0.1-0.25 moderate shift, >0.25 drift.
     * **KS test (Kolmogorov-Smirnov)** — a hypothesis test; a low p-value means
       the two samples are unlikely to come from the same distribution.

We flag a feature as drifted if EITHER signal fires.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Drift thresholds (the well-known rules of thumb).
PSI_THRESHOLD = 0.25
KS_P_THRESHOLD = 0.05


# ---- Paths (container-aware) --------------------------------------------
def reference_path() -> Path:
    """Training-data reference. Baked next to the model in the container."""
    model_path = os.environ.get("MODEL_PATH")
    if model_path:
        return Path(model_path) / "reference_data.csv"
    return PROJECT_ROOT / "monitoring" / "reference_data.csv"


def predictions_log_path() -> Path:
    """Where served predictions are appended (override with PREDICTIONS_LOG)."""
    env = os.environ.get("PREDICTIONS_LOG")
    if env:
        return Path(env)
    return PROJECT_ROOT / "monitoring" / "predictions.jsonl"


# ---- Prediction logging --------------------------------------------------
def log_prediction(
    features: list[float],
    malignant_probability: float,
    prediction: str,
) -> None:
    """Append one prediction record as a line of JSON. Never raises fatally."""
    path = predictions_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "malignant_probability": malignant_probability,
        "prediction": prediction,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def load_reference() -> pd.DataFrame:
    return pd.read_csv(reference_path())


def load_recent_predictions(window: int | None = None) -> pd.DataFrame:
    """Load logged prediction features as a DataFrame (optionally last `window`)."""
    path = predictions_log_path()
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if window is not None:
        rows = rows[-window:]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([r["features"] for r in rows])


# ---- Drift math ----------------------------------------------------------
def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    """PSI between two samples, binning on reference quantiles."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:  # near-constant feature; nothing meaningful to bin
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)
    # Floor zeros so the log is finite.
    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    psi_threshold: float = PSI_THRESHOLD,
    ks_p_threshold: float = KS_P_THRESHOLD,
) -> dict:
    """Per-feature PSI + KS drift report with an overall summary.

    `current_df` is matched to `reference_df`'s columns positionally.
    """
    current_df = current_df.copy()
    current_df.columns = list(reference_df.columns)

    features = []
    for col in reference_df.columns:
        ref = reference_df[col].to_numpy(dtype=float)
        cur = current_df[col].to_numpy(dtype=float)
        psi_val = population_stability_index(ref, cur)
        ks_stat, ks_p = ks_2samp(ref, cur)
        drifted = bool(psi_val > psi_threshold or ks_p < ks_p_threshold)
        features.append(
            {
                "feature": col,
                "psi": round(psi_val, 4),
                "ks_p_value": round(float(ks_p), 4),
                "drifted": drifted,
            }
        )

    n = len(features)
    n_drifted = sum(f["drifted"] for f in features)
    return {
        "summary": {
            "n_features": n,
            "n_drifted": n_drifted,
            "share_drifted": round(n_drifted / n, 3) if n else 0.0,
            "psi_threshold": psi_threshold,
            "ks_p_threshold": ks_p_threshold,
            "overall_drift": n_drifted > 0,
        },
        # Most-drifted features first.
        "features": sorted(features, key=lambda f: f["psi"], reverse=True),
    }
