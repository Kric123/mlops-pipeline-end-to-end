"""Tests for drift detection + prediction logging."""

import json

import numpy as np
import pandas as pd

from src.monitoring import (
    compute_drift_report,
    log_prediction,
    population_stability_index,
)


def _normal_df(rng, shift=0.0, n=500):
    return pd.DataFrame(
        {"f1": rng.normal(shift, 1, n), "f2": rng.normal(shift, 1, n)}
    )


def test_no_drift_for_same_distribution():
    rng = np.random.default_rng(0)
    ref, cur = _normal_df(rng), _normal_df(rng)
    report = compute_drift_report(ref, cur)
    assert report["summary"]["overall_drift"] is False


def test_drift_detected_for_shifted_distribution():
    rng = np.random.default_rng(0)
    ref = _normal_df(rng)
    cur = _normal_df(rng, shift=3.0)  # both features shifted hard
    report = compute_drift_report(ref, cur)
    assert report["summary"]["overall_drift"] is True
    assert report["summary"]["n_drifted"] == 2


def test_psi_zero_for_identical_samples():
    x = np.linspace(0.0, 1.0, 200)
    assert population_stability_index(x, x) < 1e-6


def test_log_prediction_appends_record(tmp_path, monkeypatch):
    log_file = tmp_path / "preds.jsonl"
    monkeypatch.setenv("PREDICTIONS_LOG", str(log_file))

    log_prediction([1.0, 2.0, 3.0], malignant_probability=0.91, prediction="malignant")
    log_prediction([4.0, 5.0, 6.0], malignant_probability=0.02, prediction="benign")

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["prediction"] == "malignant"
    assert rec["features"] == [1.0, 2.0, 3.0]
    assert "ts" in rec
