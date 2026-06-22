"""Unit tests for the evaluation logic in train.py."""

from src.train import compute_metrics


def test_compute_metrics_perfect_predictions():
    """Perfect predictions should score 1.0 across the board."""
    y_true = [0, 1, 1, 0, 1]
    preds = [0, 1, 1, 0, 1]
    proba = [0.05, 0.95, 0.88, 0.10, 0.77]  # confident & correct
    m = compute_metrics(y_true, preds, proba)
    assert m["accuracy"] == 1.0
    assert m["precision_malignant"] == 1.0
    assert m["recall_malignant"] == 1.0
    assert m["f1_malignant"] == 1.0
    assert m["roc_auc"] == 1.0


def test_recall_drops_when_a_malignant_is_missed():
    """One malignant (1) predicted benign (0) must lower recall but not precision."""
    y_true = [1, 1, 0, 0]
    preds = [1, 0, 0, 0]  # missed the 2nd malignant -> a false negative
    proba = [0.9, 0.3, 0.2, 0.1]
    m = compute_metrics(y_true, preds, proba)
    assert m["recall_malignant"] == 0.5      # caught 1 of 2 malignant
    assert m["precision_malignant"] == 1.0   # no false alarms
