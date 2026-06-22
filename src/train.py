"""
M1 — Train a baseline model and track the run with MLflow (+ rich evaluation).

This is the foundation of the MLOps pipeline. The model itself is intentionally
simple; the point is the *practice*: every run logs its hyperparameters, metrics,
the trained model, AND a full evaluation suite (confusion matrix, per-class report,
ROC + precision-recall curves, threshold sweep) so results are reproducible,
comparable, and *interpretable*.

Positive class = MALIGNANT. See docs/evaluation.md for why: in a medical screening
context the costly error is missing a cancer (a false negative on malignant), so we
frame the metrics around detecting malignant tumors and care most about RECALL.

Run:
    python src/train.py

Then inspect runs, metrics, and the logged plots in the MLflow UI:
    mlflow ui    ->    http://127.0.0.1:5000
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend — render plots to files, no GUI window
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ---- Configuration -------------------------------------------------------
# In a real pipeline these would come from a config file or CLI args. Keeping
# them here as named constants makes the logged hyperparameters obvious.
EXPERIMENT_NAME = "breast-cancer-baseline"
REGISTERED_MODEL_NAME = "breast-cancer-classifier"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Pin the tracking backend to a SQLite DB in the project root so runs land in the
# same place regardless of the working directory the script is launched from
# (and so serve.py can find the registered model). MLflow 3.x defaults to SQLite.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")

# We relabel so MALIGNANT = 1 (the positive class we care about detecting).
# sklearn's raw encoding is 0 = malignant, 1 = benign; we flip it below.
CLASS_NAMES = ["benign", "malignant"]  # index 0 -> benign, index 1 -> malignant
POSITIVE_LABEL = 1  # malignant

# Thresholds to probe for the precision/recall trade-off (see docs/evaluation.md §4).
THRESHOLD_SWEEP = [0.3, 0.4, 0.5, 0.6, 0.7]

HYPERPARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
}


def load_data():
    """Load Breast Cancer Wisconsin and relabel so malignant = 1 (positive)."""
    data = load_breast_cancer(as_frame=True)
    X = data.data
    # Raw: 0 = malignant, 1 = benign. Flip so malignant becomes the positive (1).
    y = 1 - data.target
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def compute_metrics(y_test, preds, proba) -> dict[str, float]:
    """Standard classification metric suite, framed around the positive class."""
    return {
        "accuracy": accuracy_score(y_test, preds),
        # precision/recall/f1 are computed for the positive class (malignant).
        "precision_malignant": precision_score(y_test, preds, pos_label=POSITIVE_LABEL),
        "recall_malignant": recall_score(y_test, preds, pos_label=POSITIVE_LABEL),
        "f1_malignant": f1_score(y_test, preds, pos_label=POSITIVE_LABEL),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def log_confusion_matrix(y_test, preds) -> dict[str, int]:
    """Log the confusion matrix as both raw counts (metrics) and a plot (artifact)."""
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    # cm layout for labels [0, 1]:  [[TN, FP], [FN, TP]]  (positive = 1 = malignant)
    tn, fp, fn, tp = cm.ravel()
    counts = {"true_neg": int(tn), "false_pos": int(fp),
              "false_neg": int(fn), "true_pos": int(tp)}
    mlflow.log_metrics(counts)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix (positive = malignant)")
    fig.tight_layout()
    mlflow.log_figure(fig, "plots/confusion_matrix.png")
    plt.close(fig)
    return counts


def log_curves(y_test, proba) -> None:
    """Log ROC and Precision-Recall curves as artifacts (threshold-independent views)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_test, proba, pos_label=POSITIVE_LABEL, ax=ax)
    ax.set_title("ROC Curve")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)  # chance line
    fig.tight_layout()
    mlflow.log_figure(fig, "plots/roc_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    PrecisionRecallDisplay.from_predictions(y_test, proba, pos_label=POSITIVE_LABEL, ax=ax)
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    mlflow.log_figure(fig, "plots/precision_recall_curve.png")
    plt.close(fig)


def log_classification_report(y_test, preds) -> str:
    """Log the full per-class precision/recall/F1 report as a text artifact."""
    report = classification_report(
        y_test, preds, labels=[0, 1], target_names=CLASS_NAMES, digits=4
    )
    mlflow.log_text(report, "reports/classification_report.txt")
    return report


def threshold_sweep(y_test, proba) -> str:
    """Show how precision/recall move as the decision threshold changes.

    This makes the core trade-off concrete: lowering the threshold catches more
    cancers (recall up) at the cost of more false alarms (precision down).
    """
    lines = [f"{'threshold':>9} | {'precision':>9} | {'recall':>7} | {'f1':>7}"]
    lines.append("-" * 42)
    for t in THRESHOLD_SWEEP:
        preds_t = (proba >= t).astype(int)
        p = precision_score(y_test, preds_t, pos_label=POSITIVE_LABEL, zero_division=0)
        r = recall_score(y_test, preds_t, pos_label=POSITIVE_LABEL, zero_division=0)
        f = f1_score(y_test, preds_t, pos_label=POSITIVE_LABEL, zero_division=0)
        lines.append(f"{t:>9.2f} | {p:>9.4f} | {r:>7.4f} | {f:>7.4f}")
        # Also log recall at each threshold as a metric so it's filterable in the UI.
        mlflow.log_metric(f"recall_at_threshold_{int(t * 100)}", r)
    table = "\n".join(lines)
    mlflow.log_text(table, "reports/threshold_sweep.txt")
    return table


def main() -> None:
    X_train, X_test, y_train, y_test = load_data()

    # All runs land under one named experiment so they're easy to compare.
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        # 1. Log the hyperparameters + dataset facts that define this run.
        mlflow.log_params(HYPERPARAMS)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_rows", X_train.shape[0])
        mlflow.log_param("positive_class", "malignant")

        # 2. Train.
        model = RandomForestClassifier(**HYPERPARAMS)
        model.fit(X_train, y_train)

        # 3. Predict once, reuse everywhere.
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, POSITIVE_LABEL]  # P(malignant)

        # 4. Headline metrics.
        metrics = compute_metrics(y_test, preds, proba)
        mlflow.log_metrics(metrics)

        # 5. Rich, interpretable evaluation (see docs/evaluation.md).
        counts = log_confusion_matrix(y_test, preds)
        log_curves(y_test, proba)
        report = log_classification_report(y_test, preds)
        sweep = threshold_sweep(y_test, proba)

        # 6. Log + register the model so later milestones can pull it by name.
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X_test.iloc[:5],
        )

        # ---- Console summary -------------------------------------------------
        print(f"Run ID: {run.info.run_id}")
        print("\nHeadline metrics (positive class = malignant):")
        for k, v in metrics.items():
            print(f"  {k:20s} = {v:.4f}")
        print("\nConfusion matrix counts:")
        print(f"  true_neg(benign ok)  = {counts['true_neg']:3d}   "
              f"false_pos(false alarm) = {counts['false_pos']:3d}")
        print(f"  false_neg(MISSED CA) = {counts['false_neg']:3d}   "
              f"true_pos(caught CA)    = {counts['true_pos']:3d}")
        print("\nPer-class report:")
        print(report)
        print("Threshold sweep (precision/recall trade-off):")
        print(sweep)
        print(f"\nModel registered as '{REGISTERED_MODEL_NAME}'.")
        print("Plots + reports logged as artifacts. View with:")
        print("  mlflow ui   ->   http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
