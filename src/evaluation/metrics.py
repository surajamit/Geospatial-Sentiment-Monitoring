"""
metrics.py
Evaluation utilities.
"""

from sklearn.metrics import (
    	classification_report,
	accuracy_score,
    	precision_score,
    	recall_score,
    	f1_score,
    	confusion_matrix,
)


def evaluate_model(model, X_test, y_test) -> dict:
    """Compute evaluation metrics."""
    y_pred = model.predict(X_test)

    return {
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "report": classification_report(y_test, y_pred),
        "cm": confusion_matrix(y_test, y_pred),
    }


def compute_metrics(y_true, y_pred):
    """Return full metric dictionary."""

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
