"""
Random Forest Sentiment Classifier (3-Class)
===========================================

Classes:
    0 = negative
    1 = neutral
    2 = positive
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)


def build_rf():
    """RF parameters."""

    return RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
    )


def evaluate_rf(model, X_test, y_test):
    """Return metrics dictionary."""

    y_pred = model.predict(X_test)

    p, r, f, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted"
    )

    return {
        "f1_score": round(f, 3),
        "precision": round(p, 3),
        "recall": round(r, 3),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
    }
