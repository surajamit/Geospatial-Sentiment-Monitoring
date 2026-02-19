"""
Random Forest Sentiment Classifier
==================================

Implements the proposed model:

- n_estimators = 200
- max_depth = 20
- evaluated on manually labeled tweets

"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)


def build_rf_model():
    """Create RF with parameters."""

    return RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        n_jobs=-1,
        random_state=42,
    )


def evaluate_model(model, X_test, y_test):
    """Return full metric dictionary."""

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    p, r, f, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted"
    )

    return {
        "accuracy": round(acc, 3),
        "precision": round(p, 3),
        "recall": round(r, 3),
        "f1_score": round(f, 3),
    }
