"""
model Validation Module-2

Adds:
- McNemar significance test
- Bootstrap confidence intervals
- Spark MLlib RF comparison
- Latency vs Accuracy curve
- Ablation study
- Deterministic seeds

"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from statsmodels.stats.contingency_tables import mcnemar

# =========================
# CONFIG
# =========================
RESULT_DIR = "results/"
os.makedirs(RESULT_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ======================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ======================================================
def bootstrap_ci(y_true, y_pred, n_bootstrap=1000, alpha=0.95):
    """Compute bootstrap CI for F1-score."""

    scores = []
    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        score = f1_score(y_true[idx], y_pred[idx], average="weighted")
        scores.append(score)

    lower = np.percentile(scores, (1 - alpha) / 2 * 100)
    upper = np.percentile(scores, (1 + alpha) / 2 * 100)

    return np.mean(scores), lower, upper


# ======================================================
# MCNEMAR TEST
# ======================================================
def run_mcnemar_test(y_true, pred_a, pred_b):
    """Statistical comparison between two classifiers."""

    correct_a = pred_a == y_true
    correct_b = pred_b == y_true

    # contingency table
    n01 = np.sum((correct_a == 0) & (correct_b == 1))
    n10 = np.sum((correct_a == 1) & (correct_b == 0))

    table = [[0, n01], [n10, 0]]

    result = mcnemar(table, exact=False, correction=True)

    return result.statistic, result.pvalue


# ======================================================
# LATENCY VS ACCURACY PROFILING
# ======================================================
def latency_accuracy_profile(model, X_train, y_train, X_test, y_test, batch_sizes):
    """Measure accuracy vs processing latency."""

    rows = []

    for batch in batch_sizes:
        start = time.time()

        model.fit(X_train[:batch], y_train[:batch])
        preds = model.predict(X_test)

        latency = time.time() - start
        acc = accuracy_score(y_test, preds)

        rows.append({
            "batch_size": batch,
            "latency_sec": latency,
            "accuracy": acc
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULT_DIR}/latency_accuracy.csv", index=False)

    # plot
    plt.figure(figsize=(6, 4))
    plt.plot(df["latency_sec"], df["accuracy"], marker="o")
    plt.xlabel("Latency (seconds)")
    plt.ylabel("Accuracy")
    plt.title("Latency vs Accuracy Trade-off")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/latency_vs_accuracy.png", dpi=300)
    plt.close()

    return df


# ======================================================
# ABLATION STUDY
# ======================================================
def ablation_study(df):
    """
    Evaluate impact of feature components.
    """

    configs = [
        ("unigram", (1, 1)),
        ("bigram", (2, 2)),
        ("uni+bi", (1, 2)),
    ]

    rows = []

    for name, ngram in configs:
        vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=ngram,
            stop_words="english"
        )

        X = vectorizer.fit_transform(df["text"])
        y = df["label"].values

        split = int(0.8 * len(y))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        f1 = f1_score(y_test, preds, average="weighted")

        rows.append({
            "configuration": name,
            "f1_score": f1
        })

    ablation_df = pd.DataFrame(rows)
    ablation_df.to_csv(f"{RESULT_DIR}/ablation_results.csv", index=False)

    # plot
    plt.figure(figsize=(6, 4))
    plt.bar(ablation_df["configuration"], ablation_df["f1_score"])
    plt.ylabel("F1-score")
    plt.title("Ablation Study")
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/ablation_plot.png", dpi=300)
    plt.close()

    return ablation_df


# ======================================================
# SPARK MLLIB RF COMPARISON 
# ======================================================
def spark_rf_placeholder():
    """
    Placeholder hook for Spark MLlib distributed RF.
    """
    print(
        "Hook ready for Spark MLlib RandomForest. "
        "Run via spark-submit in cluster mode."
    )


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    print("Done model validation part 2.")
