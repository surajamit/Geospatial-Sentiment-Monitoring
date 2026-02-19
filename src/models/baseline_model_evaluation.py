"""
Implementation-driven baseline comparison Generates real metrics from trained models

Compatible with:
- Spark pipeline outputs
- Pandas datasets
"""

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# ==============================
# CONFIG
# ==============================
RESULT_DIR = "results/baseline_comparison"
os.makedirs(RESULT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

# RF parameters (as per your paper)
RF_TREES = 200
RF_MAX_DEPTH = 20


# =====================================================
# 1. DATA LOADER
# =====================================================
def load_dataset(csv_path):
    """
    Expected columns:
    - text
    - label
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text", "label"])
    return df


# =====================================================
# 2. FEATURE ENGINEERING
# =====================================================
def vectorize_text(train_texts, test_texts):
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    return X_train, X_test, vectorizer


# =====================================================
# 3. MODEL TRAINER
# =====================================================
def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    start = time.time()

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    elapsed = time.time() - start

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, average="weighted"),
        "Recall": recall_score(y_test, preds, average="weighted"),
        "F1-Score": f1_score(y_test, preds, average="weighted"),
        "Train_Time_sec": round(elapsed, 3)
    }

    return metrics


# =====================================================
# 4. BASELINE RUNNER
# =====================================================
def run_baseline_pipeline(csv_path):
    print("Loading dataset...")
    df = load_dataset(csv_path)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    print("Vectorizing text...")
    X_train, X_test, vectorizer = vectorize_text(
        X_train_text,
        X_test_text
    )

    # ---------------- Models ----------------
    models = [
        ("Naïve Bayes", MultinomialNB()),
        ("Logistic Regression", LogisticRegression(max_iter=1000)),
        ("Support Vector Machine", LinearSVC()),
        ("RF (Proposed)", RandomForestClassifier(
            n_estimators=RF_TREES,
            max_depth=RF_MAX_DEPTH,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )),
    ]

    results = []

    print("Training models...")
    for name, model in models:
        print(f"Running {name}...")
        metrics = train_and_evaluate(
            model, X_train, X_test, y_train, y_test, name
        )
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{RESULT_DIR}/model_metrics.csv", index=False)

    print("\nBaseline results:")
    print(results_df)

    return results_df


# =====================================================
# 5. VISUALIZATION
# =====================================================
def generate_plots(results_df):
    # Accuracy plot
    plt.figure(figsize=(8, 5))
    plt.bar(results_df["Model"], results_df["Accuracy"])
    plt.title("Baseline Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/accuracy_comparison.png", dpi=300)
    plt.close()

    # PRF comparison
    metrics_df = results_df.set_index("Model")[[
        "Precision", "Recall", "F1-Score"
    ]]
    metrics_df.plot(kind="bar", figsize=(9, 5))
    plt.title("Precision, Recall and F1 Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/prf_comparison.png", dpi=300)
    plt.close()

    # Heatmap
    plt.figure(figsize=(7, 4))
    sns.heatmap(
        results_df.set_index("Model")[[
            "Accuracy", "Precision", "Recall", "F1-Score"
        ]],
        annot=True,
        fmt=".3f",
        cmap="YlGnBu"
    )
    plt.title("Baseline Model Performance Heatmap")
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/performance_heatmap.png", dpi=300)
    plt.close()

    print("Plots generated.")


# =====================================================
# 6. MAIN
# =====================================================
if __name__ == "__main__":
    DATA_PATH = "data/processed_tweets.csv"  # adjust

    results_df = run_baseline_pipeline(DATA_PATH)
    generate_plots(results_df)

    print("\n Full baseline evaluation completed.")
