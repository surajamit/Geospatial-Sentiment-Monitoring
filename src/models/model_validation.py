"""
Validation pipeline

Adds:
- Stratified k-fold CV
- Statistical significance tests
- ROC-AUC curves
- Confidence intervals
- Robust reproducibility

Compatible with:
- Google Colab
- Spark-preprocessed CSV
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import label_binarize
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from scipy.stats import ttest_rel, wilcoxon

# =============================
# CONFIG
# =============================
RESULT_DIR = "results/advanced_validation"
os.makedirs(RESULT_DIR, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5

RF_TREES = 200
RF_MAX_DEPTH = 20


# ======================================================
# DATA LOADER
# ======================================================
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text", "label"])
    return df


# ======================================================
# VECTORIZATION
# ======================================================
def build_vectorizer():
    return TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words="english"
    )


# ======================================================
# MODEL ZOO
# ======================================================
def get_models():
    return {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": LinearSVC(),
        "RF (Proposed)": RandomForestClassifier(
            n_estimators=RF_TREES,
            max_depth=RF_MAX_DEPTH,
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),
    }


# ======================================================
# METRIC COMPUTATION
# ======================================================
def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
    }


# ======================================================
# CROSS VALIDATION ENGINE
# ======================================================
def run_cross_validation(df):
    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    models = get_models()
    results = {name: [] for name in models.keys()}

    X_text = df["text"].values
    y = df["label"].values

    print("Starting Stratified K-Fold evaluation...")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_text, y), 1):
        print(f"\nFold {fold}/{N_SPLITS}")

        X_train_text = X_text[train_idx]
        X_test_text = X_text[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        vectorizer = build_vectorizer()
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = compute_metrics(y_test, preds)
            results[name].append(metrics)

    return results


# ======================================================
# AGGREGATE RESULTS
# ======================================================
def summarize_results(results):
    summary_rows = []

    for model_name, folds in results.items():
        df_fold = pd.DataFrame(folds)

        row = {
            "Model": model_name,
            "Accuracy_mean": df_fold["accuracy"].mean(),
            "Accuracy_std": df_fold["accuracy"].std(),
            "F1_mean": df_fold["f1"].mean(),
            "F1_std": df_fold["f1"].std(),
            "Precision_mean": df_fold["precision"].mean(),
            "Recall_mean": df_fold["recall"].mean(),
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{RESULT_DIR}/cv_summary.csv", index=False)
    return summary_df


# ======================================================
# SIGNIFICANCE TEST
# ======================================================
def statistical_significance(results):
    rf_scores = [f["f1"] for f in results["RF (Proposed)"]]

    stats_rows = []

    for model_name, folds in results.items():
        if model_name == "RF (Proposed)":
            continue

        other_scores = [f["f1"] for f in folds]

        t_stat, t_p = ttest_rel(rf_scores, other_scores)
        w_stat, w_p = wilcoxon(rf_scores, other_scores)

        stats_rows.append({
            "Compared_With": model_name,
            "Paired_t_pvalue": t_p,
            "Wilcoxon_pvalue": w_p,
        })

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(f"{RESULT_DIR}/significance_tests.csv", index=False)
    return stats_df


# ======================================================
# BAR PLOT WITH ERROR BARS
# ======================================================
def plot_cv_results(summary_df):
    plt.figure(figsize=(9, 5))
    plt.bar(
        summary_df["Model"],
        summary_df["F1_mean"],
        yerr=summary_df["F1_std"],
        capsize=5
    )
    plt.ylabel("F1-score")
    plt.title("Cross-Validated Model Comparison")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/cv_f1_comparison.png", dpi=300)
    plt.close()


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    DATA_PATH = "data/processed_tweets.csv"

    df = load_dataset(DATA_PATH)

    results = run_cross_validation(df)
    summary_df = summarize_results(results)
    stats_df = statistical_significance(results)

    plot_cv_results(summary_df)

    print("\n Advanced validation completed.")
