"""
plots.py
Visualization Suite For:
- Stratified K-Fold CV
- Multiclass ROC
- KDE spatial density
- Statistical significance tests
- Latency vs Accuracy
- Resource utilization
- Parallel efficiency & speedup

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_curve,
    auc,
    f1_score,
    accuracy_score
)
from sklearn.preprocessing import label_binarize
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# GLOBAL STYLE (publication)
# =========================
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 300
})

sns.set_style("whitegrid")

RESULT_DIR = "results/"
import os
os.makedirs(RESULT_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ======================================================
# 1. STRATIFIED K-FOLD CROSS VALIDATION
# ======================================================
def stratified_kfold_evaluation(texts, labels, k=5):
    """Performs stratified CV and plots fold performance."""

    vectorizer = TfidfVectorizer(max_features=20000, stop_words="english")
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)

    scores = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )

        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])

        f1 = f1_score(y[test_idx], preds, average="weighted")
        scores.append(f1)

    # plot
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, k+1), scores, marker="o")
    plt.xlabel("Fold")
    plt.ylabel("F1-score")
    plt.title("Stratified K-Fold Cross-Validation Performance")
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/stratified_kfold.png")
    plt.close()

    return scores


# ======================================================
# 2. MULTICLASS ROC-AUC
# ======================================================
def plot_multiclass_roc(model, X_test, y_test, class_names):
    """Multiclass ROC curve."""

    y_bin = label_binarize(y_test, classes=np.unique(y_test))
    y_score = model.predict_proba(X_test)

    plt.figure(figsize=(6, 5))

    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC-AUC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/multiclass_roc.png")
    plt.close()


# ======================================================
# 3. KDE SPATIAL INTENSITY
# ======================================================
def plot_kde_heatmap(latitudes, longitudes, tech_name):
    """Kernel density spatial heatmap."""

    df = pd.DataFrame({"lat": latitudes, "lon": longitudes})

    plt.figure(figsize=(6, 5))
    sns.kdeplot(
        data=df,
        x="lon",
        y="lat",
        fill=True,
        cmap="Reds",
        bw_adjust=0.5,
        thresh=0.05
    )

    plt.title(f"KDE Spatial Intensity — {tech_name}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/kde_{tech_name}.png")
    plt.close()


# ======================================================
# 4. STATISTICAL SIGNIFICANCE
# ======================================================
def statistical_tests(y_true, pred_rf, pred_svm):
    """Paired t-test, Wilcoxon, McNemar."""

    rf_correct = (pred_rf == y_true).astype(int)
    svm_correct = (pred_svm == y_true).astype(int)

    # paired t-test
    t_stat, t_p = ttest_rel(rf_correct, svm_correct)

    # Wilcoxon
    w_stat, w_p = wilcoxon(rf_correct, svm_correct)

    # McNemar
    n01 = np.sum((rf_correct == 0) & (svm_correct == 1))
    n10 = np.sum((rf_correct == 1) & (svm_correct == 0))
    table = [[0, n01], [n10, 0]]
    mc = mcnemar(table, exact=False, correction=True)

    results = {
        "paired_t_p": t_p,
        "wilcoxon_p": w_p,
        "mcnemar_p": mc.pvalue
    }

    pd.DataFrame([results]).to_csv(
        f"{RESULT_DIR}/statistical_tests.csv",
        index=False
    )

    return results


# ======================================================
# 5. LATENCY vs ACCURACY
# ======================================================
def latency_accuracy_curve(model, X_train, y_train, X_test, y_test):
    """Micro-batch trade-off curve."""

    batch_sizes = [500, 1000, 2000, 3000, 4000]
    rows = []

    for b in batch_sizes:
        start = time.time()
        model.fit(X_train[:b], y_train[:b])
        preds = model.predict(X_test)

        latency = time.time() - start
        acc = accuracy_score(y_test, preds)

        rows.append((latency, acc))

    df = pd.DataFrame(rows, columns=["latency", "accuracy"])

    plt.figure(figsize=(6, 4))
    plt.plot(df["latency"], df["accuracy"], marker="o")
    plt.xlabel("Latency (sec)")
    plt.ylabel("Accuracy")
    plt.title("Latency vs Accuracy Trade-off")
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/latency_accuracy.png")
    plt.close()

    return df


# ======================================================
# 6. RESOURCE UTILIZATION
# ======================================================
def plot_resource_utilization(cpu_series, mem_series):
    """Executor CPU and memory usage."""

    plt.figure(figsize=(6, 4))
    plt.plot(cpu_series, label="CPU Utilization (%)")
    plt.xlabel("Time Window")
    plt.ylabel("CPU %")
    plt.title("Executor CPU Utilization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/cpu_utilization.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(mem_series, label="Memory Utilization (%)")
    plt.xlabel("Time Window")
    plt.ylabel("Memory %")
    plt.title("Executor Memory Utilization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/memory_utilization.png")
    plt.close()


# ======================================================
# 7. SPEEDUP & EFFICIENCY
# ======================================================
def plot_scalability_curves(nodes, runtimes):
    """Parallel speedup and efficiency."""

    T1 = runtimes[0]
    speedup = T1 / np.array(runtimes)
    efficiency = speedup / np.array(nodes)

    # speedup
    plt.figure(figsize=(6, 4))
    plt.plot(nodes, speedup, marker="o")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Speedup")
    plt.title("Parallel Speedup Curve")
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/speedup_curve.png")
    plt.close()

    # efficiency
    plt.figure(figsize=(6, 4))
    plt.plot(nodes, efficiency, marker="o")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Efficiency")
    plt.title("Parallel Efficiency Curve")
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/efficiency_curve.png")
    plt.close()



def plot_confusion_matrix(cm, labels):
    """Styled confusion matrix."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix (RF Classifier)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def plot_prf_bars(precision, recall, f1, labels):
    """Compact grouped bar chart."""
    x = np.arange(len(labels))
    width = 0.22

    plt.figure(figsize=(7, 4))
    plt.bar(x - width, precision, width, label="Precision")
    plt.bar(x, recall, width, label="Recall")
    plt.bar(x + width, f1, width, label="F1-score")

    plt.xticks(x, labels)
    plt.legend(loc="upper center", ncol=3)
    plt.tight_layout()
    plt.show()
