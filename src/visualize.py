"""Generate EDA and model-comparison figures."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, confusion_matrix

from src.features import extract_features

sns.set_theme(style="whitegrid", palette="deep")


def plot_label_distribution(df: pd.DataFrame, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["label"].value_counts()
    ax.bar(counts.index, counts.values, color=["#7C5CFF", "#4C8DFF"])
    ax.set_title("Label Distribution")
    ax.set_ylabel("Samples")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 10, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_by_label(df: pd.DataFrame, feature: str, title: str, out_path: str | Path) -> None:
    feats = df["text"].apply(extract_features).apply(pd.Series)
    feats["label"] = df["label"].values
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=feats, x="label", y=feature, ax=ax, showfliers=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_model_comparison(results_df: pd.DataFrame, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    x = range(len(results_df))
    width = 0.15
    for i, metric in enumerate(metrics):
        ax.bar([p + i * width for p in x], results_df[metric], width=width, label=metric)
    ax.set_xticks([p + width * 2 for p in x])
    ax.set_xticklabels(results_df["model"], rotation=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_test, y_pred, model_name: str, out_path: str | Path) -> None:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Purples", cbar=False,
        xticklabels=["Human", "AI"], yticklabels=["Human", "AI"], ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(pipeline, X_test, y_test, model_name: str, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax, name=model_name)
    ax.set_title(f"ROC Curve — {model_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
