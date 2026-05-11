"""Evaluation + plotting helpers shared by notebooks and the app."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def shooter_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred, pos_label=1),
    }


def keeper_metrics(y_true, y_pred, classes) -> dict:
    top1 = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return {"accuracy": top1, "macro_f1": macro_f1}


def topk_accuracy(y_true, proba, classes, k=2) -> float:
    classes = list(classes)
    idx = {c: i for i, c in enumerate(classes)}
    top_k_idx = np.argsort(-proba, axis=1)[:, :k]
    hits = 0
    for i, t in enumerate(y_true):
        if idx[t] in top_k_idx[i]:
            hits += 1
    return hits / len(y_true)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_confusion(y_true, y_pred, labels, title, out_path: Path | None = None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, cbar=False, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_roc(y_true, y_proba, title, out_path: Path | None = None):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_roc_overlay(curves: dict, title, out_path: Path | None = None):
    """curves: {model_name: (y_true, y_proba)}"""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, (y_true, y_proba) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig


def plot_metric_bars(results_df: pd.DataFrame, metric: str, title: str,
                     out_path: Path | None = None):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=results_df, x="model", y=metric, ax=ax, palette="viridis")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    for i, v in enumerate(results_df[metric]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig
