"""Train LogReg, SVM, MLP for both the shooter (binary) and keeper (6-class) tasks.

Usage:
    python -m src.train

What it does for each task:
  1. Load processed parquet.
  2. Random 80/20 stratified split (random_state=42).
  3. For each algorithm: build a Pipeline(preprocessor [-> SMOTE for keeper] -> estimator).
  4. Quick GridSearchCV over a small, sensible grid (3-fold for speed).
  5. Refit best on the full training set, evaluate on the held-out 20%.
  6. Save model joblib, predictions, metrics, plots.

All artifacts land in models/ and report/figures/.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

# Make `src.*` importable when run as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.data import ANGLE_ORDER, load_processed
from src.evaluate import (
    keeper_metrics, plot_confusion, plot_metric_bars, plot_roc, plot_roc_overlay,
    shooter_metrics, topk_accuracy,
)
from src.features import feature_columns, make_preprocessor, target_column

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "report" / "figures"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------
def shooter_models() -> dict:
    """Estimators + small grids for the binary shooter task."""
    return {
        "LogReg": (
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "SVM": (
            SVC(
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_STATE,
            ),
            {"clf__C": [0.5, 1.0, 5.0], "clf__kernel": ["linear", "rbf"]},
        ),
        "MLP": (
            MLPClassifier(
                hidden_layer_sizes=(32, 16),
                max_iter=600,
                early_stopping=True,
                random_state=RANDOM_STATE,
            ),
            {"clf__alpha": [1e-4, 1e-3, 1e-2]},
        ),
    }


# Estimators that do not support class_weight; we use SMOTE on the binary task
# to balance the training data for them instead.
SHOOTER_USE_SMOTE = {"MLP"}


def keeper_models() -> dict:
    """Estimators + small grids for the 6-class keeper task."""
    return {
        "LogReg": (
            LogisticRegression(
                max_iter=2000,
                multi_class="multinomial",
                random_state=RANDOM_STATE,
            ),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "SVM": (
            SVC(
                probability=True,
                random_state=RANDOM_STATE,
            ),
            {"clf__C": [0.5, 1.0, 5.0], "clf__kernel": ["linear", "rbf"]},
        ),
        # early_stopping disabled: validation split + string labels triggers a
        # NaN-check incompatibility in current sklearn. L2 (alpha) does the
        # regularization here on this small dataset.
        "MLP": (
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=800,
                early_stopping=False,
                random_state=RANDOM_STATE,
            ),
            {"clf__alpha": [1e-4, 1e-3, 1e-2]},
        ),
    }


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------
def _build_pipeline(task: str, estimator, use_smote: bool):
    pre = make_preprocessor(task)
    steps = [("pre", pre)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)))
    steps.append(("clf", estimator))
    return ImbPipeline(steps)


def train_shooter(df: pd.DataFrame) -> pd.DataFrame:
    num_cols, cat_cols = feature_columns("shooter")
    X = df[num_cols + cat_cols]
    y = df[target_column("shooter")]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
    )

    rows = []
    curves = {}
    for name, (est, grid) in shooter_models().items():
        # LogReg/SVM use class_weight='balanced'; MLP doesn't support that, so
        # we SMOTE the training data for MLP only.
        pipe = _build_pipeline("shooter", est, use_smote=name in SHOOTER_USE_SMOTE)
        search = GridSearchCV(
            pipe, grid, cv=3, scoring="roc_auc", n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_

        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:, 1]
        m = shooter_metrics(y_test.values, y_pred, y_proba)
        m["model"] = name
        m["best_params"] = json.dumps(search.best_params_)
        rows.append(m)
        curves[name] = (y_test.values, y_proba)

        joblib.dump(best, MODELS_DIR / f"shooter_{name.lower()}.joblib")
        plot_confusion(
            y_test.values, y_pred, labels=[0, 1],
            title=f"Shooter / {name} — Confusion",
            out_path=FIG_DIR / f"shooter_{name.lower()}_cm.png",
        )
        report_txt = classification_report(
            y_test.values, y_pred, target_names=["Missed", "Scored"], digits=3,
        )
        (MODELS_DIR / f"shooter_{name.lower()}_report.txt").write_text(report_txt)

    plot_roc_overlay(
        curves, "Shooter — ROC overlay",
        out_path=FIG_DIR / "shooter_roc_overlay.png",
    )

    res = pd.DataFrame(rows)[["model", "accuracy", "roc_auc", "f1", "best_params"]]
    res.to_csv(MODELS_DIR / "shooter_results.csv", index=False)
    plot_metric_bars(res, "roc_auc", "Shooter — ROC-AUC",
                     out_path=FIG_DIR / "shooter_roc_auc_bar.png")
    plot_metric_bars(res, "accuracy", "Shooter — Accuracy",
                     out_path=FIG_DIR / "shooter_accuracy_bar.png")
    return res


def train_keeper(df: pd.DataFrame) -> pd.DataFrame:
    num_cols, cat_cols = feature_columns("keeper")
    X = df[num_cols + cat_cols]
    y = df[target_column("keeper")]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
    )

    rows = []
    for name, (est, grid) in keeper_models().items():
        pipe = _build_pipeline("keeper", est, use_smote=True)
        search = GridSearchCV(
            pipe, grid, cv=3, scoring="f1_macro", n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_

        y_pred = best.predict(X_test)
        proba = best.predict_proba(X_test)
        classes = list(best.classes_)
        m = keeper_metrics(y_test.values, y_pred, classes)
        m["top2_accuracy"] = topk_accuracy(y_test.values, proba, classes, k=2)
        m["model"] = name
        m["best_params"] = json.dumps(search.best_params_)
        rows.append(m)

        joblib.dump(best, MODELS_DIR / f"keeper_{name.lower()}.joblib")
        plot_confusion(
            y_test.values, y_pred, labels=ANGLE_ORDER,
            title=f"Keeper / {name} — Confusion",
            out_path=FIG_DIR / f"keeper_{name.lower()}_cm.png",
        )
        report_txt = classification_report(
            y_test.values, y_pred, labels=ANGLE_ORDER, digits=3, zero_division=0,
        )
        (MODELS_DIR / f"keeper_{name.lower()}_report.txt").write_text(report_txt)

    res = pd.DataFrame(rows)[
        ["model", "accuracy", "macro_f1", "top2_accuracy", "best_params"]
    ]
    res.to_csv(MODELS_DIR / "keeper_results.csv", index=False)
    plot_metric_bars(res, "macro_f1", "Keeper — Macro F1",
                     out_path=FIG_DIR / "keeper_macro_f1_bar.png")
    plot_metric_bars(res, "top2_accuracy", "Keeper — Top-2 Accuracy",
                     out_path=FIG_DIR / "keeper_top2_bar.png")
    return res


def _plot_combined_metrics(shooter_res, keeper_res, out_path):
    """One light-themed figure combining shooter ROC-AUC and keeper Top-2 + Macro-F1."""
    import matplotlib.pyplot as plt
    plt.style.use("default")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.patch.set_facecolor("white")
    palette = ["#1565C0", "#2E7D32", "#C62828"]

    # Left: shooter
    ax = axes[0]
    x = range(len(shooter_res))
    width = 0.35
    ax.bar([i - width / 2 for i in x], shooter_res["roc_auc"], width,
           color=palette[0], label="ROC-AUC")
    ax.bar([i + width / 2 for i in x], shooter_res["accuracy"], width,
           color=palette[1], label="Accuracy", alpha=0.85)
    ax.axhline(0.5, ls="--", c="#999", lw=1, label="Random AUC")
    ax.set_xticks(list(x))
    ax.set_xticklabels(shooter_res["model"], fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title("Shooter task — binary", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    for i, (auc, acc) in enumerate(zip(shooter_res["roc_auc"], shooter_res["accuracy"])):
        ax.text(i - width / 2, auc + 0.015, f"{auc:.2f}", ha="center", fontsize=10)
        ax.text(i + width / 2, acc + 0.015, f"{acc:.2f}", ha="center", fontsize=10)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # Right: keeper
    ax = axes[1]
    x = range(len(keeper_res))
    ax.bar([i - width / 2 for i in x], keeper_res["top2_accuracy"], width,
           color=palette[0], label="Top-2 Acc")
    ax.bar([i + width / 2 for i in x], keeper_res["macro_f1"], width,
           color=palette[2], label="Macro-F1", alpha=0.85)
    ax.axhline(1 / 6, ls=":", c="#999", lw=1, label="Random Top-1")
    ax.axhline(2 / 6, ls="--", c="#999", lw=1, label="Random Top-2")
    ax.set_xticks(list(x))
    ax.set_xticklabels(keeper_res["model"], fontsize=12)
    ax.set_ylim(0, 0.7)
    ax.set_title("Keeper task — 6-class angle", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    for i, (t2, mf) in enumerate(zip(keeper_res["top2_accuracy"], keeper_res["macro_f1"])):
        ax.text(i - width / 2, t2 + 0.012, f"{t2:.2f}", ha="center", fontsize=10)
        ax.text(i + width / 2, mf + 0.012, f"{mf:.2f}", ha="center", fontsize=10)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor="white")
    plt.close(fig)


def main():
    df, _, _ = load_processed(ROOT / "data/processed")
    print(f"Loaded {len(df)} rows.")

    print("\n=== Shooter task (binary: Scored / Missed) ===")
    shooter_res = train_shooter(df)
    print(shooter_res.to_string(index=False))

    print("\n=== Keeper task (6-class angle) ===")
    keeper_res = train_keeper(df)
    print(keeper_res.to_string(index=False))

    _plot_combined_metrics(shooter_res, keeper_res, FIG_DIR / "combined_metrics.png")
    print(f"\nAll artifacts saved to {MODELS_DIR} and {FIG_DIR}.")


if __name__ == "__main__":
    main()
