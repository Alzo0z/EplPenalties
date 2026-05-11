"""Feature column lists and sklearn preprocessing pipeline.

Two prediction tasks share most features:

- SHOOTER task (binary):  predict P(scored). Angle is an INPUT here (the user picks a corner).
- KEEPER  task (6-class): predict Angle. Angle is the TARGET (drop from inputs).

`make_preprocessor(task)` returns a sklearn ColumnTransformer ready to slot into a Pipeline.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "Minute",
    "Gameweek",
    "GK_Height",
    "ShooterConvRate",
    "GKSaveRate",
]

CATEGORICAL_BASE = [
    "Foot",
    "Venue",
    "Condition",
    "Continent",
    "TeamCategory",
    "Team",
]


def feature_columns(task: str) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) feature columns for the given task."""
    if task == "shooter":
        return NUMERIC_FEATURES, CATEGORICAL_BASE + ["Angle"]
    if task == "keeper":
        return NUMERIC_FEATURES, CATEGORICAL_BASE
    raise ValueError(f"Unknown task: {task!r}. Use 'shooter' or 'keeper'.")


def make_preprocessor(task: str) -> ColumnTransformer:
    num_cols, cat_cols = feature_columns(task)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ]
    )


def target_column(task: str) -> str:
    return "Scored" if task == "shooter" else "Angle"
