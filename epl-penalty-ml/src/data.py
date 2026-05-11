"""Data loading and feature engineering for EPL penalties.

Public entrypoints:
    load_raw(path)                 -> raw DataFrame from the xlsx
    clean(df)                      -> cleaned DataFrame (drops empty rows, normalizes teams/strings)
    add_features(df)               -> adds leak-free shooter conversion rate + GK save rate
    build_dataset(raw_path, out)   -> end-to-end; saves processed parquet and lookup CSVs
    load_processed(processed_dir)  -> reads back the parquet + lookups for downstream use
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column constants
# ---------------------------------------------------------------------------
RAW_KEEP = [
    "Player Name",
    "Nationality",
    "R\\L",
    "S/F",
    "Home/Away",
    "Minute",
    "Condition",
    "GK's Name",
    "Gameweek",
    "Season",
    "Team",
    "Gk's Height",
    "Angle",
    "Continent",
    "Team Category",
]

RENAME = {
    "R\\L": "Foot",
    "S/F": "Outcome",
    "Home/Away": "Venue",
    "GK's Name": "GK",
    "Gk's Height": "GK_Height",
    "Player Name": "Player",
    "Team Category": "TeamCategory",
}

# Normalize messy team strings to a canonical form. Keep names short for the UI.
TEAM_FIX = {
    "norwich city": "Norwich City",
    "Sheffield": "Sheffield United",
    "AFC Bournemouth": "Bournemouth",
}

ANGLE_ORDER = [
    "Left Top", "Middle Top", "Right Top",
    "Left Bottom", "Middle Bottom", "Right Bottom",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    df = df.loc[:, ~df.columns.astype(str).str.contains("^Unnamed")]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[RAW_KEEP].copy()
    df = df.rename(columns=RENAME)
    df = df.dropna(subset=["Player", "Outcome", "Angle", "Team"]).copy()

    # Strip whitespace from string columns
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    # Normalize team names so logos and dropdowns are consistent
    df["Team"] = df["Team"].replace(TEAM_FIX)

    # Numeric casting (xlsx sometimes stores as object)
    df["Minute"] = pd.to_numeric(df["Minute"], errors="coerce")
    df["Gameweek"] = pd.to_numeric(df["Gameweek"], errors="coerce")
    df["GK_Height"] = pd.to_numeric(df["GK_Height"], errors="coerce")
    df = df.dropna(subset=["Minute", "Gameweek", "GK_Height"]).copy()

    # Binary target for the shooter model
    df["Scored"] = (df["Outcome"] == "Scored").astype(int)

    # Sanity: angles must be in our 6-class taxonomy
    df = df[df["Angle"].isin(ANGLE_ORDER)].copy()

    return df.reset_index(drop=True)


def _leave_one_out_rate(group: pd.Series) -> pd.Series:
    """For each row in a group, return (sum - this_row) / (count - 1).
    Used so a player's training-time conversion rate excludes the current row
    (no target leakage). If the player has only one shot, returns NaN to be
    filled with the global mean later.
    """
    total = group.sum()
    n = len(group)
    if n <= 1:
        return pd.Series(np.nan, index=group.index)
    return (total - group) / (n - 1)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds leak-free shooter conv rate and GK save rate (per row)."""
    out = df.copy()

    # Shooter conversion rate (leak-free per row)
    out["ShooterConvRate"] = (
        out.groupby("Player")["Scored"].transform(_leave_one_out_rate)
    )
    global_score_rate = out["Scored"].mean()
    out["ShooterConvRate"] = out["ShooterConvRate"].fillna(global_score_rate)

    # GK save rate = 1 - mean(scored against this GK), leak-free
    gk_scored_loo = out.groupby("GK")["Scored"].transform(_leave_one_out_rate)
    out["GKSaveRate"] = 1 - gk_scored_loo.fillna(global_score_rate)

    return out


def shooter_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """Full-dataset shooter rate + metadata for the app (no leak-removal here)."""
    g = df.groupby("Player").agg(
        Shots=("Scored", "size"),
        ConvRate=("Scored", "mean"),
        Foot=("Foot", lambda s: s.mode().iat[0]),
        Continent=("Continent", lambda s: s.mode().iat[0]),
        Team=("Team", lambda s: s.mode().iat[0]),
    ).reset_index()
    return g.sort_values("Shots", ascending=False)


def gk_lookup(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("GK").agg(
        Faced=("Scored", "size"),
        SaveRate=("Scored", lambda s: 1 - s.mean()),
        Height=("GK_Height", "mean"),
    ).reset_index()
    return g.sort_values("Faced", ascending=False)


def build_dataset(raw_path: str | Path, out_dir: str | Path) -> dict[str, Path]:
    raw_path = Path(raw_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = clean(load_raw(raw_path))
    df = add_features(df)

    parquet_path = out_dir / "penalties.parquet"
    df.to_parquet(parquet_path, index=False)

    shooters = shooter_lookup(df)
    gks = gk_lookup(df)
    shooters.to_csv(out_dir / "shooters.csv", index=False)
    gks.to_csv(out_dir / "gks.csv", index=False)

    return {
        "parquet": parquet_path,
        "shooters": out_dir / "shooters.csv",
        "gks": out_dir / "gks.csv",
        "n_rows": len(df),
        "n_players": df["Player"].nunique(),
        "n_gks": df["GK"].nunique(),
        "n_teams": df["Team"].nunique(),
    }


def load_processed(processed_dir: str | Path):
    p = Path(processed_dir)
    df = pd.read_parquet(p / "penalties.parquet")
    shooters = pd.read_csv(p / "shooters.csv")
    gks = pd.read_csv(p / "gks.csv")
    return df, shooters, gks


if __name__ == "__main__":
    here = Path(__file__).resolve().parents[1]
    info = build_dataset(here / "data/raw/Epl Penalties.xlsx", here / "data/processed")
    print({k: (str(v) if isinstance(v, Path) else v) for k, v in info.items()})
