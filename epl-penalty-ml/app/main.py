"""EPL Penalty Predictor — Streamlit demo.

Three tabs:
  - Shooter view: click an SVG goal-mouth corner; model returns P(scored).
  - Keeper view:  model predicts a corner heatmap; click to dive to save it.
  - Model comparison: tables, ROC overlay, confusion matrices.

Run:
    streamlit run app/main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `src.*` importable when launched via `streamlit run app/main.py`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import streamlit as st

from app.components.goalmouth_svg import (
    ANGLES, corner_buttons, render_svg_goal,
)
from app.components.logos import get_logo_path
from src.data import load_processed

# ---------------------------------------------------------------------------
# Page config — light theme is pinned in .streamlit/config.toml
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EPL Penalty Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject a small CSS polish for headings, metrics, and tab styling.
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
      h1, h2, h3 {color: #102A55;}
      div[data-testid="stMetricValue"] {color: #1565C0; font-weight: 700;}
      .stTabs [data-baseweb="tab-list"] {gap: 8px;}
      .stTabs [data-baseweb="tab"] {font-size: 16px; font-weight: 600;
        padding: 8px 18px; background: #F4F7FB; border-radius: 8px 8px 0 0;}
      .stTabs [aria-selected="true"] {background: #1565C0 !important;
        color: white !important;}
      div[data-testid="stSidebar"] {background: #F4F7FB;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data + model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def _load_data():
    df, shooters, gks = load_processed(ROOT / "data/processed")
    return df, shooters, gks


@st.cache_resource
def _load_models():
    # Display name must match the sidebar selectbox values exactly.
    name_map = {"logreg": "LogReg", "svm": "SVM", "mlp": "MLP"}
    out = {"shooter": {}, "keeper": {}}
    for file_stem, display in name_map.items():
        out["shooter"][display] = joblib.load(
            ROOT / f"models/shooter_{file_stem}.joblib"
        )
        out["keeper"][display] = joblib.load(
            ROOT / f"models/keeper_{file_stem}.joblib"
        )
    return out


@st.cache_data
def _load_results():
    s = pd.read_csv(ROOT / "models/shooter_results.csv")
    k = pd.read_csv(ROOT / "models/keeper_results.csv")
    return s, k


df, shooters, gks = _load_data()
models = _load_models()
shooter_res, keeper_res = _load_results()
SHOOTER_TEAMS = sorted(shooters["Team"].dropna().unique().tolist())
SHOOTER_NAMES = shooters["Player"].tolist()
GK_NAMES = gks["GK"].tolist()
GLOBAL_SCORE_RATE = float(df["Scored"].mean())


# ---------------------------------------------------------------------------
# Feature builder for a single user-selected scenario
# ---------------------------------------------------------------------------
def _build_row(shooter, gk, foot, venue, minute, condition, gameweek, team, angle):
    s_row = shooters.loc[shooters["Player"] == shooter].iloc[0]
    g_row = gks.loc[gks["GK"] == gk].iloc[0]
    row = {
        "Minute": minute,
        "Gameweek": gameweek,
        "GK_Height": float(g_row["Height"]),
        "ShooterConvRate": float(s_row["ConvRate"]),
        "GKSaveRate": float(g_row["SaveRate"]),
        "Foot": foot,
        "Venue": venue,
        "Condition": condition,
        "Continent": s_row["Continent"],
        "TeamCategory": _team_category(team),
        "Team": team,
    }
    if angle is not None:
        row["Angle"] = angle
    return pd.DataFrame([row])


def _team_category(team):
    cat = df.loc[df["Team"] == team, "TeamCategory"]
    if len(cat):
        return cat.mode().iat[0]
    return "Average"


# ---------------------------------------------------------------------------
# Sidebar — shared inputs
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Match scenario")

team = st.sidebar.selectbox(
    "Shooter's team", SHOOTER_TEAMS,
    index=SHOOTER_TEAMS.index("Manchester City") if "Manchester City" in SHOOTER_TEAMS else 0,
)
try:
    st.sidebar.image(str(get_logo_path(team)), width=96)
except Exception:
    pass

team_shooters = shooters.loc[shooters["Team"] == team, "Player"].tolist()
shooter_pool = team_shooters if team_shooters else SHOOTER_NAMES
shooter = st.sidebar.selectbox("Shooter", shooter_pool)

gk = st.sidebar.selectbox("Goalkeeper", GK_NAMES)
foot = st.sidebar.radio("Shooter's foot", ["Right-Footed", "Left-Footed"], horizontal=True)
venue = st.sidebar.radio("Venue", ["Home", "Away"], horizontal=True)
condition = st.sidebar.selectbox("Match state", ["Drawing", "Winning", "Losing"])
minute = st.sidebar.slider("Minute", 1, 95, 60)
gameweek = st.sidebar.slider("Gameweek", 1, 38, 20)
algo = st.sidebar.selectbox(
    "Algorithm", ["SVM", "LogReg", "MLP"], index=0,
    help="SVM has the best shooter ROC-AUC; MLP has the best Top-2 keeper accuracy.",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**Shooter conv rate:** {shooters.loc[shooters['Player'] == shooter, 'ConvRate'].iat[0]:.0%}  \n"
    f"**GK save rate:** {gks.loc[gks['GK'] == gk, 'SaveRate'].iat[0]:.0%}"
)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
hcol1, hcol2 = st.columns([1, 7])
with hcol1:
    try:
        st.image(str(get_logo_path(team)), width=84)
    except Exception:
        pass
with hcol2:
    st.title("EPL Penalty Predictor")
    st.caption(
        f"**{shooter}** ({team}) vs **{gk}**  •  {venue}, minute {minute}, "
        f"gameweek {gameweek}, {condition.lower()}  •  algorithm: **{algo}**"
    )

tab_shooter, tab_keeper, tab_results = st.tabs(
    ["🎯 Shooter view", "🧤 Keeper view", "📊 Model comparison"]
)


# ---------------------------------------------------------------------------
# Shooter view
# ---------------------------------------------------------------------------
with tab_shooter:
    st.subheader("Pick a corner — see your scoring chance")
    st.caption(
        "Use the six buttons under the goal to fire at a corner. The selected "
        "algorithm returns the probability of scoring at that corner against "
        "this goalkeeper, and the chosen corner glows yellow on the goal."
    )

    if "shoot_angle" not in st.session_state:
        st.session_state.shoot_angle = "Right Bottom"
    chosen = st.session_state.shoot_angle

    X_row = _build_row(shooter, gk, foot, venue, minute, condition, gameweek, team, chosen)
    proba = float(models["shooter"][algo].predict_proba(X_row)[0, 1])

    # Visual SVG with the chosen corner highlighted
    probs_dict = {a: (proba if a == chosen else 0.0) for a in ANGLES}
    render_svg_goal(probs=probs_dict, highlight=chosen, show_labels=False)

    # Click row — six buttons below the goal-mouth
    click = corner_buttons(prefix="shoot")
    if click and click != chosen:
        st.session_state.shoot_angle = click
        st.rerun()

    c1, c2, c3 = st.columns(3)
    c1.metric("P(Scored)", f"{proba:.1%}")
    c2.metric("Aimed at", chosen)
    c3.metric("Algorithm", algo)
    st.progress(min(max(proba, 0.0), 1.0), text=f"{proba:.1%} scoring chance")

    if proba >= 0.6:
        st.success("**Good odds** — fire it!")
    elif proba >= 0.4:
        st.info("**Coin-flip territory.** Confidence matters.")
    else:
        st.warning("**Tough corner** against this keeper.")


# ---------------------------------------------------------------------------
# Keeper view
# ---------------------------------------------------------------------------
with tab_keeper:
    st.subheader("Read the shooter — which corner is the model betting on?")
    st.caption(
        "The model's predicted heatmap is drawn on the goal-mouth. Use the "
        "buttons below to **dive** — if your dive lands on the model's top "
        "prediction, it's a save."
    )

    X_row_k = _build_row(shooter, gk, foot, venue, minute, condition, gameweek, team, None)
    keeper_model = models["keeper"][algo]
    proba_k = keeper_model.predict_proba(X_row_k)[0]
    classes = list(keeper_model.classes_)
    probs_map = {c: float(p) for c, p in zip(classes, proba_k)}
    predicted = max(probs_map, key=probs_map.get)

    st.session_state.setdefault("dive", None)
    st.session_state.setdefault("rounds", 0)
    st.session_state.setdefault("saves", 0)

    render_svg_goal(
        probs=probs_map,
        highlight=st.session_state.dive,
        actual=predicted if st.session_state.dive else None,
        show_labels=True,
    )

    dive_click = corner_buttons(prefix="dive", probs=probs_map)
    if dive_click:
        st.session_state.dive = dive_click
        st.session_state.rounds += 1
        if dive_click == predicted:
            st.session_state.saves += 1
        st.rerun()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top prediction", predicted)
    c2.metric("Top-1 confidence", f"{probs_map[predicted]:.1%}")
    second = sorted(probs_map.items(), key=lambda x: -x[1])[1]
    c3.metric("Second guess", f"{second[0]} ({second[1]:.0%})")
    c4.metric("Saves / dives", f"{st.session_state.saves} / {st.session_state.rounds}")

    if st.session_state.dive:
        if st.session_state.dive == predicted:
            st.success(f"**SAVED!** You dove to {st.session_state.dive}.")
        else:
            st.error(
                f"**GOAL** — you dove {st.session_state.dive}, "
                f"model predicted {predicted}."
            )

    if st.button("🔄 Reset score", key="reset_score"):
        st.session_state.dive = None
        st.session_state.rounds = 0
        st.session_state.saves = 0
        st.rerun()


# ---------------------------------------------------------------------------
# Model comparison tab
# ---------------------------------------------------------------------------
with tab_results:
    st.subheader("How the three algorithms compare")
    st.caption(
        "80/20 stratified random split, 3-fold GridSearchCV. Imbalance: "
        "class_weight (LogReg/SVM shooter) or SMOTE (MLP shooter, all keeper)."
    )

    combined = ROOT / "report/figures/combined_metrics.png"
    if combined.exists():
        st.image(str(combined), caption="Headline metrics — both tasks side by side")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🎯 Shooter — binary (Scored vs Missed)")
        st.dataframe(
            shooter_res.drop(columns=["best_params"]).style.format({
                "accuracy": "{:.3f}", "roc_auc": "{:.3f}", "f1": "{:.3f}"
            }).highlight_max(axis=0, subset=["accuracy", "roc_auc", "f1"],
                             props="background-color: #C8E6C9;"),
            use_container_width=True, hide_index=True,
        )
        roc_path = ROOT / "report/figures/shooter_roc_overlay.png"
        if roc_path.exists():
            st.image(str(roc_path), caption="ROC curves — shooter task")
    with c2:
        st.markdown("### 🧤 Keeper — 6-class corner")
        st.dataframe(
            keeper_res.drop(columns=["best_params"]).style.format({
                "accuracy": "{:.3f}", "macro_f1": "{:.3f}", "top2_accuracy": "{:.3f}"
            }).highlight_max(axis=0, subset=["accuracy", "macro_f1", "top2_accuracy"],
                             props="background-color: #C8E6C9;"),
            use_container_width=True, hide_index=True,
        )
        bar_path = ROOT / "report/figures/keeper_top2_bar.png"
        if bar_path.exists():
            st.image(str(bar_path), caption="Top-2 accuracy — keeper task")

    st.markdown("### Confusion matrices — keeper task")
    grid = st.columns(3)
    for i, name in enumerate(["logreg", "svm", "mlp"]):
        cm_path = ROOT / f"report/figures/keeper_{name}_cm.png"
        if cm_path.exists():
            grid[i].image(str(cm_path), caption=f"Keeper / {name.upper()}")

    with st.expander("📈 Per-class classification report"):
        for name in ("logreg", "svm", "mlp"):
            rp = ROOT / f"models/keeper_{name}_report.txt"
            if rp.exists():
                st.markdown(f"**Keeper / {name.upper()}**")
                st.code(rp.read_text(), language="text")

    with st.expander("📋 Dataset summary"):
        st.write(f"**Rows used:** {len(df)}")
        st.write(f"**Unique shooters:** {df['Player'].nunique()}")
        st.write(f"**Unique goalkeepers:** {df['GK'].nunique()}")
        st.write(f"**Unique teams:** {df['Team'].nunique()}")
        st.write(f"**Class balance (Scored):** {GLOBAL_SCORE_RATE:.1%}")
