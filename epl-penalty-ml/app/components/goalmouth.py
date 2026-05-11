"""HTML goal-mouth renderer for Streamlit.

We render the goal as an HTML/CSS grid of 6 zones because Streamlit doesn't
expose per-button background styling reliably. Interaction happens via separate
st.button rows below the visual.
"""
from __future__ import annotations

import streamlit as st

ANGLES = [
    "Left Top", "Middle Top", "Right Top",
    "Left Bottom", "Middle Bottom", "Right Bottom",
]


def _color_for(prob: float, max_prob: float) -> str:
    """Map probability to an RGBA heat color (cool-to-warm)."""
    if max_prob <= 0:
        return "rgba(40, 40, 40, 0.0)"
    t = min(1.0, prob / max_prob)
    r = int(255 * t)
    g = int(60 * (1 - t) + 40)
    b = int(140 * (1 - t))
    a = 0.25 + 0.7 * t
    return f"rgba({r}, {g}, {b}, {a:.2f})"


def render_goal(probs: dict[str, float] | None = None,
                highlight: str | None = None,
                actual: str | None = None,
                show_labels: bool = True) -> None:
    """Render a goal with optional heatmap of probabilities.

    Args:
        probs: angle -> probability (0-1). If None, neutral colors.
        highlight: angle to outline (e.g., user's chosen corner).
        actual:  angle to mark with a small ball icon (true shot location).
        show_labels: whether to render '63%' style labels in each zone.
    """
    if probs is None:
        probs = {a: 0.0 for a in ANGLES}
    max_p = max(probs.values()) if probs else 0.0

    cells_html = []
    for a in ANGLES:
        p = probs.get(a, 0.0)
        bg = _color_for(p, max_p)
        outline = "4px solid #FFD400" if a == highlight else "1px solid #333"
        ball_icon = "<div style='font-size:32px;'>&#9917;&#65039;</div>" if a == actual else ""
        label = f"<div style='font-weight:600;font-size:13px;color:#fff;text-shadow:0 1px 2px #000;'>{a}</div>"
        prob_txt = (
            f"<div style='font-size:22px;font-weight:700;color:#fff;text-shadow:0 1px 3px #000;'>"
            f"{p * 100:.0f}%</div>"
            if show_labels else ""
        )
        cells_html.append(
            f"<div style='background:{bg};border:{outline};display:flex;"
            f"flex-direction:column;align-items:center;justify-content:center;"
            f"height:120px;border-radius:6px;'>"
            f"{label}{prob_txt}{ball_icon}</div>"
        )

    grid = (
        "<div style='display:grid;grid-template-columns:repeat(3, 1fr);"
        "gap:6px;padding:14px;background:linear-gradient(180deg, #1B5E20 0%, #2E7D32 100%);"
        "border:6px solid #fff;border-radius:10px;'>"
        + "".join(cells_html)
        + "</div>"
        + "<div style='height:8px;background:#fff;margin:0 14px;"
          "border-radius:0 0 4px 4px;'></div>"
    )

    st.markdown(grid, unsafe_allow_html=True)


def corner_buttons(prefix: str = "btn", disabled: bool = False) -> str | None:
    """Render 6 corner buttons in a 3-col x 2-row grid. Returns clicked corner or None."""
    clicked = None
    for row_start in (0, 3):
        cols = st.columns(3)
        for i, c in enumerate(cols):
            angle = ANGLES[row_start + i]
            if c.button(angle, key=f"{prefix}_{angle}",
                        use_container_width=True, disabled=disabled):
                clicked = angle
    return clicked
