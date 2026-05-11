"""SVG goal-mouth + Streamlit-button click grid.

The goal-mouth is rendered as inline SVG inside a `components.v1.html` iframe so
the browser actually renders it (Streamlit's markdown sanitizer strips `<svg>`).

Click handling cannot navigate the parent page from a sandboxed iframe, so the
six corners are exposed as a Streamlit-button grid below the SVG. Each button
matches the heatmap zone above it.
"""
from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

ANGLES = [
    "Left Top", "Middle Top", "Right Top",
    "Left Bottom", "Middle Bottom", "Right Bottom",
]


def _zone_color(prob: float, max_prob: float) -> str:
    """Heatmap color: cool teal (low) → warm red (high)."""
    if max_prob <= 0:
        return "rgba(40, 90, 140, 0.18)"
    t = min(1.0, prob / max_prob)
    cold = (66, 145, 199)
    hot = (215, 71, 47)
    r = int(cold[0] + (hot[0] - cold[0]) * t)
    g = int(cold[1] + (hot[1] - cold[1]) * t)
    b = int(cold[2] + (hot[2] - cold[2]) * t)
    a = 0.40 + 0.55 * t
    return f"rgba({r}, {g}, {b}, {a:.2f})"


def _build_svg(probs: dict[str, float], highlight: str | None,
               actual: str | None, show_labels: bool) -> str:
    max_p = max(probs.values()) if probs else 0.0

    W, H = 820, 520
    GOAL_X, GOAL_Y, GOAL_W, GOAL_H = 70, 60, 680, 340
    POST_W = 14
    INNER_X = GOAL_X + POST_W
    INNER_Y = GOAL_Y + POST_W
    INNER_W = GOAL_W - 2 * POST_W
    INNER_H = GOAL_H - POST_W
    ZONE_W = INNER_W / 3
    ZONE_H = INNER_H / 2

    zones_svg: list[str] = []
    for r, row in enumerate((("Left Top", "Middle Top", "Right Top"),
                              ("Left Bottom", "Middle Bottom", "Right Bottom"))):
        for c, angle in enumerate(row):
            x = INNER_X + c * ZONE_W
            y = INNER_Y + r * ZONE_H
            p = probs.get(angle, 0.0)
            fill = _zone_color(p, max_p)
            stroke, sw = (
                ("#FFC400", 5) if angle == highlight
                else ("rgba(255,255,255,0.30)", 1)
            )
            cx = x + ZONE_W / 2
            cy = y + ZONE_H / 2

            label_svg = ""
            if show_labels:
                label_svg = (
                    f'<text x="{cx:.1f}" y="{cy - 4:.1f}" text-anchor="middle" '
                    f'fill="#ffffff" font-weight="800" font-size="36" '
                    f'style="paint-order:stroke;stroke:#000;stroke-width:3px;'
                    f'stroke-opacity:0.35;">{p * 100:.0f}%</text>'
                    f'<text x="{cx:.1f}" y="{cy + 22:.1f}" text-anchor="middle" '
                    f'fill="rgba(255,255,255,0.92)" font-size="13" '
                    f'font-weight="600" letter-spacing="1" '
                    f'style="text-transform:uppercase;">{angle}</text>'
                )

            ball_svg = ""
            if angle == actual:
                bx = x + ZONE_W - 32
                by = y + 30
                ball_svg = (
                    f'<g><circle cx="{bx}" cy="{by}" r="14" fill="#ffffff" '
                    f'stroke="#212121" stroke-width="2"/>'
                    f'<path d="M{bx - 6} {by - 8} L{bx + 6} {by - 8} '
                    f'L{bx + 9} {by} L{bx + 4} {by + 8} L{bx - 4} {by + 8} '
                    f'L{bx - 9} {by} Z" fill="#212121" opacity="0.85"/></g>'
                )

            zones_svg.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{ZONE_W:.1f}" '
                f'height="{ZONE_H:.1f}" fill="{fill}" stroke="{stroke}" '
                f'stroke-width="{sw}" rx="4"/>'
                f'{label_svg}{ball_svg}'
            )

    posts_svg = (
        f'<rect x="{GOAL_X}" y="{GOAL_Y}" width="{POST_W}" '
        f'height="{GOAL_H}" fill="#ffffff" rx="2"/>'
        f'<rect x="{GOAL_X + GOAL_W - POST_W}" y="{GOAL_Y}" width="{POST_W}" '
        f'height="{GOAL_H}" fill="#ffffff" rx="2"/>'
        f'<rect x="{GOAL_X}" y="{GOAL_Y}" width="{GOAL_W}" '
        f'height="{POST_W}" fill="#ffffff" rx="2"/>'
    )

    return f"""
<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg"
     preserveAspectRatio="xMidYMid meet"
     style="width:100%;height:100%;display:block;">
  <defs>
    <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#cce8ff"/>
      <stop offset="100%" stop-color="#eaf5ff"/>
    </linearGradient>
    <linearGradient id="grass" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#7CB342"/>
      <stop offset="100%" stop-color="#558B2F"/>
    </linearGradient>
    <linearGradient id="netBg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#1d2c44"/>
      <stop offset="100%" stop-color="#2c4366"/>
    </linearGradient>
    <pattern id="netLines" x="0" y="0" width="22" height="22"
             patternUnits="userSpaceOnUse">
      <path d="M0,0 L22,22 M22,0 L0,22"
            stroke="rgba(255,255,255,0.18)" stroke-width="0.8"/>
    </pattern>
    <filter id="postShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="3" stdDeviation="3" flood-opacity="0.30"/>
    </filter>
  </defs>

  <rect x="0" y="0" width="{W}" height="410" fill="url(#sky)"/>
  <rect x="0" y="410" width="{W}" height="{H - 410}" fill="url(#grass)"/>
  <rect x="0" y="410" width="{W}" height="14" fill="rgba(255,255,255,0.06)"/>
  <rect x="0" y="450" width="{W}" height="14" fill="rgba(255,255,255,0.06)"/>

  <rect x="{INNER_X}" y="{INNER_Y}" width="{INNER_W}" height="{INNER_H}"
        fill="url(#netBg)" rx="2"/>
  <rect x="{INNER_X}" y="{INNER_Y}" width="{INNER_W}" height="{INNER_H}"
        fill="url(#netLines)" rx="2"/>

  {''.join(zones_svg)}

  <g filter="url(#postShadow)">{posts_svg}</g>

  <circle cx="{W / 2}" cy="475" r="6" fill="#ffffff" stroke="#212121"
          stroke-width="1.5"/>
  <text x="{W / 2}" y="500" text-anchor="middle" font-size="13"
        fill="#1a3a1a" font-weight="700" letter-spacing="2"
        style="text-transform:uppercase;">Penalty Spot</text>
</svg>
""".strip()


def render_svg_goal(probs: dict[str, float] | None = None,
                    highlight: str | None = None,
                    actual: str | None = None,
                    show_labels: bool = True,
                    height: int = 520) -> None:
    """Render the visual SVG goal-mouth (no clicks)."""
    if probs is None:
        probs = {a: 0.0 for a in ANGLES}
    svg = _build_svg(probs, highlight, actual, show_labels)
    html = f"""<!doctype html>
<html><head><style>
  html, body {{ margin:0; padding:0; background:transparent; height:100%; }}
  .frame {{ width:100%; height:100%; }}
</style></head>
<body><div class="frame">{svg}</div></body></html>"""
    components.html(html, height=height, scrolling=False)


def corner_buttons(prefix: str = "corner",
                   probs: dict[str, float] | None = None) -> str | None:
    """Render six labelled buttons in a 3×2 grid; return clicked corner or None.

    When `probs` is given, each button also shows its predicted probability.
    """
    if probs is None:
        probs = {}
    clicked = None
    rows = (("Left Top", "Middle Top", "Right Top"),
            ("Left Bottom", "Middle Bottom", "Right Bottom"))
    for row in rows:
        cols = st.columns(3, gap="small")
        for col, angle in zip(cols, row):
            p = probs.get(angle)
            label = (
                f"⬆ {angle}" if "Top" in angle else f"⬇ {angle}"
            )
            if p is not None:
                label = f"{label}\n{p * 100:.0f}%"
            if col.button(
                label,
                key=f"{prefix}_{angle}",
                use_container_width=True,
            ):
                clicked = angle
    return clicked
