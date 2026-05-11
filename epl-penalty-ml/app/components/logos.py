"""EPL club logo lookup.

Strategy:
  1. Try to fetch the main image of the team's Wikipedia page once,
     cache as PNG under app/assets/logos/.
  2. If that fails (offline, blocked, ambiguous), generate a coloured
     badge with the club initials using Pillow as a graceful fallback.

The first time the app runs it downloads ~28 small images; afterwards it
reads from disk.
"""
from __future__ import annotations

import re
import time
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image, ImageDraw, ImageFont

LOGOS_DIR = Path(__file__).resolve().parents[1] / "assets" / "logos"
LOGOS_DIR.mkdir(parents=True, exist_ok=True)

# Wikipedia page title overrides where the bare team name is ambiguous.
WIKI_TITLE = {
    "Manchester City": "Manchester City F.C.",
    "Manchester United": "Manchester United F.C.",
    "Chelsea": "Chelsea F.C.",
    "Liverpool": "Liverpool F.C.",
    "Arsenal": "Arsenal F.C.",
    "Tottenham": "Tottenham Hotspur F.C.",
    "Brighton": "Brighton & Hove Albion F.C.",
    "Leicester City": "Leicester City F.C.",
    "Crystal Palace": "Crystal Palace F.C.",
    "West Ham": "West Ham United F.C.",
    "Newcastle": "Newcastle United F.C.",
    "Everton": "Everton F.C.",
    "Southampton": "Southampton F.C.",
    "Fulham": "Fulham F.C.",
    "Wolverhampton Wanderers": "Wolverhampton Wanderers F.C.",
    "Aston Villa": "Aston Villa F.C.",
    "Brentford": "Brentford F.C.",
    "Bournemouth": "A.F.C. Bournemouth",
    "Burnley": "Burnley F.C.",
    "Leeds United": "Leeds United F.C.",
    "Watford": "Watford F.C.",
    "Sheffield United": "Sheffield United F.C.",
    "Nottingham Forest": "Nottingham Forest F.C.",
    "Luton Town": "Luton Town F.C.",
    "Norwich City": "Norwich City F.C.",
    "Cardiff City": "Cardiff City F.C.",
    "West Bromwich Albion": "West Bromwich Albion F.C.",
    "Huddersfield Town": "Huddersfield Town A.F.C.",
}

# Club primary colors for the fallback badge.
CLUB_COLOR = {
    "Manchester City": "#6CABDD",
    "Manchester United": "#DA291C",
    "Chelsea": "#034694",
    "Liverpool": "#C8102E",
    "Arsenal": "#EF0107",
    "Tottenham": "#132257",
    "Brighton": "#0057B8",
    "Leicester City": "#003090",
    "Crystal Palace": "#1B458F",
    "West Ham": "#7A263A",
    "Newcastle": "#241F20",
    "Everton": "#003399",
    "Southampton": "#D71920",
    "Fulham": "#000000",
    "Wolverhampton Wanderers": "#FDB913",
    "Aston Villa": "#670E36",
    "Brentford": "#E30613",
    "Bournemouth": "#DA291C",
    "Burnley": "#6C1D45",
    "Leeds United": "#FFCD00",
    "Watford": "#FBEE23",
    "Sheffield United": "#EE2737",
    "Nottingham Forest": "#DD0000",
    "Luton Town": "#F78F1E",
    "Norwich City": "#FFF200",
    "Cardiff City": "#0070B5",
    "West Bromwich Albion": "#122F67",
    "Huddersfield Town": "#0E63AD",
}

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "epl-penalty-ml/1.0 (educational project)"}


def _safe_name(team: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", team.lower()).strip("_")


def _fetch_wikipedia_image(title: str, attempts: int = 3) -> bytes | None:
    """Fetch the club crest from a Wikipedia article.

    Strategy:
      1. Parse the article HTML for the og:image meta tag (set by Wikipedia
         for the main infobox image — this is the club crest on team pages).
      2. Download that image as bytes.
      3. Retry transient failures up to `attempts` times with backoff so the
         first-run cache fills reliably even when Wikipedia throttles us.

    The older pageimages API returns empty thumbnails for most football club
    pages because no `pageimage` metadata is set on them.
    """
    page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    for attempt in range(attempts):
        try:
            r = requests.get(page_url, headers=HEADERS, timeout=12)
            r.raise_for_status()
            m = re.search(
                r'<meta\s+property="og:image"\s+content="([^"]+)"',
                r.text,
                re.IGNORECASE,
            )
            if not m:
                # No og:image — not a transient failure, no point retrying.
                return None
            img_url = m.group(1)
            img = requests.get(img_url, headers=HEADERS, timeout=15)
            img.raise_for_status()
            if len(img.content) < 1500:
                # Suspiciously small payload — try again.
                raise ValueError("payload too small")
            return img.content
        except Exception:
            if attempt < attempts - 1:
                # Polite back-off: 0.6s, 1.8s, ...
                time.sleep(0.6 * (3 ** attempt))
    return None


def _make_badge(team: str, color_hex: str) -> bytes:
    """Generate a circular initials badge as PNG bytes."""
    size = 256
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    color = tuple(int(color_hex.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    draw.ellipse([(8, 8), (size - 8, size - 8)], fill=color + (255,),
                 outline=(255, 255, 255, 255), width=4)
    initials = "".join(w[0] for w in team.split() if w)[:3].upper()
    try:
        font = ImageFont.truetype("arial.ttf", 90)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), initials, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((size - tw) / 2, (size - th) / 2 - bbox[1]), initials,
              fill=(255, 255, 255, 255), font=font)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _is_fallback_badge(path: Path) -> bool:
    """A cached file is a placeholder badge if it is exactly 256x256 px."""
    if not path.exists():
        return True
    try:
        with Image.open(path) as img:
            return img.size == (256, 256) and path.stat().st_size < 8000
    except Exception:
        return True


def get_logo_path(team: str, replace_fallback: bool = False) -> Path:
    """Return a local PNG path for the team, downloading or generating if needed.

    If ``replace_fallback`` is True and the cached file is a placeholder badge,
    the Wikipedia fetch is retried (useful for batch re-runs).
    """
    path = LOGOS_DIR / f"{_safe_name(team)}.png"
    if path.exists() and path.stat().st_size > 0:
        if not (replace_fallback and _is_fallback_badge(path)):
            return path

    title = WIKI_TITLE.get(team, team + " F.C.")
    data = _fetch_wikipedia_image(title)
    if data is None:
        # Only write a fresh fallback if we don't already have one to keep.
        if path.exists():
            return path
        color = CLUB_COLOR.get(team, "#2E2E2E")
        data = _make_badge(team, color)
    path.write_bytes(data)
    return path


def ensure_all(teams: list[str], replace_fallback: bool = True,
               request_delay: float = 0.3) -> dict[str, Path]:
    """Download/generate logos for all teams and return mapping."""
    out = {}
    for t in teams:
        out[t] = get_logo_path(t, replace_fallback=replace_fallback)
        time.sleep(request_delay)
    return out


if __name__ == "__main__":
    # Pre-warm the cache for every club we know about.
    paths = ensure_all(list(WIKI_TITLE.keys()))
    print(f"Cached {len(paths)} logos in {LOGOS_DIR}")
    for t, p in paths.items():
        print(f"  {t:30s} -> {p.name} ({p.stat().st_size // 1024} KB)")
