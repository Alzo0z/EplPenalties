import requests, re
HEADERS = {"User-Agent": "epl-penalty-ml/1.0"}
for team in ("Liverpool F.C.", "Newcastle United F.C.", "Fulham F.C.",
             "Aston Villa F.C.", "Watford F.C."):
    url = f"https://en.wikipedia.org/wiki/{team.replace(' ', '_')}"
    r = requests.get(url, headers=HEADERS, timeout=8)
    m = re.search(r'<meta\s+property="og:image"\s+content="([^"]+)"', r.text, re.IGNORECASE)
    if not m:
        print(f"{team:30s} -> NO OG IMAGE")
        continue
    img_url = m.group(1)
    img = requests.get(img_url, headers=HEADERS, timeout=8)
    magic = img.content[:8]
    if magic.startswith(b"\x89PNG"):
        kind = "PNG"
    elif b"<svg" in img.content[:200].lower() or magic.startswith(b"<?xml"):
        kind = "SVG"
    else:
        kind = repr(magic)
    print(f"{team:30s} -> ...{img_url[-90:]}  size={len(img.content):>8d}B  {kind}")
