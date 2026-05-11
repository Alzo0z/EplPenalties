import sys
sys.path.insert(0, ".")
from app.components.logos import WIKI_TITLE, _fetch_wikipedia_image

for team, title in WIKI_TITLE.items():
    data = _fetch_wikipedia_image(title)
    if data is None:
        print(f"FAIL  {team:30s}  title={title}")
    else:
        print(f"OK    {team:30s}  title={title}  ({len(data) // 1024} KB)")
