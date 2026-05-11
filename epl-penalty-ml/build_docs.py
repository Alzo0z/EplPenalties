"""Build self-contained HTML (and optionally PDF) for the report and slides.

Run:
    python build_docs.py

Outputs:
    report/report.html   — single-file report, images embedded as base64
    report/slides.html   — slide deck, prints one slide per page in Chrome
    report/report.pdf    — only if playwright + chromium are installed
    report/slides.pdf    — only if playwright + chromium are installed

If the PDFs aren't generated you can still print either HTML file to PDF
from any browser (File → Print → Save as PDF, choose "More settings" →
"Background graphics" → checked).
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "report"
FIG_DIR = REPORT_DIR / "figures"


# ---------------------------------------------------------------------------
# Embed images as base64 data URIs
# ---------------------------------------------------------------------------
def _data_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _embed_images_md(md_text: str, base_dir: Path) -> str:
    """Replace `![alt](relative/path.png)` with embedded data URIs."""
    def repl(m: re.Match) -> str:
        alt = m.group(1)
        size_attr = m.group(2) or ""  # captures "{ width=480 }" etc — Marp uses ![w:480]
        src = m.group(3).strip()
        img_path = (base_dir / src).resolve()
        if img_path.exists():
            return f'![{alt}{size_attr}]({_data_uri(img_path)})'
        return m.group(0)
    return re.sub(r'!\[([^\]]*?)(\{[^}]*\})?\]\(([^)]+)\)', repl, md_text)


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------
REPORT_CSS = """
@page { size: A4; margin: 18mm 16mm; }
html { box-sizing: border-box; }
*, *:before, *:after { box-sizing: inherit; }
body {
  font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
  color: #1a1a1a;
  background: #ffffff;
  max-width: 880px;
  margin: 40px auto;
  padding: 0 28px 60px;
  line-height: 1.55;
  font-size: 14px;
}
h1, h2, h3, h4 { color: #102A55; line-height: 1.25; margin-top: 1.4em; }
h1 { font-size: 28px; border-bottom: 3px solid #1565C0; padding-bottom: 6px; }
h2 { font-size: 22px; border-bottom: 1px solid #c5cae9; padding-bottom: 4px; }
h3 { font-size: 17px; color: #1565C0; }
h4 { font-size: 14px; color: #1565C0; }
p, ul, ol { margin: 0.7em 0; }
ul, ol { padding-left: 1.4em; }
table { border-collapse: collapse; margin: 1em 0; font-size: 13px; width: 100%; }
th { background: #102A55; color: #ffffff; padding: 8px 10px; text-align: left; }
td { padding: 6px 10px; border-bottom: 1px solid #d6dff0; vertical-align: top; }
tr:nth-child(even) td { background: #f4f7fb; }
code { background: #eef3fb; color: #0d3a76; padding: 1px 5px; border-radius: 3px; font-family: Consolas, monospace; font-size: 12px; }
pre { background: #f6f8fa; border: 1px solid #d6dff0; border-radius: 4px; padding: 10px 14px; overflow-x: auto; font-size: 12px; }
pre code { background: none; padding: 0; }
blockquote { border-left: 4px solid #1565C0; background: #f4f7fb; padding: 8px 14px; margin: 0.8em 0; color: #102A55; }
img { max-width: 100%; height: auto; display: block; margin: 1em auto; box-shadow: 0 2px 10px rgba(16, 42, 85, 0.08); border-radius: 4px; }
hr { border: none; border-top: 1px dashed #c5cae9; margin: 2em 0; }
strong { color: #C62828; }
.cover { text-align: center; padding: 60px 0 40px; border-bottom: 3px solid #1565C0; margin-bottom: 30px; }
.cover h1 { border: none; font-size: 32px; }
@media print {
  body { margin: 0; max-width: none; }
  h1, h2, h3 { page-break-after: avoid; }
  table, img, pre, blockquote { page-break-inside: avoid; }
}
"""


def build_report():
    md_path = REPORT_DIR / "report.md"
    md_text = md_path.read_text(encoding="utf-8")
    md_text = _embed_images_md(md_text, REPORT_DIR)
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>EPL Penalty Predictor — Report</title>
<style>{REPORT_CSS}</style>
</head><body>
{html_body}
</body></html>"""
    out = REPORT_DIR / "report.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB)")
    return out


# ---------------------------------------------------------------------------
# Slides builder — parse Marp markdown into slide-per-page HTML
# ---------------------------------------------------------------------------
SLIDES_CSS = """
@page { size: 13.33in 7.5in; margin: 0; }
html { box-sizing: border-box; }
*, *:before, *:after { box-sizing: inherit; }
body {
  margin: 0;
  font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
  color: #1A2942;
  background: #d6dff0;
}
.slide {
  width: 13.33in;
  height: 7.5in;
  padding: 0.6in 0.85in 0.5in;
  background: linear-gradient(180deg, #ffffff 0%, #f6f9ff 100%);
  position: relative;
  overflow: hidden;
  page-break-after: always;
  margin: 22px auto;
  box-shadow: 0 6px 30px rgba(16, 42, 85, 0.12);
  border-radius: 6px;
}
.slide:last-child { page-break-after: auto; }
.slide.lead, .slide.divider {
  background: linear-gradient(135deg, #102A55 0%, #1565C0 100%);
  color: #ffffff;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
}
.slide.lead h1, .slide.divider h1 { color: #ffffff; border: none; font-size: 60px; margin: 0 0 20px; }
.slide.lead h2, .slide.divider h2 { color: #BBDEFB; border: none; font-size: 28px; margin: 0 0 30px; }
.slide.divider h1 { font-size: 72px; }
h1 {
  color: #102A55; font-size: 36px;
  border-bottom: 4px solid #1565C0;
  padding-bottom: 6px;
  margin: 0 0 18px;
}
h2 { color: #1565C0; font-size: 24px; margin: 0 0 10px; }
h3 { color: #283593; font-size: 18px; margin: 0 0 8px; }
p, li { font-size: 18px; line-height: 1.45; }
ul, ol { padding-left: 22px; margin: 8px 0; }
table { border-collapse: collapse; margin: 10px 0; font-size: 17px; width: 100%; }
th { background: #102A55; color: #ffffff; padding: 7px 11px; text-align: left; }
td { padding: 5px 11px; border-bottom: 1px solid #d6dff0; }
tr:nth-child(even) td { background: #f4f7fb; }
code { background: #eef3fb; color: #0d3a76; padding: 1px 5px; border-radius: 3px; font-size: 16px; }
blockquote { border-left: 5px solid #1565C0; background: #f4f7fb; padding: 8px 14px; margin: 10px 0; color: #102A55; font-size: 17px; }
strong { color: #C62828; }
img { max-width: 100%; max-height: 5.2in; height: auto; display: block; margin: 6px auto; border-radius: 4px; box-shadow: 0 3px 14px rgba(16, 42, 85, 0.12); }
.twocol { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; align-items: start; }
.page-num { position: absolute; bottom: 18px; right: 28px; color: #6c7a91; font-size: 14px; }
@media print {
  body { background: #ffffff; }
  .slide { margin: 0; box-shadow: none; border-radius: 0; }
}
"""


def parse_marp_slides(md_text: str) -> list[tuple[str, list[str]]]:
    """Return list of (class_modifiers, lines) per slide."""
    # Strip the YAML front matter
    if md_text.startswith("---"):
        end = md_text.find("\n---", 3)
        if end != -1:
            md_text = md_text[end + 4:]
    # Split slides on the `---` separator (must be on its own line)
    raw_slides = re.split(r'^---\s*$', md_text, flags=re.MULTILINE)
    slides = []
    for s in raw_slides:
        s = s.strip("\n")
        if not s.strip():
            continue
        # Look for <!-- _class: lead --> or similar directives
        cls_match = re.search(r'<!--\s*_class:\s*([^\s\-]+)\s*-->', s)
        cls = cls_match.group(1) if cls_match else ""
        # Remove HTML comment directives
        s = re.sub(r'<!--.*?-->', '', s, flags=re.DOTALL).strip()
        slides.append((cls, s))
    return slides


def build_slides():
    md_path = REPORT_DIR / "slides.md"
    md_text = md_path.read_text(encoding="utf-8")
    md_text = _embed_images_md(md_text, REPORT_DIR)
    # Strip Marp w:480 sizing hints from image alt text; we use CSS instead
    md_text = re.sub(r'!\[(?:w:\d+|h:\d+)\s*\]', '![]', md_text)
    slides = parse_marp_slides(md_text)

    html_slides = []
    for i, (cls, body) in enumerate(slides, 1):
        body_html = markdown.markdown(
            body,
            extensions=["tables", "fenced_code", "sane_lists", "md_in_html"],
        )
        # Unescape the `<div class="twocol">` blocks (markdown lib HTML-escapes them)
        body_html = body_html.replace('&lt;div class="twocol"&gt;', '<div class="twocol">')
        body_html = body_html.replace('&lt;/div&gt;', '</div>')
        body_html = body_html.replace('&lt;div&gt;', '<div>')
        html_slides.append(
            f'<section class="slide {cls}">'
            f'<div class="page-num">{i}</div>'
            f'{body_html}'
            f'</section>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>EPL Penalty Predictor — Slides</title>
<style>{SLIDES_CSS}</style>
</head><body>
{''.join(html_slides)}
</body></html>"""
    out = REPORT_DIR / "slides.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB, {len(slides)} slides)")
    return out


# ---------------------------------------------------------------------------
# Optional: render HTML to PDF via playwright (headless Chromium)
# ---------------------------------------------------------------------------
def to_pdf(html_path: Path, pdf_path: Path, landscape: bool = False,
           page_format: str = "A4") -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return False
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(f"file:///{html_path.as_posix()}")
            page.emulate_media(media="print")
            pdf_kwargs = {
                "path": str(pdf_path),
                "print_background": True,
                "margin": {"top": "0", "right": "0", "bottom": "0", "left": "0"},
            }
            if page_format:
                pdf_kwargs["format"] = page_format
            if landscape:
                pdf_kwargs["landscape"] = True
            page.pdf(**pdf_kwargs)
            browser.close()
        print(f"Wrote {pdf_path} ({pdf_path.stat().st_size // 1024} KB)")
        return True
    except Exception as e:
        print(f"PDF generation skipped for {pdf_path.name}: {e}")
        return False


def main():
    report_html = build_report()
    slides_html = build_slides()
    print()

    # Try PDF generation; fall back silently if chromium isn't installed yet.
    to_pdf(report_html, REPORT_DIR / "report.pdf", landscape=False, page_format="A4")
    # For slides, custom 13.33×7.5 inch page set via @page CSS — no format hint
    to_pdf(slides_html, REPORT_DIR / "slides.pdf", landscape=True, page_format="")

    print()
    print("Outputs in report/:")
    for f in ("report.html", "slides.html", "report.pdf", "slides.pdf"):
        p = REPORT_DIR / f
        if p.exists():
            print(f"  ✓ {f}  ({p.stat().st_size // 1024} KB)")
        else:
            print(f"  – {f}  (not generated — open the .html and Print → Save as PDF)")


if __name__ == "__main__":
    main()
