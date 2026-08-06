# -*- coding: utf-8 -*-
"""Styled MD->PDF for docs in the storyboard design (reuses design from scripts/md2pdf_styled.py)."""
import os, re, sys
import markdown
from xhtml2pdf import pisa
import fitz

# ---- reuse design system: exec everything in md2pdf_styled.py before 'bg_div =' ----
_src = open("scripts/md2pdf_styled.py", encoding="utf-8").read()
_prefix = _src[:_src.index("bg_div =")]
ns = {}
exec(compile(_prefix, "md2pdf_styled_prefix", "exec"), ns)
CSS = ns["CSS"]; L = ns["L"]
BG = ns["BG"]; CARD = ns["CARD"]; CARD2 = ns["CARD2"]; LINE = ns["LINE"]
TEXT = ns["TEXT"]; MUTED = ns["MUTED"]; GOLD = ns["GOLD"]

# ---- extra CSS for generic markdown documents ----
DOC_CSS = CSS + f"""
h1 {{ font-family: Y; font-size: 19pt; color: {TEXT}; margin: 2pt 0 1pt 0; line-height: 1.18; }}
h2 {{ font-family: Y; font-size: 13.5pt; color: {TEXT}; border-bottom: 0.7pt solid {LINE};
     padding-bottom: 2pt; margin: 13pt 0 5pt 0; }}
h3 {{ font-family: CAPS; font-weight: bold; font-size: 9.2pt; color: {GOLD}; margin: 10pt 0 3pt 0; }}
h4 {{ font-family: CAPS; font-weight: bold; font-size: 8pt; color: {TEXT}; margin: 8pt 0 2pt 0; }}
p {{ margin: 3pt 0; }}
ul, ol {{ margin: 3pt 0 3pt 14pt; }}
li {{ margin: 1.5pt 0; font-size: 8.6pt; color: {TEXT}; }}
table {{ width: 100%; border-collapse: collapse; margin: 4pt 0 9pt 0; }}
th {{ font-family: CAPS; font-weight: bold; font-size: 7.2pt; color: {TEXT};
     background-color: #382A1C; border: 0.6pt solid {LINE}; padding: 3.5pt 5pt; text-align: left; }}
td {{ font-family: PS; font-size: 7.9pt; color: {TEXT}; border: 0.6pt solid {LINE};
     padding: 3.5pt 5pt; vertical-align: top; line-height: 1.4; }}
pre {{ font-family: MONO; font-size: 6.6pt; line-height: 1.4; color: #D8C9AE;
      background-color: {CARD2}; border: 0.7pt solid {LINE}; padding: 8pt 9pt;
      white-space: pre; margin: 5pt 0 9pt 0; }}
code {{ font-family: MONO; font-size: 7.2pt; color: #E5C993; }}
blockquote {{ background-color: {CARD2}; border-left: 2.2pt solid {GOLD};
             margin: 6pt 0; padding: 5pt 9pt; }}
blockquote p {{ font-family: SERI; font-size: 9.5pt; color: {TEXT}; line-height: 1.5; }}
hr {{ border: none; border-top: 0.7pt solid {LINE}; margin: 9pt 0; }}
strong {{ color: {TEXT}; }}
a {{ color: {GOLD}; }}
"""

def build(md_path, out_pdf, kicker):
    md = open(md_path, encoding="utf-8").read().replace("\ufe0f", "")
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    bg_div = f'<div id="bg_content" style="width:21cm; height:29.7cm; background-color:{BG};"></div>'
    foot = (f'<div id="footer_content">'
            f'<table style="width:100%; border:none; border-top:0.5pt solid {LINE}; margin:0;"><tr>'
            f'<td style="border:none; padding:2pt 0 0 0;"><span class="footl">{L(kicker + " · стандарт", short=True)}</span></td>'
            f'<td style="border:none; padding:2pt 0 0 0; text-align:right;"><span class="footr">{L("стр.", short=True)} <pdf:pagenumber/></span></td>'
            f'</tr></table></div>')
    body_html = f'<div class="kicker">{L(kicker, short=True)}</div>{body}'
    body_html = (body_html.replace("&#8592;", '<span class="arr">&#8592;</span>')
                          .replace("&#8594;", '<span class="arr">&#8594;</span>')
                          .replace("\u2192", '<span class="arr">\u2192</span>')
                          .replace("\u2190", '<span class="arr">\u2190</span>'))
    html = f"<html><head><style>{DOC_CSS}</style></head><body>{bg_div}{foot}{body_html}</body></html>"
    with open(out_pdf, "w+b") as f:
        status = pisa.CreatePDF(html, dest=f, encoding="utf-8")
    print(os.path.basename(out_pdf), "| pisa errors:", status.err)
    doc = fitz.open(out_pdf)
    for page in doc:
        page.draw_rect(page.rect, fill=(0x22/255, 0x15/255, 0x0D/255), color=None, overlay=False)
    doc.save(out_pdf, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
    doc.close()
    doc = fitz.open(out_pdf)
    print("  pages:", len(doc))
    return len(doc)

os.makedirs("/tmp/prev2", exist_ok=True)
JOBS = [
    ("examples/shaman-rod/reels-all-concepts.md", "examples/shaman-rod/reels-all-concepts.pdf",
     "reels · 15 сценариев · шаманская ниша"),
    ("examples/shaman-rod/prompt-pack.md", "examples/shaman-rod/prompt-pack.pdf",
     "пак заданий генератору · шаманская ниша"),
]
for i, (mdp, pdf, k) in enumerate(JOBS):
    n = build(mdp, pdf, k)
    d = fitz.open(pdf)
    for pg in range(min(3, n)):
        d[pg].get_pixmap(dpi=88).save(f"/tmp/prev2/doc{i+1}_p{pg+1}.png")
print("done:", sorted(os.listdir("/tmp/prev2")))
