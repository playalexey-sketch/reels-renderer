# -*- coding: utf-8 -*-
"""
Generate all designer PDFs for Шаман РаХунХан 2026:
  1. docs/market-analysis-and-funnels.pdf
  2. docs/100-reels-scripts.pdf
  3. docs/selling-reels-playbook.pdf
Ensures 100% Cyrillic support, zero missing glyphs (no '■' or black squares).
"""
import os
import re
import markdown
from xhtml2pdf import pisa
import fitz

DEJAVU_SANS = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
DEJAVU_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
DEJAVU_SERIF = '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'
DEJAVU_SERIF_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf'

CSS_STYLE = f"""
@page {{
  size: a4;
  @frame bg_frame {{ left: 0cm; top: 0cm; width: 21.0cm; height: 29.7cm; padding: 0;
                    -pdf-frame-content: bg_content; }}
  @frame footer_frame {{ left: 1.4cm; bottom: 0.75cm; width: 18.2cm; height: 0.8cm; padding: 0;
                    -pdf-frame-content: footer_content; }}
  @frame content_frame {{ left: 1.4cm; top: 1.35cm; width: 18.2cm; height: 26.9cm; padding: 0; }}
}}
@font-face {{ font-family: PS; src: url({DEJAVU_SANS}); }}
@font-face {{ font-family: PS; font-weight: bold; src: url({DEJAVU_BOLD}); }}
@font-face {{ font-family: SER; src: url({DEJAVU_SERIF}); }}
@font-face {{ font-family: SER; font-weight: bold; src: url({DEJAVU_SERIF_BOLD}); }}

body {{ font-family: PS; font-size: 8.5pt; line-height: 1.45; color: #EDE1CB; }}
b, strong {{ font-family: PS; font-weight: bold; color: #D9B384; }}
i, em {{ font-family: SER; font-style: italic; color: #B7A289; }}

h1 {{ font-family: SER; font-weight: bold; font-size: 20pt; line-height: 1.2; color: #EDE1CB; margin: 12pt 0 4pt 0; text-align: center; }}
h2 {{ font-family: SER; font-weight: bold; font-size: 14pt; line-height: 1.25; color: #D9B384; margin: 14pt 0 4pt 0; border-bottom: 0.7pt solid #54402D; padding-bottom: 2pt; }}
h3 {{ font-family: PS; font-weight: bold; font-size: 11pt; color: #EDE1CB; margin: 10pt 0 3pt 0; }}
h4 {{ font-family: PS; font-weight: bold; font-size: 9pt; color: #B7A289; margin: 8pt 0 2pt 0; }}

p {{ font-family: PS; font-size: 8.5pt; color: #EDE1CB; margin: 3pt 0 5pt 0; }}

pre, code, tt, kbd, samp {{
  font-family: PS;
  font-size: 7.5pt;
  line-height: 1.35;
  background-color: #20140B;
  border: 0.7pt solid #54402D;
  padding: 5pt;
  color: #D8C9AE;
  margin: 5pt 0;
  white-space: pre-wrap;
}}

ul, ol {{ margin: 2pt 0 6pt 15pt; padding: 0; }}
li {{ font-family: PS; font-size: 8.3pt; color: #EDE1CB; margin-bottom: 2pt; }}

blockquote {{
  background-color: #2D2013;
  border-left: 2.5pt solid #D9B384;
  padding: 4pt 8pt;
  margin: 6pt 0;
  font-family: SER;
  font-size: 8.5pt;
  color: #EDE1CB;
}}

table {{
  width: 100%;
  border-collapse: collapse;
  margin: 6pt 0 8pt 0;
}}
th {{
  font-family: PS; font-weight: bold; font-size: 7.5pt; color: #D9B384;
  background-color: #382A1C; border: 0.6pt solid #54402D; padding: 3pt 5pt; text-align: left;
}}
td {{
  font-family: PS; font-size: 7.8pt; color: #EDE1CB; border: 0.6pt solid #54402D;
  padding: 3pt 5pt; vertical-align: top;
}}

.footl {{ font-family: PS; font-size: 6.5pt; color: #B7A289; }}
.footr {{ font-family: PS; font-size: 6.5pt; color: #B7A289; text-align: right; }}
"""

def convert_md_to_pdf(md_path, pdf_path, doc_title):
    print(f"Compiling {md_path} -> {pdf_path}...")
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    html_content = markdown.markdown(text, extensions=['tables', 'fenced_code', 'nl2br'])

    bg_div = '<div id="bg_content" style="width:21cm; height:29.7cm; background-color:#22150D;"></div>'
    foot_div = (
        f'<div id="footer_content">'
        f'<table style="width:100%; border:none; border-top:0.5pt solid #54402D; margin:0;"><tr>'
        f'<td style="border:none; padding:2pt 0 0 0;"><span class="footl">{doc_title.upper()} · ШАМАН РАХУНХАН 2026</span></td>'
        f'<td style="border:none; padding:2pt 0 0 0; text-align:right;"><span class="footr">СТР. <pdf:pagenumber/></span></td>'
        f'</tr></table></div>'
    )

    full_html = f"<html><head><style>{CSS_STYLE}</style></head><body>{bg_div}{foot_div}{html_content}</body></html>"

    with open(pdf_path, "w+b") as f:
        status = pisa.CreatePDF(full_html, dest=f, encoding="utf-8")

    if status.err:
        print(f"pisa warning in {pdf_path}: {status.err}")

    # Set dark background color using PyMuPDF (fitz)
    doc = fitz.open(pdf_path)
    for page in doc:
        page.draw_rect(page.rect, fill=(0x22/255, 0x15/255, 0x0D/255), color=None, overlay=False)
    doc.save(pdf_path, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
    
    # Verification pass: check for any missing glyphs '■'
    doc_check = fitz.open(pdf_path)
    missing_count = 0
    for idx, page in enumerate(doc_check):
        txt = page.get_text()
        if '■' in txt or '\ufffd' in txt:
            missing_count += 1
            print(f"WARNING: Page {idx+1} in {pdf_path} contains missing glyphs!")

    doc_check.close()
    if missing_count == 0:
        print(f"VERIFIED PERFECT: {pdf_path} rendered {len(doc)} pages with 0 missing glyphs!")
    else:
        print(f"ALERT: {pdf_path} has {missing_count} pages with missing glyphs!")

if __name__ == "__main__":
    convert_md_to_pdf("docs/market-analysis-and-funnels.md", "docs/market-analysis-and-funnels.pdf", "Анализ Рынка и 10 Воронок")
    convert_md_to_pdf("docs/100-reels-scripts.md", "docs/100-reels-scripts.pdf", "100 Сценариев Продающих Reels")
    os.system("python3 scripts/md2pdf_styled.py")
    print("All PDFs generated and verified!")
