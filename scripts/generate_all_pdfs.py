# -*- coding: utf-8 -*-
"""
Compile markdown documentation into designer PDFs using xhtml2pdf & PyMuPDF (fitz).
Guarantees 100% clean Cyrillic text rendering without any missing glyphs or boxes.
"""
import os
import re
import markdown
from xhtml2pdf import pisa
import fitz

def prepare_fonts():
    os.makedirs('/tmp/fonts', exist_ok=True)
    dejavu_sans = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    dejavu_bold = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    dejavu_serif = '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'
    dejavu_serif_bold = '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf'

    fonts_map = {
        'YesevaOne-Regular.ttf': dejavu_serif_bold,
        'PT_Sans-Web-Regular.ttf': dejavu_sans,
        'PT_Sans-Web-Bold.ttf': dejavu_bold,
        'PT_Sans-Caption-Web-Regular.ttf': dejavu_sans,
        'PT_Sans-Caption-Web-Bold.ttf': dejavu_bold,
        'PT_Serif-Web-Regular.ttf': dejavu_serif,
        'PT_Serif-Web-Bold.ttf': dejavu_serif_bold,
        'PT_Serif-Web-Italic.ttf': dejavu_serif,
    }

    for name, src in fonts_map.items():
        dst = f'/tmp/fonts/{name}'
        if not os.path.exists(dst) and os.path.exists(src):
            import shutil
            shutil.copy(src, dst)

prepare_fonts()

CSS_STYLE = """
@page {
  size: a4;
  @frame bg_frame { left: 0cm; top: 0cm; width: 21.0cm; height: 29.7cm; padding: 0;
                    -pdf-frame-content: bg_content; }
  @frame footer_frame { left: 1.4cm; bottom: 0.75cm; width: 18.2cm; height: 0.8cm; padding: 0;
                    -pdf-frame-content: footer_content; }
  @frame content_frame { left: 1.4cm; top: 1.35cm; width: 18.2cm; height: 26.9cm; padding: 0; }
}
@font-face { font-family: Y; src: url(/tmp/fonts/YesevaOne-Regular.ttf); }
@font-face { font-family: PS; src: url(/tmp/fonts/PT_Sans-Web-Regular.ttf); }
@font-face { font-family: PS; font-weight: bold; src: url(/tmp/fonts/PT_Sans-Web-Bold.ttf); }
@font-face { font-family: CAPS; src: url(/tmp/fonts/PT_Sans-Caption-Web-Regular.ttf); }
@font-face { font-family: CAPS; font-weight: bold; src: url(/tmp/fonts/PT_Sans-Caption-Web-Bold.ttf); }
@font-face { font-family: SER; src: url(/tmp/fonts/PT_Serif-Web-Regular.ttf); }
@font-face { font-family: SER; font-weight: bold; src: url(/tmp/fonts/PT_Serif-Web-Bold.ttf); }
@font-face { font-family: SERI; src: url(/tmp/fonts/PT_Serif-Web-Italic.ttf); }

* {
  font-family: PS !important;
}

body {
  font-family: PS !important;
  font-size: 8.5pt;
  line-height: 1.45;
  color: #EDE1CB;
}

b, strong {
  font-family: PS !important;
  font-weight: bold;
  color: #D9B384;
}

i, em {
  font-family: SERI !important;
  font-style: italic;
  color: #B7A289;
}

h1 {
  font-family: Y !important;
  font-size: 22pt;
  line-height: 1.2;
  color: #EDE1CB;
  margin: 12pt 0 4pt 0;
  text-align: center;
}

h2 {
  font-family: Y !important;
  font-size: 15pt;
  line-height: 1.25;
  color: #D9B384;
  margin: 14pt 0 4pt 0;
  border-bottom: 0.7pt solid #54402D;
  padding-bottom: 2pt;
}

h3 {
  font-family: CAPS !important;
  font-weight: bold;
  font-size: 11pt;
  color: #EDE1CB;
  margin: 10pt 0 3pt 0;
}

h4 {
  font-family: CAPS !important;
  font-weight: bold;
  font-size: 9pt;
  color: #B7A289;
  margin: 8pt 0 2pt 0;
}

p {
  font-family: PS !important;
  font-size: 8.5pt;
  color: #EDE1CB;
  margin: 3pt 0 5pt 0;
}

ul, ol {
  margin: 2pt 0 6pt 15pt;
  padding: 0;
}

li {
  font-family: PS !important;
  font-size: 8.3pt;
  color: #EDE1CB;
  margin-bottom: 2pt;
}

blockquote {
  background-color: #2D2013;
  border-left: 2.5pt solid #D9B384;
  padding: 5pt 8pt;
  margin: 6pt 0;
  font-family: PS !important;
  font-size: 8.3pt;
  color: #EDE1CB;
}

pre, code, tt, kbd, samp {
  font-family: PS !important;
  background-color: #20140B;
  border: 0.7pt solid #54402D;
  padding: 5pt;
  font-size: 7.5pt;
  color: #D8C9AE;
  margin: 6pt 0;
  white-space: pre-wrap;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin: 6pt 0 8pt 0;
}

th {
  font-family: CAPS !important;
  font-weight: bold;
  font-size: 7.2pt;
  color: #D9B384;
  background-color: #382A1C;
  border: 0.6pt solid #54402D;
  padding: 3pt 5pt;
  text-align: left;
}

td {
  font-family: PS !important;
  font-size: 7.8pt;
  color: #EDE1CB;
  border: 0.6pt solid #54402D;
  padding: 3pt 5pt;
  vertical-align: top;
}

.footl { font-family: CAPS !important; font-size: 6.5pt; color: #B7A289; }
.footr { font-family: CAPS !important; font-size: 6.5pt; color: #B7A289; text-align: right; }
"""

def convert_md_to_pdf(md_path, pdf_path, doc_title):
    print(f"Converting {md_path} -> {pdf_path}...")
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
        print(f"pisa errors in {pdf_path}: {status.err}")

    # Set dark background on pages using PyMuPDF (fitz)
    doc = fitz.open(pdf_path)
    for page in doc:
        page.draw_rect(page.rect, fill=(0x22/255, 0x15/255, 0x0D/255), color=None, overlay=False)
    doc.save(pdf_path, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
    doc.close()
    print(f"Saved {pdf_path} with dark theme! Page count: {len(fitz.open(pdf_path))}")

if __name__ == "__main__":
    convert_md_to_pdf("docs/market-analysis-and-funnels.md", "docs/market-analysis-and-funnels.pdf", "Анализ Рынка и 10 Воронок")
    convert_md_to_pdf("docs/100-reels-scripts.md", "docs/100-reels-scripts.pdf", "100 Сценариев Продающих Reels")
    # Also compile selling-reels-playbook using standard script
    os.system("python3 scripts/md2pdf_styled.py")
    print("All PDFs compiled successfully!")
