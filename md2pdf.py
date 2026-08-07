#!/usr/bin/env python3
"""md2pdf.py — конвертер Markdown → PDF с поддержкой кириллицы.

Использование:
    .venv/bin/python md2pdf.py input.md [output.pdf]

Если выходной файл не указан, создаётся PDF с тем же именем рядом с MD.
Зависимости (в .venv): markdown, xhtml2pdf. Шрифты: DejaVu (системные).
"""
import sys
from pathlib import Path

import markdown as md_lib
from xhtml2pdf import pisa

FONTS = "/usr/share/fonts/truetype/dejavu"

CSS = f"""
@font-face {{ font-family: DV; src: url({FONTS}/DejaVuSans.ttf); }}
@font-face {{ font-family: DV; src: url({FONTS}/DejaVuSans-Bold.ttf); font-weight: bold; }}
@font-face {{ font-family: DVM; src: url({FONTS}/DejaVuSansMono.ttf); }}
@page {{ size: A4; margin: 1.8cm 1.6cm 2cm 1.6cm; }}
body {{ font-family: DV; font-size: 10pt; line-height: 1.45; color: #1a1a1a; }}
h1 {{ font-size: 19pt; border-bottom: 2px solid #555; padding-bottom: 4px; margin-top: 6px; }}
h2 {{ font-size: 14pt; margin-top: 16px; color: #111; }}
h3 {{ font-size: 11.5pt; margin-top: 12px; margin-bottom: 2px; }}
p {{ margin: 4px 0 8px 0; }}
hr {{ border: none; border-top: 1px solid #ccc; margin: 12px 0; }}
ul, ol {{ margin: 4px 0 8px 0; padding-left: 22px; }}
li {{ margin-bottom: 3px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 9pt; margin: 8px 0; }}
th, td {{ border: 1px solid #999; padding: 4px 6px; text-align: left; }}
th {{ background-color: #eeeeee; font-weight: bold; }}
code {{ font-family: DVM; font-size: 8.5pt; background-color: #f4f4f4; padding: 0 3px; }}
pre {{ background-color: #f4f4f4; padding: 8px; font-size: 8.5pt; }}
blockquote {{ border-left: 3px solid #bbb; margin-left: 0; padding-left: 10px; color: #555; }}
strong {{ font-weight: bold; }}
"""


def convert(src: Path, dst: Path) -> None:
    text = src.read_text(encoding="utf-8")
    html_body = md_lib.markdown(
        text, extensions=["tables", "fenced_code", "sane_lists", "nl2br"]
    )
    full = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html_body}</body></html>"
    with dst.open("wb") as out:
        status = pisa.CreatePDF(full, dest=out)
    if status.err:
        sys.exit(f"Ошибка генерации PDF: {status.err}")
    print(f"OK: {dst} ({dst.stat().st_size / 1024:.0f} КБ)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".pdf")
    convert(src, dst)
