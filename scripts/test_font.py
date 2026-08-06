# -*- coding: utf-8 -*-
import fitz
from xhtml2pdf import pisa
import markdown

test_md = """
### Рилс #001
**Описание поста и хэштеги:**
```text
Большинство людей пытаются бороться с симптомами, не замечая главного.

#шаманрахунхан #род #обряды
```
"""

html = markdown.markdown(test_md, extensions=['tables', 'fenced_code'])

full_html = f"""
<html>
<head>
<style>
@font-face {{ font-family: PS; src: url(/tmp/fonts/PT_Sans-Web-Regular.ttf); }}
body {{ font-family: PS; }}
pre, code, tt, kbd, samp {{ font-family: PS !important; font-size: 8pt; }}
</style>
</head>
<body>{html}</body>
</html>
"""

with open('/tmp/test.pdf', 'wb') as f:
    pisa.CreatePDF(full_html, dest=f)

doc = fitz.open('/tmp/test.pdf')
print("Extracted text:")
print(repr(doc[0].get_text()))
