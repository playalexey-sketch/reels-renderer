# -*- coding: utf-8 -*-
"""Designer PDF for 'Продающий рилс от обратного' playbook — storyboard style."""
import os
from xhtml2pdf import pisa
import fitz

OUT = "docs/selling-reels-playbook.pdf"
PREV = "/tmp/prev"
os.makedirs(PREV, exist_ok=True)

F = "/tmp/fonts"
Y      = f"file://{F}/YesevaOne-Regular.ttf"
PSR    = f"file://{F}/PT_Sans-Web-Regular.ttf"
PSB    = f"file://{F}/PT_Sans-Web-Bold.ttf"
CAP    = f"file://{F}/PT_Sans-Caption-Web-Regular.ttf"
CAPB   = f"file://{F}/PT_Sans-Caption-Web-Bold.ttf"
SER    = f"file://{F}/PT_Serif-Web-Regular.ttf"
SERB   = f"file://{F}/PT_Serif-Web-Bold.ttf"
SERI   = f"file://{F}/PT_Serif-Web-Italic.ttf"
MONO   = "file:///usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

# ---------- palette ----------
BG     = "#22150D"   # page background (deep mocha)
CARD   = "#2D2013"   # card background
CARD2  = "#20140B"   # deep card (code)
LINE   = "#54402D"   # hairline
TEXT   = "#EDE1CB"   # cream
MUTED  = "#B7A289"   # secondary
GOLD   = "#D9B384"   # accent
DARKB  = "#160D07"   # stat cell bg

TRACK = "\u2009"  # thin space

def L(t, short=False):
    """uppercase + manual tracking for labels (per word)"""
    t = t.upper()
    ch = TRACK if short else TRACK + TRACK
    gap = "\u00a0\u00a0"
    return gap.join(ch.join(w) for w in t.split())

def A():   # arrow left  (design direction)
    return f'<span style="color:{GOLD};">&#8592;</span>'

def R():   # arrow right (publish direction)
    return f'<span style="color:{GOLD};">&#8594;</span>'

def lab(t):
    return f'<div class="lab">{L(t)}</div>'

def head_row(left, right):
    return (f'<table style="width:100%; border:none; margin:0; padding:0;"><tr>'
            f'<td style="border:none; padding:0 0 3pt 1pt;"><span class="kl">{L(left, short=True)}</span></td>'
            f'<td style="border:none; padding:0 1pt 3pt 0; text-align:right;"><span class="kr">{L(right, short=True)}</span></td>'
            f'</tr></table>')

def chip(t):
    return (f'<span class="cta">{L(t, short=True)}</span>')

def kicker(t):
    return f'<div class="kicker">{L(t, short=True)}</div>'

def h2(title):
    return f'<div class="h2wrap">{title}</div>'

CSS_TPL = """
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
@font-face { font-family: MONO; src: url(/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf); }
@font-face { font-family: DVS; src: url(/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf); }
body { font-family: PS; font-size: 9pt; line-height: 1.5; color: __TEXT__; }
b, strong { font-weight: bold; }

/* ---------- labels & kickers ---------- */
.lab     { font-family: CAPS; font-weight: bold; font-size: 6.6pt; color: __MUTED__; margin: 4pt 0 1.5pt 0; }
.kicker  { font-family: CAPS; font-weight: bold; font-size: 7.8pt; color: __GOLD__; margin: 0 0 3pt 0; }
.kl      { font-family: CAPS; font-weight: bold; font-size: 8.1pt; color: __GOLD__; }
.kr      { font-family: CAPS; font-weight: bold; font-size: 8.1pt; color: __TEXT__; }
.footl   { font-family: CAPS; font-size: 6.6pt; color: __MUTED__; }
.footr   { font-family: CAPS; font-size: 6.6pt; color: __MUTED__; text-align: right; }

/* ---------- type blocks ---------- */
.y   { font-family: Y; }
.si  { font-family: SERI; }
.h1  { font-family: Y; font-size: 30pt; line-height: 1.14; color: __TEXT__; margin: 8pt 0 0 0; }
.h1i { font-family: SERI; font-size: 26pt; line-height: 1.15; color: __GOLD__; margin: 2pt 0 0 0; }
.h2wrap { font-family: Y; font-size: 16.5pt; color: __TEXT__; margin: 0 0 2pt 0; line-height: 1.2; }
.sub { font-family: PS; font-size: 10pt; color: __MUTED__; line-height: 1.55; }
.txt { font-family: PS; font-size: 8.7pt; color: __TEXT__; line-height: 1.5; }
.txtsmall { font-family: PS; font-size: 8.2pt; color: __MUTED__; line-height: 1.45; }
.quote { font-family: SERI; font-size: 12.5pt; color: __TEXT__; line-height: 1.55; }

/* ---------- rules & cards ---------- */
.rule { border-top: 0.7pt solid __LINE__; margin: 10pt 0; font-size: 1pt; }
.card { background-color: __CARD__; border: 0.7pt solid __LINE__; padding: 6.5pt 9pt 7.5pt 9pt; margin: 0 0 9pt 0; -pdf-keep-in-frame-mode: shrink; }
.carddeep { background-color: __CARD2__; border: 0.7pt solid __LINE__; padding: 9pt 10pt 10pt 10pt; margin: 0 0 11pt 0; }

/* ---------- chains & codes ---------- */
.chain { font-family: PS; font-size: 7.9pt; color: __TEXT__; line-height: 1.42; }
.chainlead { font-family: CAPS; font-weight: bold; font-size: 8pt; color: __GOLD__; }
.time { font-family: MONO; font-size: 7.3pt; color: __GOLD__; line-height: 1.6; }
pre.code { font-family: MONO; font-size: 7.15pt; line-height: 1.42; color: #D8C9AE; white-space: pre; margin: 0; }
.hook { font-family: SERI; font-size: 10pt; color: #F3E6CF; line-height: 1.4; margin: 1pt 0 2.5pt 0; }
.flow { font-family: PS; font-size: 8pt; color: __MUTED__; line-height: 1.4; margin: 0 0 3.5pt 0; }
.arr { font-family: DVS; color: __GOLD__; }
.cta  { font-family: CAPS; font-weight: bold; font-size: 7.9pt; color: #F0D9AC;
        border: 0.8pt solid #A9855A; padding: 2pt 6pt; }

/* ---------- tables ---------- */
table.std { width: 100%; border-collapse: collapse; margin: 6pt 0 8pt 0; }
table.std th { font-family: CAPS; font-weight: bold; font-size: 7.3pt; color: __TEXT__;
               background-color: #382A1C; border: 0.6pt solid __LINE__; padding: 4pt 6pt; text-align: left; }
table.std td { font-family: PS; font-size: 8.3pt; color: __TEXT__; border: 0.6pt solid __LINE__;
               padding: 4pt 6pt; vertical-align: top; }
table.std td.p { font-family: CAPS; font-weight: bold; font-size: 7.6pt; color: __GOLD__; white-space: nowrap; }

/* ---------- stats row (cover) ---------- */
table.stats { width: 100%; border-collapse: collapse; margin: 12pt 0 0 0; }
table.stats td { border: 0.7pt solid __LINE__; background-color: __DARKB__;
                 padding: 12pt 10pt 10pt 10pt; vertical-align: top; width: 33%; }
.statn { font-family: Y; font-size: 25pt; color: __TEXT__; }
.statl { font-family: CAPS; font-weight: bold; font-size: 6.9pt; color: __MUTED__; }

/* steps grid */
table.steps { width: 100%; border-collapse: collapse; }
table.steps td { border: 0.7pt solid __LINE__; background-color: __CARD__;
                 padding: 8pt 9pt; vertical-align: top; width: 50%; }
.stepn { font-family: Y; font-size: 16pt; color: __GOLD__; }
.stept { font-family: CAPS; font-weight: bold; font-size: 8pt; color: __TEXT__; margin: 0 0 2pt 0; }
.stepx { font-family: PS; font-size: 8pt; color: __MUTED__; line-height: 1.45; }

/* QC */
table.qc { width: 100%; border-collapse: collapse; margin-top: 4pt; }
table.qc td { border-bottom: 0.5pt solid __LINE__; padding: 4.5pt 4pt; vertical-align: top; }
table.qc td.n { font-family: Y; font-size: 10.5pt; color: __GOLD__; width: 22pt; }
table.qc td.q { font-family: PS; font-size: 8.5pt; color: __TEXT__; }
"""

CSS = (CSS_TPL
       .replace("__Y__", Y).replace("__PSR__", PSR).replace("__PSB__", PSB)
       .replace("__CAP__", CAP).replace("__CAPB__", CAPB)
       .replace("__SER__", SER).replace("__SERB__", SERB).replace("__SERI__", SERI)
       .replace("__MONO__", MONO)
       .replace("__BG__", BG).replace("__LINE__", LINE).replace("__CARD__", CARD)
       .replace("__CARD2__", CARD2).replace("__TEXT__", TEXT).replace("__MUTED__", MUTED)
       .replace("__GOLD__", GOLD).replace("__DARKB__", DARKB))

bg_div = (f'<div id="bg_content" style="width:21cm; height:29.7cm; background-color:{BG};"></div>')
foot_div = (
    f'<div id="footer_content">'
    f'<table style="width:100%; border:none; border-top:0.5pt solid {LINE}; margin:0;"><tr>'
    f'<td style="border:none; padding:2pt 0 0 0;"><span class="footl">{L("Продающий рилс от обратного · стандарт", short=True)}</span></td>'
    f'<td style="border:none; padding:2pt 0 0 0; text-align:right;"><span class="footr">{L("стр.", short=True)} <pdf:pagenumber/></span></td>'
    f'</tr></table></div>')

# ============================ CONTENT ============================
pages = []

# ---------------- P1 · COVER ----------------
pages.append(f"""
<div style="height:78pt;"></div>
<table style="width:100%; border:none; margin:0;"><tr>
  <td style="border:none; padding:0;"><span class="kl">{L("стандарт · reels", short=True)}</span></td>
  <td style="border:none; padding:0; text-align:right;"><span class="kr">{L("playbook · 2026", short=True)}</span></td>
</tr></table>
<div class="rule" style="margin:7pt 0 26pt 0;"></div>
<div class="kicker" style="text-align:center;">{L("ТЗ генератору · короткие видео · продажи", short=True)}</div>
<div class="h1" style="text-align:center;">Продающий рилс</div>
<div class="h1i" style="text-align:center;">от обратного</div>
<div style="height:18pt;"></div>
<div class="sub" style="text-align:center; width:15.5cm; margin-left:auto; margin-right:auto;">
Как ставить детальное задание генератору коротких видео, чтобы на выходе — рилс по стандартам
продающего ролика. Универсальный алгоритм, 12 концепций и формат ответа под пайплайн
HeyGen API + ffmpeg (озвучка, субтитры, SFX, CTA-кадр).
</div>
<table class="stats"><tr>
  <td><span class="statn">12</span><br/><span class="statl">{L("12 концепций К1–К12")}</span></td>
  <td><span class="statn">7+1</span><br/><span class="statl">{L("шагов алгоритма")}</span></td>
  <td><span class="statn">1</span><br/><span class="statl">{L("CTA на ролик")}</span></td>
</tr></table>
<div style="height:26pt;"></div>
<div class="txtsmall" style="text-align:center;">
Ниша-пример: «Шаманские практики и обряды» · сценарии по концепциям — в файле reels-all-concepts.md
</div>
""")

# ---------------- P2 · СХЕМА ОТ ОБРАТНОГО ----------------
def lane(title, items, arrow, lead_first=True):
    n = len(items)
    cells = ""
    for i, it in enumerate(items):
        hl = (i == 0) if lead_first else (i == n - 1)
        st = f' style="font-family:CAPS; font-weight:bold; color:{GOLD};"' if hl else ""
        cells += f"<td style='border:none; padding:2pt;'><span{st}>{it}</span></td>"
        if i < n - 1:
            cells += f"<td style='border:none; padding:2pt; width:13pt; text-align:center;'>{arrow}</td>"
    return (f"<div class='card' style='padding-bottom:8pt;'>"
            f"<div class='lab' style='margin:0 0 5pt 0;'>{L(title, short=True)}</div>"
            f"<table style='width:100%; border:none; margin:0; font-family:CAPS; font-size:7.4pt; color:#DECFB6;'>"
            f"<tr>{cells}</tr></table></div>")

chain_design = ["CTA", "оффер", "возражение", "обещание", "доказательство", "боль", "ХУК"]
chain_public = ["ХУК", "боль", "доказательство", "обещание", "оффер", "CTA"]

pages.append(f"""
{kicker("Раздел 01 · логика")}
{h2("Собираем от продажи, публикуем от хука")}
<div class="rule"></div>
<div class="txt" style="margin-bottom:8pt;">
Продающий рилс не сочиняют от первой фразы — его <b>проектируют от целевого действия</b>.
Сначала фиксируем, что зритель должен сделать, и идём назад по логике доверия — до стоп-кадра.
На публикацию цепочка выходит зеркальной.
</div>
{lane("проектируем · направо", chain_design, A())}
{lane("публикуем · налево", chain_public, R(), lead_first=False)}
<div class="carddeep">
<div class="quote" style="text-align:center; padding:4pt 8pt;">
«Хук, который не приводит к CTA, — кликбейт.<br/>
CTA, к которому не ведёт хук, — приклеенная фраза.»
</div>
</div>
<div class="txtsmall">
Тест обратной сборки: читаем готовую цепочку вслух <b>вперёд</b>. Каждый блок должен отвечать на вопрос,
который задал предыдущий; удаление любого блока ломает продажу; CTA — единственный логичный следующий шаг.
Если блок можно удалить безболезненно — он лишний.
</div>
""")

# ---------------- P3 · СТАНДАРТЫ ----------------
std_rows = [
    ("Длительность", "20–40 сек — продающий; списки и чек-листы до 60 сек: нумерация удерживает досмотр."),
    ("Хук", "1–3 сек, без «привет» и представлений. Первый кадр = обложка с тезой-хуком."),
    ("Единство", "Один ролик = одна мысль = один CTA. Всё лишнее — в следующий рилс."),
    ("Смысловой блок", "3–7 сек, дальше — поворот или новый аргумент. Каждая секунда продаёт следующую."),
    ("Паттерн-интеррапт", "Каждые 2–4 сек: зум, джамп-кат, смена плана, SFX-акцент. Мозг не должен «привыкнуть»."),
    ("Субтитры", "Дублируют речь, строки по 3–5 слов, крупно, в сейф-зонах (низ ~250px, верх ~200px)."),
    ("Звук", "Голос в приоритете, музыка −20…−25 дБ под голосом, 2–4 SFX-акцента на ролик."),
    ("CTA", "Последние 3–5 сек + статичный кадр 1–2 сек. ≤6 слов: глагол + кодовое слово + куда."),
    ("Закадр", "≈60–75 слов на 30 сек ролика (под озвучку аватаром)."),
    ("Удержание", "Ориентиры: 3-я сек ≥ 70% · середина ≥ 45–50% · досмотр ≥ 25–30%."),
    ("Диагностика", "Провал 0–3 сек → переписать хук · середина → резать воду · перед CTA → CTA поздний или выпал из сюжета."),
]
rows_html = "".join(
    f'<tr><td class="p" style="width:24%;">{r[0].upper()}</td><td>{r[1]}</td></tr>' for r in std_rows)
pages.append(f"""
{kicker("Раздел 02 · стандарты")}
{h2("Контрольная рамка продающего рилса")}
<div class="rule"></div>
<table class="std">
<tr><th style="width:24%;">{L("параметр", short=True)}</th><th>{L("стандарт", short=True)}</th></tr>
{rows_html}
</table>
<div class="carddeep">
<div class="lab" style="margin-top:0;">{L("зачем рамка", short=True)}</div>
<div class="txtsmall">
Генератор без рамки выдаёт «красивый текст». Генератор с рамкой выдаёт конструкцию, которую можно
измерить и улучшить: хук держит 3-ю секунду, нумерация держит середину, CTA-кадр конвертирует досмотревших.
</div></div>
""")

# ---------------- P4 · МАСТЕР-АЛГОРИТМ 7+1 ----------------
steps = [
    ("00", "Единственное действие", "Что зритель должен сделать после просмотра: кодовое слово в директ / ссылка в шапке / запись. Одно ролик — одно действие. Не два."),
    ("01", "CTA-фраза", "Глагол + кодовое слово + место: «Напиши “РОД” в директ». Всё, что не ведёт к ней, — удалить из сценария."),
    ("02", "Оффер", "Формат + одна ключевая выгода + снятие риска: «Диагностика 40 минут: найдём повторяющийся сценарий, план из трёх шагов, без страшилок»."),
    ("03", "Возражение-фильтр", "Главное «а что, если…» аудитории — и одна фраза-антидот внутри ролика, пока зритель не успел подумать её сам."),
    ("04", "Обещание", "Одна конкретная выгода — зеркало боли. Не «гармонизация энергий», а «перестать повторять сценарий разводов по женской линии»."),
    ("05", "Доказательство", "Одно, но настоящее: кейс / демонстрация процесса / цифра практики / логика механизма («род передаёт не только гены, но и сценарии»)."),
    ("06", "Боль и триггер", "Словами клиента из директа, не эксперта: «проживаю мамину жизнь» — боль. «Негативные программы рода» — жаргон, доверия не даёт."),
    ("07", "Хук", "Стоп-фраза на 1–3 сек. Три проверки: цепляет именно ЦА · честен (ролик реально отвечает) · на него нельзя мысленно ответить «ну и ладно»."),
]
cells = ""
for i, (n, t, x) in enumerate(steps):
    cells += (f'<td><span class="stepn">{n}</span>&nbsp;&nbsp;<span class="stept">{L(t, short=True)}</span>'
              f'<div class="stepx">{x}</div></td>')
    if i % 2 == 1:
        cells = f'<tr>{cells}</tr>' if False else cells
step_rows = ""
for i in range(0, 8, 2):
    n1, t1, x1 = steps[i]; n2, t2, x2 = steps[i + 1]
    step_rows += (f'<tr>'
        f'<td><span class="stepn">{n1}</span>&nbsp;&nbsp;<span class="stept">{L(t1, short=True)}</span><div class="stepx">{x1}</div></td>'
        f'<td><span class="stepn">{n2}</span>&nbsp;&nbsp;<span class="stept">{L(t2, short=True)}</span><div class="stepx">{x2}</div></td>'
        f'</tr>')
pages.append(f"""
{kicker("Раздел 03 · мастер-алгоритм")}
{h2("7 шагов «от обратного» + шаг ноль")}
<div class="rule"></div>
<table class="steps">{step_rows}</table>
<div class="carddeep" style="margin-top:9pt;">
<div class="lab" style="margin-top:0;">{L("тест обратной сборки", short=True)}</div>
<div class="txt">Читаем цепочку вслух вперёд: хук → боль → доказательство → обещание → оффер → CTA.
Три правила: каждый блок отвечает на вопрос предыдущего · удаление любого блока ломает продажу ·
CTA — естественное завершение истории, а не наклейка.</div>
</div>
""")

# ---------------- P5 · ШАБЛОН ТЗ ----------------
tz = """РОЛЬ
Ты — сценарист продающих коротких видео (Reels/Shorts) в нише {НИША}.
Пишешь по стандартам удержания: хук 1–3 сек, смысловой блок 3–7 сек,
паттерн-интеррапт каждые 2–4 сек, один ролик = одна мысль = один CTA.

ВХОДНЫЕ ДАННЫЕ
Продукт: {что продаём: формат, длительность, цена, что входит}
ЦА: {кто: возраст, пол, боль словами клиента, желание, топ-3 возражения, холодная/тёплая}
Цель ролика: {единственное действие: кодовое слово в директ / ссылка в шапке / запись}
Концепция: {К1–К12 из списка}
Длительность: {20 / 30 / 40 / 60 сек}
Тон: {спокойный эксперт / тёплый рассказчик / жёсткая провокация}
Запреты: без медицинских и финансовых гарантий, без запугивания, без жаргона ЦА.

ЗАДАЧА — СОБЕРИ РОЛИК «ОТ ОБРАТНОГО»
1. Сформулируй CTA (глагол + кодовое слово + куда, ≤6 слов).
2. Запиши оффер: формат + одна ключевая выгода + снятие риска.
3. Назови главное возражение и одну фразу-антидот внутри ролика.
4. Сформулируй обещание — конкретную выгоду, зеркало боли.
5. Выбери ОДНО доказательство: кейс / процесс / цифра / механизм.
6. Запиши боль словами клиента.
7. Напиши 3 варианта хука на 1–3 сек; каждый должен логически «требовать»
   цепочку боль → доказательство → оффер → CTA.
8. Собери ролик вперёд. Проверка: каждая секунда продаёт следующую;
   удаление любого блока ломает продажу; CTA — естественный финал.

ФОРМАТ ОТВЕТА (под пайплайн HeyGen + ffmpeg)
Таблица: № | таймкод | закадровый текст (для озвучки) | текст на экране (субтитры/тезы)
        | визуал/план | SFX / паттерн-интеррапт | visual prompt (EN, если генерим футаж)
Отдельно: а) 3 варианта хука; б) текст CTA-кадра (статичный, 1–2 сек);
в) закадровый текст сплошняком (≤{N} слов под {сек} сек);
г) описание поста 150–300 знаков + вопрос для комментариев; д) 5 хэштегов."""

pages.append(f"""
{kicker("Раздел 04 · шаблон")}
{h2("Универсальное ТЗ генератору")}
<div class="rule"></div>
<div class="txt" style="margin-bottom:7pt;">Шаблон заполняется под конкретный ролик за 2–3 минуты: подставляете
продукт, ЦА, цель и выбираете концепцию из раздела 05. Остальное — неизменная рамка качества.</div>
<div class="carddeep"><pre class="code">{tz}</pre></div>
""")

# ---------------- CONCEPTS ----------------
CONCEPTS = [
 ("К1 · PAS", "30 сек",
  "Проблема → усиление → решение",
  "Боль явная, ЦА её осознаёт. Рабочая лошадка продающих рилс.",
  "CTA " + "&#8592; ".join(["оффер-решение (что это, как проходит, снятие риска)",
  "3 качания боли по нарастающей: сейчас → если не решать → кого ещё заденет",
  "проблема словами клиента", "хук-диагноз, в котором ЦА узнаёт себя"]),
  "0–3 ХУК · 3–9 проблема · 9–18 три качания (экран-тезы) · 18–26 решение + как проходит · 26–30 CTA",
  "«Три развода по женской линии — это не “не везёт”. Это сценарий рода.»",
  "Проблема: ссоры-копирки. Качания: совпадение → год спустя громче → то же у дочери. "
  "Решение: диагностика 40 минут — находим точку запуска сценария.",
  "Напиши «СЦЕНАРИЙ» в директ"),
 ("К2 · AIDA", "35 сек",
  "Внимание → интерес → желание → действие",
  "Холодная охватная ЦА: создаём спрос с нуля за полминуты.",
  "CTA + ограничитель («первым 5») " + "&#8592; ".join(["три выгоды «жизни после»",
  "доказательство", "интерес: новизна механизма («как это работает»)", "хук-внимание: конкретная, чуть провокационная постановка"]),
  "0–3 внимание · 3–10 интерес (механизм) · 10–25 желание (3 выгоды, нумерация) · 25–31 доказательство · 31–35 действие",
  "«Почему в одних семьях деньги копятся, а в других “сгорают” за день?»",
  "Интерес: денежный потолок рода — внутренний запрет богатеть. Желание: хватит сливать суммы · "
  "спокойно называешь цену · наследство без страха. Доказательство: кейс/цифра.",
  "«ДЕНЬГИ» · первым 5 — бесплатно"),
 ("К3 · До/После/Мост", "25 сек",
  "BAB: before → after → bridge",
  "Есть контраст результата, который легко почувствовать; тёплая ЦА.",
  "CTA " + "&#8592; ".join(["мост: конкретный продукт-действие между состояниями",
  "«после» в ощущениях («просыпаюсь без камня в груди»)", "«до», максимально узнаваемое зрителем",
  "хук-контраст: спайка «до/после» в первой фразе"]),
  "0–3 хук-контраст · 3–10 до · 10–17 после · 17–22 мост · 22–25 CTA",
  "«Каждое утро — камень в груди. Через сорок дней она просыпается тихой.»",
  "До: тревога «не её», но работает на полную. После: то же утро, но сон глубокий. "
  "Мост: одна диагностика рода + ритуал возвращения силы.",
  "«СИЛА» в директ"),
 ("К4 · Сторителлинг-кейс", "40 сек",
  "Герой → драма → поворот → результат → мораль",
  "Продаём дорогое и нужное доверие: аудитория любит «послушать истории».",
  "CTA " + "&#8592; ".join(["мораль, из которой оффер логичен («проблема была не в ней, а в сценарии»)",
  "результат героя — то, чего хочет ЦА", "поворот: какое действие привело к результату",
  "драма-топка, узнаваемая ЦА", "герой = портрет зрителя (возраст, статус, боль)",
  "хук из пика драмы с недосказанностью"]),
  "0–3 хук-пик · 3–10 герой и драма · 10–18 попытки и тупик · 18–28 поворот · 28–34 результат · 34–40 мораль + CTA",
  "«Она 12 лет не разговаривала с мамой. Причина нашлась в 1943 году.»",
  "Марина, 41: психологи и тренинги — откаты. В древе — три поколения обид на матерей. "
  "Возврат: после обряда сама набрала номер.",
  "«РОД» в директ"),
 ("К5 · Разрушение мифа", "30 сек",
  "Миф → правда → что делать",
  "Ниша полна страхов и стереотипов (для шаманики — хлеб). Спор в комментариях разгоняет охват.",
  "CTA " + "&#8592; ".join(["правильный шаг (что делать вместо)", "правда с двумя аргументами",
  "миф — точной цитатой ЦА (в него должна верить половина зала)", "хук = миф + «ответ практика»"]),
  "0–4 миф-хук · 4–14 почему миф неверен (2 аргумента) · 14–22 как на самом деле · 22–27 микро-доказательство · 27–30 CTA",
  "«“Чистка рода — это опасно”? Отвечаю как практик с 12 годами работы.»",
  "Диагностика — фонарик, а не лом. «Чтобы разбудить род, нужны силы больше самого рода». "
  "Опаснее годами таскать чужой страх и звать его характером.",
  "«РОД» в директ"),
 ("К6 · Список ошибок", "45 сек",
  "«N ошибок, которые…»",
  "ЦА боится навредить. Нумерация удерживает до конца — счётчик на экране.",
  "CTA (чек-лист по кодовому слову) " + "&#8592; ".join(["антидот к каждой ошибке одной фразой",
  "сортировка: №1 самая частая (узнавание) … последняя — самая дорогая (страх потери ведёт к CTA)",
  "хук-угроза ошибкой"]),
  "0–3 хук · 3–33 ошибки по 9–10 сек (ошибка → последствие → антидот) · 33–40 связка с оффером · 40–45 CTA",
  "«Три ошибки, из-за которых ваша “работа с родом” работает против вас.»",
  "1) Молчать о «вычеркнутых» — роль доигрывают дети. 2) Поминать с укором. "
  "3) Пугать детей «у нас в роду все такие» — ребёнок не анализирует, он соглашается.",
  "«ЧЕК» в директ — пришлю бесплатно"),
 ("К7 · ТОП-признаки", "50 сек",
  "Чек-лист самодиагностики",
  "Холодная ЦА «диагностирует» себя по списку — самосегментация прямо в просмотре.",
  "CTA («если 2+ признака про вас…») " + "&#8592; ".join(["развязка: «это не приговор, вот первый шаг»",
  "признаки: №1 самый массовый … финальный — самый тревожный (доводит до CTA)",
  "хук-дозировка: «третий — почти у всех»"]),
  "0–3 хук · 3–38 признаки по 6–7 сек, нумерация на экране · 38–47 развязка · 47–50 CTA",
  "«Пять признаков, что ваш род просит внимания. Третий — почти у всех.»",
  "Одна драма у всех поколений · имя = роль · деньги не задерживаются · частые сны с ушедшими · "
  "детский страх в 30+. Признак — симптом словами клиента, не диагноз в лоб.",
  "«ПРИЗНАКИ» в директ"),
 ("К8 · Секрет / инсайд", "30 сек",
  "«То, что вижу только практик»",
  "Продаём диагностику-вход: бесплатный инсайд создаёт аппетит, не насыщает.",
  "CTA (" + "&#8592; ".join(["«полная диагностика глубже в десять раз»)",
  "граница: «всё древо показываю на сессии»", "три инсайда — занавес поднят на 10%",
  "обещание «смотрите, как читается род»", "хук-приглашение"]),
  "0–3 хук · 3–20 три инсайда · 20–26 граница «дальше — на диагностике» · 26–30 CTA",
  "«Что шаман видит в вашем родовом древе за первые три минуты?»",
  "Пустые ветви (о ком молчат) · повторы имён (имя = роль) · перекос силы на одной ветви. "
  "Это видно за минуты — дальше глубже.",
  "«ДРЕВО» в директ"),
 ("К9 · Провокация", "25 сек",
  "«От противного» / антитренд",
  "Ниша перегрета «халявщиками»: отсев нецелевых прогревает зрелого клиента и чек.",
  "CTA " + "&#8592; ".join(["портрет зрелого клиента (кому НАДО)", "список отсева (кому НЕ надо)",
  "тезис-переворот", "хук-разворот: «не приходите…, если…»"]),
  "0–3 хук-отказ · 3–15 отсев («волшебная таблетка», «это всё бабушка») · 15–21 «а если готовы честно…» · 21–25 CTA",
  "«Не приходите ко мне на чистку рода. Если вы…»",
  "…ищете таблетку; перекладываете жизнь на прабабушку; ждёте эффект с дивана. "
  "Бить по боли, не по уязвимости: травма, вера, политика — табу.",
  "«ПРАВДА» в директ"),
 ("К10 · Процесс / бэкстейдж", "30 сек",
  "Эстетика практики + ASMR-удержание",
  "Продукт атмосферный и визуальный: продажа ощущением, звук фактур в микс.",
  "CTA (" + "&#8592; ".join(["«хочешь свою — кодовое слово»)",
  "интрига скрытого: «полный ритуал — только на сессии»", "три элемента: показ 2–3 сек макро + смысл в одну фразу",
  "обещание атмосферы", "хук-антураж"]),
  "0–3 хук + крупный план · 3–24 три элемента (макро → смысл) · 24–27 интрига · 27–30 CTA",
  "«Сбор обряда очищения рода. Каждый предмет здесь — не декорация.»",
  "Круг камней = границы работы · земля/соль = память рода · дым полыни = переход. "
  "Полный ритуал рождается под конкретный запрос.",
  "«ОБРЯД» в директ"),
 ("К11 · Соцдоказательство", "30 сек",
  "Отзыв / кейс / цифры",
  "ЦА интересуется, но не решается. Дожим конкретикой, не восторгами.",
  "CTA («вы следующая история») " + "&#8592; ".join(["механика «как такое повторить»",
  "кейс с деталями: кто, что было, что стало", "зачем показываю: «спрашиваете — работает ли»",
  "хук-агрегат: цифра + закономерность"]),
  "0–3 хук-цифра · 3–20 кейс/скрин отзыва (с согласия) · 20–26 «как повторить» · 26–30 CTA",
  "«400 семей прошли диагностику. Знаете, что повторяется у восьми из десяти?»",
  "Фраза «наконец-то понятно». Кейс Ольги: роль «миротворицы» досталась вместе с именем. "
  "«Наладились отношения со свекровью за месяц» продаёт лучше, чем «изменилась жизнь!!!»",
  "Ссылка в шапке / «РОД» в директ"),
 ("К12 · FAQ", "25 сек",
  "Один вопрос из директа = один рилс",
  "Контент-конвейер: серия закрывает возражения и собирает вопросы в комментарии.",
  "CTA (" + "&#8592; ".join(["«остальные вопросы — в директ»)",
  "«как правильно» в 2–3 пунктах", "развёрнутое «почему»",
  "короткий честный ответ в первые 5 сек (да/нет)", "хук = цитата вопроса дословно"]),
  "0–3 вопрос-цитата на экране · 3–6 короткий ответ · 6–18 почему · 18–22 как правильно · 22–25 CTA",
  "«Можно ли работать с родом без ведома родственников?»",
  "Можно: вы работаете со своим местом в системе, а не «за» кого-то. "
  "Не тащите непрошеных — это то же насилие, только с бубном.",
  "«ВОПРОС» в директ"),
]

def concept_card(c, first_on_page=False):
    code_dur, dur, title, when, chain, time, hook, flow, cta = c
    chain_html = (f'<span class="chainlead">{chain.split(" ",1)[0]}</span> '
                  + f'<span style="color:{GOLD};">&#8592;</span> ' + c[4].split(" ",1)[1]) if False else \
                 (f'<span class="chainlead">{chain.split("&#8592;",1)[0].strip()}</span> '
                  + f'<span style="color:{GOLD};">&#8592;</span> ' + chain.split("&#8592;",1)[1])
    return (('' if first_on_page else '') +
      head_row(code_dur, dur) +
      f'<div class="card">'
      f'<div style="font-family:Y; font-size:11.5pt; color:{TEXT}; margin:0 0 2pt 0;">{title}</div>'
      f'<div class="flow" style="margin-bottom:4pt;">{when}</div>'
      f'{lab("от обратного")}<div class="chain">{chain_html}</div>'
      f'{lab("каркас")}<div class="time">{time}</div>'
      f'{lab("пример · шаманская ниша")}<div class="hook">{hook}</div>'
      f'<div class="flow">{flow}</div>'
      f'<div>{chip(cta)}</div>'
      f'</div>')

concepts_intro = f"""
{kicker("Раздел 05 · концепции")}
{h2("12 концепций: алгоритм «от обратного» + каркас + пример")}
<div class="rule"></div>
<div class="txtsmall" style="margin-bottom:8pt;">Секунды в каркасах ориентировочны — масштабируйте под длительность,
сохраняя пропорции. Полные сценарии по каждой концепции с закадром и раскадровкой — в файле
examples/shaman-rod/reels-all-concepts.md.</div>
"""
# 2 concepts per page
pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
for idx, (i, j) in enumerate(pairs):
    body = ""
    if idx == 0:
        body += concepts_intro
    body += concept_card(CONCEPTS[i], first_on_page=True)
    body += concept_card(CONCEPTS[j])
    pages.append(body)

# ---------------- P · ДОП КОНЦЕПЦИИ + QC ----------------
extras = [
 ("Д1 · Сравнение X vs Y", "35 сек",
  "«Психология или работа с родом — что когда»",
  "CTA &#8592; «чего не решает X» &#8592; честная симметрия (доверие растёт) &#8592; хук-дилемма "
  "«3 года у психолога, а сценарий повторяется. Почему?»", "«РОД» в директ"),
 ("Д2 · POV / тренд", "20 сек",
  "Трендовая механика + нишевой смысл",
  "CTA &#8592; нишевой смысл &#8592; трендовый формат/звук &#8592; хук «как у всех, но…». "
  "Пример: «POV: тебе 35, а детский страх никуда не делся»", "«СИЛА» в директ"),
 ("Д3 · Прямой оффер", "20 сек",
  "Только для прогретой аудитории",
  "Дедлайн &#8592; состав мест/цена &#8592; кому &#8592; хук-объявление: "
  "«Открываю 5 мест на обряд очищения рода до конца месяца»", "Ссылка в шапке"),
]
extras_html = ""
for code_dur, dur, title, chain, cta in extras:
    extras_html += (head_row(code_dur, dur) +
      f'<div class="card"><div class="flow" style="margin-bottom:3pt;">{title}</div>'
      f'<div class="chain">{chain}</div><div style="height:4pt;"></div><div>{chip(cta)}</div></div>')

qc = [
 "Одно целевое действие? Хук сходится с CTA одной логической линией?",
 "Хук 1–3 сек, без приветствий; на нём нельзя «закрыть вопрос» в уме?",
 "Боль сформулирована словами клиента, а не эксперта?",
 "Одно доказательство — конкретное (цифра / кейс / процесс / механизм)?",
 "Главное возражение закрыто одной фразой внутри ролика?",
 "Обещание конкретное; ноль медицинских и финансовых гарантий?",
 "Паттерн-интеррапт каждые 2–4 сек — помечен в таблице планов/SFX?",
 "Закадр ≤ 75 слов на 30 сек; субтитры 3–5 слов в строке, сейф-зоны соблюдены?",
 "CTA ≤ 6 слов: глагол + кодовое слово + куда; статичный CTA-кадр 1–2 сек?",
 "Прямая читка цепочки вслух: нет «дыр» и воды?",
 "Есть 3 варианта хука для A/B-теста первых трёх секунд?",
 "Описание поста задаёт вопрос для комментариев?",
]
qc_rows = "".join(f'<tr><td class="n">{i+1:02d}</td><td class="q">{q}</td></tr>' for i, q in enumerate(qc))

pages.append(f"""
{kicker("Раздел 06 · дополнительно и контроль")}
{h2("Ещё 3 концепции + чек-лист перед рендером")}
<div class="rule"></div>
{extras_html}
{lab("QC-чек-лист · 12 вопросов")}
<div class="card"><table class="qc">{qc_rows}</table></div>
""")

# ---------------- P · ИТЕРАЦИИ + ФИНАЛ ----------------
iters = [
 ("A/B-хуки", "3 версии ролика отличаются только первым блоком (0–3 сек и обложка). Победитель по удержанию 3-й секунды — в масштаб."),
 ("Читаем график удержания", "Обвал 0–3 сек → хук/обложка · ступень в середине → «вода» в блоке — резать · обвал перед CTA → CTA выпал из сюжета: добавить мостик."),
 ("Комментарии = новые боли", "Вопросы из комментариев превращаем в конвейер К12 (FAQ) и К6 (ошибки)."),
 ("Серийность", "Одна боль → 3–4 ролика на разных концепциях (К1, К5, К7, К12): разные входы в один оффер."),
 ("Метрика продажи", "Не охваты: доля досмотревших, кликнувших CTA; директ-конверсия по кодовому слову."),
]
iter_rows = "".join(f'<tr><td class="n">{i+1:02d}</td><td class="q"><b style="color:{TEXT};">{t}.</b> {x}</td></tr>'
                    for i, (t, x) in enumerate(iters))

pages.append(f"""
{kicker("Раздел 07 · итерации")}
{h2("Как читать аналитику и переписывать")}
<div class="rule"></div>
{lab("цикл улучшений")}
<div class="card"><table class="qc">{iter_rows}</table></div>
<div class="carddeep" style="margin-top:14pt;">
<div style="text-align:center; padding:10pt 6pt 12pt 6pt;">
<div class="kicker" style="text-align:center;">{L("финал · проверка цепочки", short=True)}</div>
<div class="h1" style="font-size:22pt; text-align:center;">Собирайте от продажи.</div>
<div class="quote" style="text-align:center; margin-top:6pt;">
«Хук, который не приводит к CTA, — кликбейт.<br/>CTA, к которому не ведёт хук, — наклейка.»
</div>
<div style="height:8pt;"></div>
<div class="footl" style="text-align:center;">{L("reels-renderer · heygen api + ffmpeg · субтитры · sfx · cta", short=True)}</div>
</div></div>
""")

body = '<div style="page-break-after: always;"></div>'.join(
    f'<div>{p}</div>' for p in pages)

body = (body.replace(";&#8592;", ";&larr;")  # placeholder guard
   .replace("&:#8592;", "&&#8592;"))
body = body.replace("&#8592;", '<span class="arr">&#8592;</span>')
body = body.replace("&#8594;", '<span class="arr">&#8594;</span>')
body = body.replace("\u2192", '<span class="arr">\u2192</span>')
body = body.replace("\u2190", '<span class="arr">\u2190</span>')
html = f"<html><head><style>{CSS}</style></head><body>{bg_div}{foot_div}{body}</body></html>"



with open(OUT, "w+b") as f:
    status = pisa.CreatePDF(html, dest=f, encoding="utf-8")
print("pisa errors:", status.err)

doc = fitz.open(OUT)
for page in doc:
    page.draw_rect(page.rect, fill=(0x22/255, 0x15/255, 0x0D/255), color=None, overlay=False)
doc.save(OUT, incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
doc.close()
doc = fitz.open(OUT)
print("pages:", len(doc))
for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=88)
    pix.save(f"{PREV}/p{i+1:02d}.png")
print("previews saved:", sorted(os.listdir(PREV)))
