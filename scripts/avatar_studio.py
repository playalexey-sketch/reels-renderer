# -*- coding: utf-8 -*-
"""
AVATAR STUDIO v2 — САМОПРОВЕРЯЮЩИЙСЯ конвейер (Kaggle / Colab / сервер).

Этапы (каждый проверяет сам себя; уже готовые — пропускаются):
  1) setup    — Wav2Lip + веса + ffmpeg (проверка размеров и импортов)
  2) dataset  — нарезка видео на сегменты + разбор на кадры (метки .ok/.bad)
  3) train    — дообучение Wav2Lip (чекпоинты каждые 500 итераций, resume)
  4) generate — финальное видео (проверка результата по размеру и кадрам)

Что делает систему надёжной:
  • watchdog-поток: каждые 30 сек печатает «жив ли процесс»; если шаг молчит
    дольше STALL_SEC секунд — принудительно перезапускает зависший этап;
  • после КАЖДОЙ попытки этапа идёт проверка результата (verify), а не «на веру»;
  • внутри циклов — самопроверка места на диске и живости;
  • всё возобновляется: повторный запуск ячейки продолжает с места остановки;
  • журнал всего: STUDIO/studio.log, состояние: STUDIO/state.json.

Запуск:  python avatar_studio.py   (или runpy из ячейки Kaggle)
Переменные окружения:
  STUDIO_DIR   — рабочая папка (Kaggle: /kaggle/working/STUDIO)
  MARIA_VIDEO  — путь к исходному видео
  MAX_FRAMES   — сколько кадров собрать (по умолчанию 2000 — этого достаточно)
  TRAIN_EPOCHS — сколько эпох обучения (по умолчанию 2)
  STALL_SEC    — порог молчания для watchdog в секундах (по умолчанию 1800)
  PREVIEW_ITERS — через сколько итераций обучения сделать первый предпросмотр
                  5-сек видео (по умолчанию 200 — это ~30 секунд на T4)
  PREVIEW_STOP — '1' = остановиться сразу после первого предпросмотра,
                 чтобы посмотреть результат и решить, продолжать или нет
"""
import os, sys, time, glob, json, re, shutil, subprocess, threading, traceback

BASE   = os.environ.get('STUDIO_DIR', '/content/STUDIO')
W2L    = os.path.join(BASE, 'Wav2Lip')
SEGD   = os.path.join(BASE, 'segs')
DSD    = os.path.join(BASE, 'dataset')
LOGF   = os.path.join(BASE, 'studio.log')
STATEF = os.path.join(BASE, 'state.json')
CKPT   = os.path.join(BASE, 'personal_maria.pth')
LASTP  = os.path.join(BASE, 'personal_maria.last.pth')
FLAG   = os.path.join(BASE, '.done_train')
RAWV   = os.path.join(BASE, 'personal_raw.mp4')
FINV   = os.path.join(BASE, 'personal_final.mp4')
SRCV   = os.path.join(BASE, 'src_maria.mp4')

VIDEO      = os.environ.get('MARIA_VIDEO', '/kaggle/input/datasets/alexeyms/mariairkhina/Maria.mp4')
MAX_FRAMES = int(os.environ.get('MAX_FRAMES', '2000'))
MIN_FRAMES = int(os.environ.get('MIN_FRAMES', '800'))
EPOCHS     = int(os.environ.get('TRAIN_EPOCHS', '2'))
LR         = float(os.environ.get('TRAIN_LR', '1e-5'))
STALL_SEC  = int(os.environ.get('STALL_SEC', '1800'))
PREVIEW_ITERS = int(os.environ.get('PREVIEW_ITERS', '200'))   # первый предпросмотр после N итераций
PREVIEW_STOP  = os.environ.get('PREVIEW_STOP', '0') == '1'    # остановиться после первого предпросмотра
PHRASE     = os.environ.get('PHRASE', '')                     # тестовая фраза для озвучки (клон голоса)
XTTS_URL   = os.environ.get('XTTS_URL', 'http://195.209.214.155:9090')
PHRASE_AUD = None   # путь к синтезированной фразе (заполняет resolve_phrase_audio)
DATA_VERSION = 2   # версия формата обучающих данных (менялась — переобучаем)
SEG_SECONDS = 8       # длительность одного куска
SEG_FRAMES = 190      # сколько кадров в среднем даёт один кусок
FF = 'ffmpeg'

W2L_URL  = 'https://github.com/Rudrabha/Wav2Lip.git'
CKPT_URL = 'https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/wav2lip.pth'
S3FD_URL = 'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth'

MAIN_TID = None
HEART = None

HELPER_INFER = (
    "import torch, runpy, sys, os\n"
    "sys.path.insert(0, os.getcwd())\n"
    "ckpt, face, audio, out, bs = sys.argv[1:6]\n"
    "_tl = torch.load\n"
    "torch.load = lambda *a, **k: _tl(*a, **{**k, 'weights_only': False})\n"
    "sys.argv = ['i', '--checkpoint_path', ckpt, '--face', face, '--audio', audio, "
    "'--outfile', out, '--pads', '0', '20', '0', '0', '--wav2lip_batch_size', bs]\n"
    "runpy.run_path('inference.py', run_name='__main__')\n")


class PreviewStop(Exception):
    """Специальная остановка: первый предпросмотр готов, дальше не идём."""


# ─────────────────────────── журнал и состояние ───────────────────────────

def log(msg):
    line = time.strftime('%H:%M:%S') + ' | ' + str(msg)
    print(line, flush=True)
    try:
        os.makedirs(BASE, exist_ok=True)
        with open(LOGF, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        pass


def state():
    try:
        with open(STATEF, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(**kw):
    s = state()
    s.update(kw)
    try:
        with open(STATEF, 'w', encoding='utf-8') as f:
            json.dump(s, f, ensure_ascii=False, indent=1)
    except Exception:
        pass


# ─────────────────────────── утилиты ───────────────────────────

def run_cmd(cmd, timeout=1800, cwd=None):
    """Выполнить команду; зависание ограничено timeout (сек)."""
    return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                          timeout=timeout, cwd=cwd)


def run_stream(cmd, timeout=3600, cwd=None):
    """Как run_cmd, но вывод команды идёт в журнал вживую и обновляет watchdog —
    видно, что процесс реально движется, а зависание ловится честно."""
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, cwd=cwd)
    t0 = time.time()
    tail = []
    for line in p.stdout:
        line = line.rstrip()
        if line:
            beat(line.strip()[:60])
            log('   › ' + line[:160])
            tail.append(line)
            if len(tail) > 150:
                tail.pop(0)
        if time.time() - t0 > timeout:
            p.kill()
            raise RuntimeError('таймаут команды (%d с)' % timeout)
    rc = p.wait()

    class R:
        returncode = rc
        stdout = '\n'.join(tail)
        stderr = ''
    return R


def media_duration(path):
    """Длительность аудио/видео в секундах (None если не узнать)."""
    try:
        p = run_cmd('"%s" -i "%s"' % (FF, path), timeout=60)
        m = re.search(r'Duration: (\d+):(\d+):([\d.]+)', (p.stderr or ''))
        if m:
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    except Exception:
        pass
    return None


def free_gpu():
    """Освобождает видеопамять перед рендером, чтобы не было OOM."""
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log('GPU: кеш очищен, свободно %.1f ГБ' % (torch.cuda.mem_get_info()[0] / 2**30))
    except Exception as e:
        log('free_gpu: ' + repr(e))


def disk_ok(need_mb=500):
    free = shutil.disk_usage(BASE).free // (1024 * 1024)
    if free < need_mb:
        raise RuntimeError('мало места на диске: %d МБ (нужно >= %d МБ)' % (free, need_mb))
    return True


def du_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total // (1024 * 1024)


def disk_report(tag=''):
    try:
        u = shutil.disk_usage(BASE)
        log('💾 диск: свободно %.1f ГБ из %.1f ГБ%s' % (u.free / 2**30, u.total / 2**30, (' | ' + tag) if tag else ''))
        for name, p in (('сегменты', SEGD), ('кадры', DSD), ('папка STUDIO', BASE)):
            if os.path.isdir(p):
                log('   %s: %d МБ' % (name, du_mb(p)))
    except Exception as e:
        log('disk_report: ' + repr(e))


def need_segs():
    """Сколько кусков реально нужно под MAX_FRAMES (с запасом на брак)."""
    n = int((MAX_FRAMES + SEG_FRAMES - 1) // SEG_FRAMES) + 6
    return max(8, min(n, 40))


def download(url, dst, timeout=1800):
    if shutil.which('wget'):
        p = run_cmd('wget -q --tries=3 --timeout=120 "%s" -O "%s"' % (url, dst), timeout=timeout)
        if p.returncode == 0:
            return
        raise RuntimeError('wget ошибка: ' + (p.stderr or '')[-160:])
    import urllib.request
    urllib.request.urlretrieve(url, dst)


def resolve_ffmpeg():
    global FF
    p = shutil.which('ffmpeg')
    if p:
        FF = p
        return True
    try:
        import imageio_ffmpeg
        FF = imageio_ffmpeg.get_ffmpeg_exe()
        return True
    except Exception:
        return False


# ─────────────────────────── watchdog (сердцебиение) ───────────────────────────

class Heart:
    """Поток-наблюдатель: печатает статус каждые 30 с; если этап молчит дольше
    stall секунд — посылает KeyboardInterrupt в главный поток (перезапуск этапа)."""

    def __init__(self, stall):
        self.stage = 'init'
        self.msg = ''
        self.last = time.time()
        self.stall = stall
        self._inj = 0.0
        self.alive = True
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def beat(self, msg=None):
        self.last = time.time()
        if msg is not None:
            self.msg = msg

    def stop(self):
        self.alive = False

    def _loop(self):
        while self.alive:
            time.sleep(30)
            if not self.alive:
                break
            idle = int(time.time() - self.last)
            log('💓 жив | этап: %s | %s | %d с с последнего шага' % (self.stage, self.msg, idle))
            if idle > self.stall and time.time() - self._inj > 60 and MAIN_TID:
                self._inj = time.time()
                log('⚠️ WATCHDOG: шаг молчит > %d с — принудительно перезапускаю этап' % self.stall)
                try:
                    import ctypes
                    r = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        ctypes.c_long(MAIN_TID), ctypes.py_object(KeyboardInterrupt))
                    if r != 1:
                        log('watchdog: сигнал не доставлен (код %s)' % r)
                except Exception as e:
                    log('watchdog ошибка: ' + repr(e))


def beat(msg=None):
    if HEART is not None:
        HEART.beat(msg)


# ─────────────────────────── исполнитель этапов ───────────────────────────

def run_stage(name, done_fn, work_fn, verify_fn, attempts=3):
    """Этап: проверка «уже сделано?» → попытка → проверка результата.
    Возврат True только если результат ПРОВЕРЕН."""
    try:
        if done_fn():
            log('✅ этап «%s» уже выполнен ранее — пропускаю' % name)
            return True
    except Exception as e:
        log('проверка done_fn: ' + repr(e))
    for a in range(1, attempts + 1):
        if HEART:
            HEART.stage = name
            HEART.beat('попытка %d/%d' % (a, attempts))
        log('▶️ этап «%s» — попытка %d/%d' % (name, a, attempts))
        t0 = time.time()
        try:
            disk_ok(500)
            work_fn()
        except PreviewStop:
            raise
        except KeyboardInterrupt:
            log('⏹ этап «%s»: остановлено watchdog-ом (зависание) — буду повторять' % name)
        except Exception as e:
            log('❌ ошибка этапа «%s»: %s: %s' % (name, type(e).__name__, e))
            log(traceback.format_exc(limit=3))
        ok = False
        try:
            ok = bool(verify_fn())
        except Exception as e:
            log('ошибка проверки результата: ' + repr(e))
        if ok:
            log('✅ этап «%s» выполнен и ПРОВЕРЕН (%.1f мин)' % (name, (time.time() - t0) / 60.0))
            return True
        log('🔴 «%s»: проверка результата не пройдена — повтор через 10 с' % name)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        time.sleep(10)
    log('⛔ этап «%s»: не удался после %d попыток' % (name, attempts))
    return False


# ─────────────────────────── ЭТАП 1: setup ───────────────────────────

def setup_work():
    os.makedirs(SEGD, exist_ok=True)
    os.makedirs(DSD, exist_ok=True)
    ck = os.path.join(W2L, 'checkpoints')
    w0 = os.path.join(ck, 'wav2lip.pth')
    s0 = os.path.join(ck, 's3fd.pth')
    need_dl = not ((os.path.isfile(w0) and os.path.getsize(w0) >= 400000000)
                   and (os.path.isfile(s0) and os.path.getsize(s0) >= 80000000))
    if need_dl:
        disk_ok(1500)  # место нужно только если веса придётся скачивать
    if not resolve_ffmpeg():
        log('ffmpeg не найден — ставлю imageio-ffmpeg…')
        run_cmd('%s -m pip install -q imageio-ffmpeg' % sys.executable, timeout=600)
        if not resolve_ffmpeg():
            raise RuntimeError('ffmpeg недоступен')
    log('ffmpeg: ' + FF)
    if not os.path.isfile(os.path.join(W2L, 'inference.py')):
        if os.path.isdir(W2L):
            shutil.rmtree(W2L, ignore_errors=True)
        beat('клонирование Wav2Lip')
        run_cmd('git clone -q %s %s' % (W2L_URL, W2L), timeout=900)
        if not os.path.isfile(os.path.join(W2L, 'inference.py')):
            raise RuntimeError('Wav2Lip не склонировался')
    ck = os.path.join(W2L, 'checkpoints')
    os.makedirs(ck, exist_ok=True)
    w = os.path.join(ck, 'wav2lip.pth')
    s = os.path.join(ck, 's3fd.pth')
    if not (os.path.isfile(w) and os.path.getsize(w) >= 400000000):
        for i in range(5):
            beat('скачивание wav2lip.pth (попытка %d)' % (i + 1))
            log('скачиваю wav2lip.pth (попытка %d)…' % (i + 1))
            try:
                download(CKPT_URL, w, timeout=1800)
            except Exception as e:
                log('скачивание: ' + str(e))
            if os.path.isfile(w) and os.path.getsize(w) >= 400000000:
                break
    if not (os.path.isfile(s) and os.path.getsize(s) >= 80000000):
        for i in range(5):
            beat('скачивание s3fd.pth (попытка %d)' % (i + 1))
            try:
                download(S3FD_URL, s, timeout=900)
            except Exception as e:
                log('скачивание: ' + str(e))
            if os.path.isfile(s) and os.path.getsize(s) >= 80000000:
                break
    shutil.copy(s, os.path.join(W2L, 'face_detection', 'detection', 'sfd', 's3fd.pth'))
    ap = os.path.join(W2L, 'audio.py')
    src = open(ap, encoding='utf-8').read()
    if 'librosa.filters.mel(hp.sample_rate' in src:
        src = src.replace('librosa.filters.mel(hp.sample_rate, hp.n_fft,',
                          'librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft,')
        open(ap, 'w', encoding='utf-8').write(src)
        log('audio.py пропатчен под librosa 0.10+')


def setup_verify():
    if not resolve_ffmpeg():
        log('⚠️ проверка setup: нет ffmpeg')
        return False
    w = os.path.join(W2L, 'checkpoints', 'wav2lip.pth')
    s = os.path.join(W2L, 'checkpoints', 's3fd.pth')
    if not (os.path.isfile(w) and os.path.getsize(w) >= 400000000):
        log('⚠️ проверка setup: wav2lip.pth отсутствует/неполный')
        return False
    if not (os.path.isfile(s) and os.path.getsize(s) >= 80000000):
        log('⚠️ проверка setup: s3fd.pth отсутствует/неполный')
        return False
    if not os.path.isfile(os.path.join(W2L, 'inference.py')):
        log('⚠️ проверка setup: нет inference.py')
        return False
    code = ('import sys; sys.path.insert(0, %r); '
            'from models.wav2lip import Wav2Lip; '
            'from face_detection import FaceAlignment; '
            "print('IMPORTS_OK')") % W2L
    p = run_cmd('%s -c "%s"' % (sys.executable, code), timeout=300)
    if 'IMPORTS_OK' not in (p.stdout or ''):
        log('⚠️ проверка setup: импорты не работают: ' + (p.stderr or '')[-300:])
        return False
    return True


# ─────────────────────────── ЭТАП 2: dataset ───────────────────────────

def ensure_src():
    if os.path.isfile(SRCV) and os.path.getsize(SRCV) > 1000000:
        return SRCV
    cand = VIDEO if os.path.isfile(VIDEO) else None
    if cand is None:
        for c in ('/kaggle/input/datasets/alexeyms/mariairkhina/Maria.mp4',):
            if os.path.isfile(c):
                cand = c
                break
    if cand is None:
        raise RuntimeError('исходное видео НЕ найдено: %s — задайте MARIA_VIDEO' % VIDEO)
    try:
        if os.path.islink(SRCV):
            os.remove(SRCV)
        os.symlink(cand, SRCV)
    except Exception:
        shutil.copy(cand, SRCV)
    log('исходное видео: ' + cand)
    return SRCV


def seg_list():
    return sorted(glob.glob(os.path.join(SEGD, 'seg_*.avi')))


def seg_status():
    """Возвращает ('ok'|'partial'|'broken'|'empty', список, битые файлы).
    'ok' — все нарезано, ничего резать НЕ надо."""
    ss = seg_list()
    if not ss:
        return 'empty', [], []
    names = [os.path.basename(s) for s in ss]
    n = len(names)
    contiguous = all(names[i] == 'seg_%03d.avi' % i for i in range(n))
    if not contiguous:
        return 'broken', ss, []
    bad = [s for s in ss[:-1] if os.path.getsize(s) < 100000]
    return ('ok' if not bad else 'partial'), ss, bad


def _seg_index(path):
    return int(os.path.basename(path)[4:7])


def trim_segs(need):
    """Убирает куски сверх нужного количества (и их кадры) — освобождает место,
    НЕ трогая нужные куски."""
    ss = seg_list()
    if len(ss) <= need:
        return
    removed_mb = 0
    for s in ss[need:]:
        sid = os.path.splitext(os.path.basename(s))[0]
        d = os.path.join(DSD, sid)
        if os.path.isdir(d):
            removed_mb += du_mb(d)
            shutil.rmtree(d, ignore_errors=True)
        try:
            removed_mb += os.path.getsize(s) // (1024 * 1024)
            os.remove(s)
        except OSError:
            pass
    log('💾 удалил лишних кусков: %d (оставил %d из %d), освобождено ~%d МБ'
        % (len(ss) - need, need, len(ss), removed_mb))


def _cut_cmd(src, dst, start=None, total_sec=None):
    """Команда ffmpeg: компактные куски (720p, q4) — в разы меньше места."""
    scale = '-vf "scale=min(1280,iw):-2"'
    ss_part = ('-ss %d ' % start) if start is not None else ''
    t_part = ('-t %d ' % total_sec) if total_sec is not None else ''
    return ('"%s" -y -loglevel error %s-i "%s" %s-r 25 %s -c:v mjpeg -q:v 4 %s'
            % (FF, ss_part, src, t_part, scale, dst))


def cut_segments(src):
    """Главное правило: если куски уже нарезаны — НИЧЕГО не режем заново,
    только убираем лишнее. Чиним точечно битые; полная перенарезка — лишь когда
    всё сломано (и тоже только нужное число кусков)."""
    need = need_segs()
    st, ss, bad = seg_status()
    if len(ss) >= need:
        bad_need = [b for b in bad if _seg_index(b) < need]
        if st == 'ok' or not bad_need:
            trim_segs(need)
            log('сегменты уже нарезаны (%d шт, нужно %d) — пропускаю нарезку' % (len(seg_list()), need))
            return
        log('сегменты нарезаны, битых среди нужных: %d — чиню ТОЛЬКО их' % len(bad_need))
        for b in bad_need:
            idx = _seg_index(b)
            beat('починка сегмента %d' % idx)
            try:
                os.remove(b)
            except OSError:
                pass
            run_cmd(_cut_cmd(src, '"%s"' % b, start=idx * SEG_SECONDS, total_sec=SEG_SECONDS), timeout=600)
        if seg_status()[0] == 'ok':
            trim_segs(need)
            log('сегменты готовы (%d шт), перенарезка не понадобилась' % len(seg_list()))
            return
        log('точечная починка не помогла — перережу только нужные %d кусков' % need)
    for s in seg_list():
        try:
            os.remove(s)
        except OSError:
            pass
    log('режу только первые %d кусков (%d сек видео) — этого хватит на %d кадров'
        % (need, need * SEG_SECONDS, MAX_FRAMES))
    beat('ffmpeg: нарезка %d сегментов' % need)
    dst = os.path.join(SEGD, 'seg_%03d.avi')
    p = run_cmd(_cut_cmd(src, '"%s"' % dst, total_sec=need * SEG_SECONDS), timeout=1800)
    if p.returncode != 0 or seg_status()[0] != 'ok':
        raise RuntimeError('нарезка сегментов не удалась: ' + (p.stderr or '')[-200:])
    log('нарезано сегментов: %d' % len(seg_list()))


def count_frames():
    n = 0
    for d in glob.glob(DSD + '/*'):
        if os.path.isdir(d):
            n += len(glob.glob(os.path.join(d, '*.jpg')))
    return n


def dataset_done():
    n = count_frames()
    if n >= MAX_FRAMES:
        return True
    ss = seg_list()
    if not ss:
        return False
    marked = 0
    for s in ss:
        sid = os.path.splitext(os.path.basename(s))[0]
        fdir = os.path.join(DSD, sid)
        if os.path.isfile(os.path.join(fdir, '.ok')) or os.path.isfile(os.path.join(fdir, '.bad')):
            marked += 1
    return marked >= len(ss) and n >= MIN_FRAMES


def pending_segs():
    """Сегменты, у которых ещё нет метки .ok/.bad — т.е. реально осталась работа."""
    out = []
    for seg in seg_list():
        sid = os.path.splitext(os.path.basename(seg))[0]
        fdir = os.path.join(DSD, sid)
        if os.path.isfile(os.path.join(fdir, '.ok')) or os.path.isfile(os.path.join(fdir, '.bad')):
            continue
        out.append(seg)
    return out


def _ensure_audio(seg, fdir):
    aw = os.path.join(fdir, 'audio.wav')
    if os.path.isfile(aw) and os.path.getsize(aw) > 1000:
        return True
    p = run_cmd('"%s" -y -loglevel error -i "%s" -ar 16000 -ac 1 "%s"' % (FF, seg, aw), timeout=300)
    return p.returncode == 0 and os.path.isfile(aw) and os.path.getsize(aw) > 1000


def frames_work():
    src = ensure_src()
    cut_segments(src)
    # Проверка ДО тяжёлых импортов: если всё уже готово — ничего не запускаем.
    if dataset_done():
        log('датасет уже собран (%d кадров) — разбор кадров пропускаю' % count_frames())
        return
    pend = pending_segs()
    if not pend:
        log('все сегменты уже обработаны ранее')
        return
    import numpy as np
    import cv2
    sys.path.insert(0, W2L)
    import torch
    disk_ok(1000)
    from face_detection import FaceAlignment, LandmarksType
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    log('детектор лиц на устройстве: ' + dev)
    fa = FaceAlignment(LandmarksType._2D, flip_input=False, device=dev)
    total = count_frames()
    log('кадров уже есть: %d, цель: %d, сегментов в очереди: %d' % (total, MAX_FRAMES, len(pend)))
    for seg in pend:
        if total >= MAX_FRAMES:
            break
        sid = os.path.splitext(os.path.basename(seg))[0]
        fdir = os.path.join(DSD, sid)
        os.makedirs(fdir, exist_ok=True)
        beat('сегмент ' + sid)
        try:
            cap = cv2.VideoCapture(seg)
            frames = []
            while True:
                okk, fr = cap.read()
                if not okk:
                    break
                frames.append(fr)
            cap.release()
            if len(frames) < 25:
                open(os.path.join(fdir, '.bad'), 'w').write('короткий')
                continue
            have = len(glob.glob(os.path.join(fdir, '*.jpg')))
            if have >= len(frames):
                # сегмент уже был разобран целиком, просто не помечен
                if not _ensure_audio(seg, fdir):
                    raise RuntimeError('аудио не извлечено')
                open(os.path.join(fdir, '.ok'), 'w').write(str(len(frames)))
                log('🟢 сегмент %s уже был готов (%d кадров) — пометил, не переделываю' % (sid, have))
                continue
            if have > 0:
                log('🟢 сегмент %s: продолжаю с кадра %d/%d (готовое не трогаю)' % (sid, have, len(frames)))
            last = None
            for i in range(have, len(frames)):
                fr = frames[i]
                if total >= MAX_FRAMES:
                    break
                if i % 5 == 0 or last is None:
                    beat('%s: кадр %d/%d' % (sid, i, len(frames)))
                    preds = fa.get_detections_for_batch(np.array([fr]))
                    det = preds[0] if len(preds) else None
                    if det is None or len(det) == 0:
                        last = None
                    else:
                        det = np.asarray(det)
                        if det.ndim == 1:
                            det = det[None, :]
                        last = det[0][:4].astype(int)
                if last is not None:
                    x1, y1, x2, y2 = last
                    pad = int((y2 - y1) * 0.25)
                    y1 = max(0, y1 - pad)
                    y2 = min(fr.shape[0], y2 + int(pad * 1.6))
                    x1 = max(0, x1 - pad)
                    x2 = min(fr.shape[1], x2 + pad)
                    crop = cv2.resize(fr[y1:y2, x1:x2], (96, 96))
                else:
                    h, w = fr.shape[:2]
                    crop = cv2.resize(fr[int(h * 0.1):int(h * 0.7), int(w * 0.25):int(w * 0.75)], (96, 96))
                cv2.imwrite(os.path.join(fdir, '%05d.jpg' % i), crop)
                total += 1
                if total % 100 == 0:
                    log('🟢 кадры: %d/%d (сейчас %s, кадр %d/%d)' % (total, MAX_FRAMES, sid, i, len(frames)))
                    disk_ok(500)
            if not _ensure_audio(seg, fdir):
                raise RuntimeError('аудио не извлечено')
            open(os.path.join(fdir, '.ok'), 'w').write(str(len(frames)))
        except Exception as e:
            log('⚠️ сегмент %s: %s: %s — пропускаю сегмент' % (sid, type(e).__name__, e))
            try:
                open(os.path.join(fdir, '.bad'), 'w').write(str(e)[:100])
            except Exception:
                pass
            continue
    log('кадров собрано: %d' % count_frames())


# ─────────────────────────── ЭТАП 3: train ───────────────────────────

def crop_win(spec, frame_id):
    """Окно мел-спектрограммы (16, 80) под номер кадра.
    Если аудио короче видео (край клипа) — дополняет нулями до полного окна,
    чтобы батч всегда собирался."""
    import numpy as np
    s0 = max(0, int(80.0 * (frame_id / 25.0)))
    win = spec[s0:s0 + 16, :]
    if win.shape[0] >= 16:
        return win
    out = np.zeros((16, 80), dtype=np.float32)
    if win.size:
        out[:win.shape[0], :win.shape[1]] = win
    return out


def clips_ok():
    return [d for d in sorted(glob.glob(DSD + '/*'))
            if os.path.isdir(d) and not os.path.isfile(os.path.join(d, '.bad'))]


def sanitize_dataset():
    removed = 0
    for d in sorted(glob.glob(DSD + '/*')):
        if not os.path.isdir(d):
            continue
        if os.path.isfile(os.path.join(d, '.bad')):
            continue
        jpgs = glob.glob(os.path.join(d, '*.jpg'))
        aw = os.path.join(d, 'audio.wav')
        if len(jpgs) < 10:
            open(os.path.join(d, '.bad'), 'w').write('мало кадров')
            removed += 1
            continue
        if not (os.path.isfile(aw) and os.path.getsize(aw) > 1000):
            sid = os.path.basename(d)
            seg = os.path.join(SEGD, sid + '.avi')
            okf = False
            if os.path.isfile(seg):
                p = run_cmd('"%s" -y -loglevel error -i "%s" -ar 16000 -ac 1 "%s"' % (FF, seg, aw), timeout=300)
                okf = (p.returncode == 0 and os.path.isfile(aw) and os.path.getsize(aw) > 1000)
            if not okf:
                open(os.path.join(d, '.bad'), 'w').write('нет аудио')
                removed += 1
    if removed:
        log('sanitize: исключено битых клипов: %d' % removed)


def train_work():
    import numpy as np
    import cv2
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    sys.path.insert(0, W2L)
    from models.wav2lip import Wav2Lip
    sanitize_dataset()
    clips = clips_ok()
    if len(clips) < 5:
        raise RuntimeError('слишком мало хороших клипов: %d' % len(clips))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    log('обучение на устройстве: ' + dev)

    class ClipDS(Dataset):
        def __init__(self, ds, T=5):
            self.T = T
            self.clips = [d for d in sorted(glob.glob(ds + '/*'))
                          if os.path.isdir(d) and not os.path.isfile(os.path.join(d, '.bad'))]
            import audio as A
            if not getattr(A, '_mel_kw', False):
                import librosa as _lb

                def _mb():
                    return _lb.filters.mel(sr=A.hp.sample_rate, n_fft=A.hp.n_fft,
                                           n_mels=A.hp.num_mels, fmin=A.hp.fmin, fmax=A.hp.fmax)
                A._build_mel_basis = _mb
                A._mel_kw = True
            self.A = A
            self.mels = {}
            self.files = {}
            for c in self.clips:
                wav = A.load_wav(os.path.join(c, 'audio.wav'), 16000)
                self.mels[c] = A.melspectrogram(wav).T
                self.files[c] = sorted(glob.glob(os.path.join(c, '*.jpg')))

        def __len__(self):
            return sum(max(1, len(self.files[c]) - self.T - 2) for c in self.clips)

        def __getitem__(self, idx):
            c = self.clips[idx % len(self.clips)]
            fs = self.files[c]
            T = self.T
            hi = max(3, len(fs) - T)
            i = int(np.random.randint(2, hi))
            w = int(np.random.randint(2, hi))

            def prep(ids):
                x = np.asarray([cv2.imread(fs[j]) for j in ids]) / 255.0
                x = np.transpose(x, (3, 0, 1, 2))
                return x

            # ВАЖНО (как в официальном wav2lip_train.py): зануляется нижняя половина
            # ТОЛЬКО первого окна; второе («неверное») окно остаётся целым —
            # это опорная картинка, модель без неё даёт артефакты.
            xw = prep(range(i, i + T))
            xw[:, :, 48:, :] = 0.0
            y = np.asarray([cv2.imread(fs[j]) for j in range(i, i + T)]) / 255.0
            y = np.transpose(y, (3, 0, 1, 2))
            xw2 = prep(range(w, w + T))
            x = np.concatenate([xw, xw2], axis=0)
            spec = self.mels[c]
            mel = crop_win(spec, i)
            indiv = np.asarray([crop_win(spec, j - 2).T for j in range(i + 1, i + 1 + T)])
            return (torch.FloatTensor(x), torch.FloatTensor(indiv).unsqueeze(1),
                    torch.FloatTensor(mel.T).unsqueeze(0), torch.FloatTensor(y))

    prog = state().get('train', {})
    start_e = int(prog.get('epoch', 0))
    skip_it = int(prog.get('iter', 0))
    if not os.path.isfile(LASTP) or state().get('data_v') != DATA_VERSION:
        start_e, skip_it = 0, 0
    model = Wav2Lip().to(dev)
    src_ckpt = LASTP if (start_e > 0 or skip_it > 0) else os.path.join(W2L, 'checkpoints', 'wav2lip.pth')
    log('стартовые веса: %s (эпоха %d, итерация %d)' % (os.path.basename(src_ckpt), start_e, skip_it))
    ck = torch.load(src_ckpt, map_location=dev, weights_only=False)
    sd = ck.get('state_dict', ck)
    if any(k.startswith('module.') for k in sd):
        sd = {k[7:]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    ds = ClipDS(DSD)
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for e in range(start_e, EPOCHS):
        it = 0
        run_it = 0
        t_it = time.time()
        last_loss = 0.0
        for x_t, indiv_t, mel_t, y_t in dl:
            it += 1
            if e == start_e and it <= skip_it:
                continue
            x_t, indiv_t, y_t = x_t.to(dev), indiv_t.to(dev), y_t.to(dev)
            g = model(indiv_t, x_t)
            loss = nn.L1Loss()(g, y_t)
            lv = float(loss.item())
            if not (lv == lv) or lv > 10:
                raise RuntimeError('loss сошёл с ума: %s — перезапускаю обучение' % lv)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_it += 1
            last_loss = lv
            if run_it % 10 == 0:
                beat('эпоха %d/%d, итерация %d, loss=%.4f' % (e + 1, EPOCHS, run_it, lv))
            if run_it % 100 == 0:
                spd = (time.time() - t_it) / 100.0
                t_it = time.time()
                log('🟢 обучение: эпоха %d/%d, итерация %d, loss=%.4f (%.0f мс/ит)'
                    % (e + 1, EPOCHS, run_it, lv, spd * 1000.0))
                disk_ok(500)
            if run_it % 500 == 0:
                torch.save({'state_dict': model.state_dict()}, LASTP)
                save_state(train={'epoch': e, 'iter': it})
            if run_it == PREVIEW_ITERS:
                # первый предпросмотр: модель сохраняем, рендерим 5 сек, обучение потом продолжится
                torch.save({'state_dict': model.state_dict()}, LASTP)
                save_state(train={'epoch': e, 'iter': it})
                pv = make_preview(LASTP, 'iter%d' % run_it)
                if PREVIEW_STOP and pv:
                    raise PreviewStop()
        torch.save({'state_dict': model.state_dict()}, CKPT)
        save_state(train={'epoch': e + 1, 'iter': 0})
        log('🟢 эпоха %d/%d завершена, loss=%.4f' % (e + 1, EPOCHS, last_loss))
        pv = make_preview(CKPT, 'epoch%d' % (e + 1))
        if PREVIEW_STOP and pv:
            raise PreviewStop()
    open(FLAG, 'w').write('ok')
    save_state(data_v=DATA_VERSION)
    log('обучение завершено, модель сохранена: ' + CKPT)
    try:
        del model, opt, ds, dl
    except Exception:
        pass
    free_gpu()


def train_verify():
    if not (os.path.isfile(FLAG) and os.path.isfile(CKPT)):
        return False
    if state().get('data_v') != DATA_VERSION:
        log('⚠️ формат обучающих данных изменился (v%d) — требуется переобучение' % DATA_VERSION)
        return False
    import torch
    sys.path.insert(0, W2L)
    from models.wav2lip import Wav2Lip
    m = Wav2Lip()
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    sd = ck.get('state_dict', ck)
    m.load_state_dict(sd)
    return True


# ─────────────────────────── ЭТАП 4: generate ───────────────────────────

def xtts_synth(text, voice_ref, out, timeout=30):
    """Синтез фразы клонированным голосом через HTTP-сервис XTTS пользователя.
    Пробует несколько форматов API. True — аудио записано в out."""
    import urllib.request
    import json as _json
    import uuid as _uuid

    def post_json(u, payload):
        req = urllib.request.Request(u, data=_json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()

    def post_multipart(u, fields, filepath):
        boundary = _uuid.uuid4().hex
        body = b''
        for k, v in fields.items():
            body += ('--%s\r\nContent-Disposition: form-data; name="%s"\r\n\r\n%s\r\n'
                     % (boundary, k, v)).encode()
        fn = os.path.basename(filepath)
        body += ('--%s\r\nContent-Disposition: form-data; name="voice_file"; '
                 'filename="%s"; Content-Type: audio/wav\r\n\r\n' % (boundary, fn)).encode()
        body += open(filepath, 'rb').read() + b'\r\n--%s--\r\n' % boundary.encode()
        req = urllib.request.Request(u, data=body,
                                     headers={'Content-Type': 'multipart/form-data; boundary=%s' % boundary})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()

    def looks_audio(b):
        return len(b) > 8000 and (b[:4] == b'RIFF' or b[:3] == b'ID3' or b[:2] == b'\xff\xfb')

    strategies = [
        ('multipart /tts_with_voice_clone',
         lambda: post_multipart(XTTS_URL + '/tts_with_voice_clone', {'text': text, 'language': 'ru'}, voice_ref)),
        ('json /tts_with_voice_clone',
         lambda: post_json(XTTS_URL + '/tts_with_voice_clone', {'text': text, 'language': 'ru', 'voice_file': voice_ref})),
        ('json /v1/tts_with_voice_clone',
         lambda: post_json(XTTS_URL + '/v1/tts_with_voice_clone', {'text': text, 'language': 'ru', 'voice_file': voice_ref})),
        ('json /tts_to_speaker',
         lambda: post_json(XTTS_URL + '/tts_to_speaker', {'text': text, 'language': 'ru', 'speaker_id': 'xtts_ru'})),
    ]
    for name, fn in strategies:
        try:
            beat('XTTS: ' + name)
            status, body = fn()
            if status == 200 and looks_audio(body):
                open(out, 'wb').write(body)
                log('🎙 фраза синтезирована (' + name + ')')
                return True
            if status == 200:
                try:
                    d = _json.loads(body)
                    if isinstance(d, dict) and 'audio' in d:
                        import base64
                        open(out, 'wb').write(base64.b64decode(d['audio']))
                        log('🎙 фраза синтезирована (' + name + ', base64)')
                        return True
                except Exception:
                    pass
        except Exception as e:
            log('⚠️ xtts ' + name + ': ' + repr(e))
    return False


def xtts_local_synth(text, voice_ref, out):
    """Клонирование голоса ЛОКАЛЬНО на этой машине (Coqui XTTS v2).
    TTS 0.22 не ставится на python>=3.12, поэтому разворачиваем изолированный
    python 3.11 + venv (одноразово, кешируется). True — аудио записано в out."""
    try:
        env_py = os.path.join(BASE, 'xtts_env', 'bin', 'python')
        if not os.path.isfile(env_py):
            log('🎙 одноразовая установка: python 3.11 + XTTS v2 (~3–5 мин)')
            pydir = os.path.join(BASE, 'py311')
            pybin = os.path.join(pydir, 'bin', 'python3.11')
            if not os.path.isfile(pybin):
                tar = os.path.join(BASE, 'py311.tar.gz')
                download('https://github.com/indygreg/python-build-standalone/releases/'
                         'download/20240415/cpython-3.11.9+20240415-x86_64-unknown-linux-gnu-install_only.tar.gz',
                         tar, timeout=900)
                run_cmd('mkdir -p "%s" && tar -xzf "%s" -C "%s" --strip-components=1'
                        % (pydir, tar, pydir), timeout=600)
                try:
                    os.remove(tar)
                except OSError:
                    pass
            if not os.path.isfile(pybin):
                raise RuntimeError('python3.11 не развернулся')
            beat('XTTS: создаю venv')
            run_cmd('"%s" -m venv "%s"' % (pybin, os.path.join(BASE, 'xtts_env')), timeout=600)
            beat('XTTS: ставлю torch+TTS в venv')
            p = run_cmd('"%s" -m pip install -q torch==2.2.2+cu121 torchaudio==2.2.2+cu121 '
                        '--index-url https://download.pytorch.org/whl/cu121 && '
                        '"%s" -m pip install -q TTS==0.22.0' % (env_py, env_py), timeout=2400)
            if p.returncode != 0:
                log('⚠️ pip install XTTS: ' + ((p.stdout or '') + (p.stderr or ''))[-300:])
                raise RuntimeError('pip install TTS не удался')
        synth = os.path.join(BASE, 'xtts_synth.py')
        open(synth, 'w', encoding='utf-8').write(SYNTH_PY)
        log('🎙 синтезирую фразу локальным XTTS v2…')
        p = run_stream('"%s" "%s" "%s" "%s" "%s"' % (env_py, synth, text, voice_ref, out), timeout=1800)
        okk = os.path.isfile(out) and os.path.getsize(out) > 8000
        if okk:
            log('🎙 фраза синтезирована локальным XTTS v2')
        else:
            log('⚠️ XTTS не выдал аудио: ' + (p.stdout or '')[-300:])
        return okk
    except Exception as e:
        log('⚠️ локальный XTTS: %r' % e)
        return False


SYNTH_PY = (
    "import sys, torch\n"
    "from TTS.api import TTS\n"
    "text, ref, out = sys.argv[1:4]\n"
    "dev = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
    "print('XTTS: device', dev, flush=True)\n"
    "tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2').to(dev)\n"
    "print('XTTS: model loaded', flush=True)\n"
    "tts.tts_to_file(text=text, speaker_wav=ref, language='ru', file_path=out)\n"
    "print('XTTS: done', flush=True)\n")


def xtts_server_alive():
    """Быстрый пинг HTTP-сервиса (5 сек). Если молчит — HTTP-стратегии вообще не пробуем."""
    import urllib.request
    try:
        with urllib.request.urlopen(XTTS_URL + '/', timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def resolve_phrase_audio():
    """Готовит аудио тестовой фразы (клон голоса Марии). Приоритеты:
    кеш → HTTP-сервис (если живой) → локальный XTTS на этой машине → None (фолбэк)."""
    global PHRASE_AUD
    if not PHRASE:
        return None
    want = os.path.join(BASE, 'phrase_audio.wav')
    if state().get('phrase') == PHRASE and os.path.isfile(want) and os.path.getsize(want) > 8000:
        PHRASE_AUD = want
        log('🎙 фраза взята из кеша: ' + want)
        return want
    voice_ref = os.path.join(BASE, 'voice_ref.wav')
    if not (os.path.isfile(voice_ref) and os.path.getsize(voice_ref) > 8000):
        ref_src = os.path.join(BASE, 'golos_iz_video.wav')
        if not os.path.isfile(ref_src):
            ref_src = resolve_audio()
        run_cmd('"%s" -y -loglevel error -i "%s" -t 15 -ar 16000 -ac 1 "%s"'
                % (FF, ref_src, voice_ref), timeout=300)
    log('🎙 синтезирую фразу клонированным голосом: "%s"' % PHRASE)
    if xtts_server_alive():
        if xtts_synth(PHRASE, voice_ref, want, timeout=30):
            save_state(phrase=PHRASE)
            PHRASE_AUD = want
            return want
    else:
        log('🎙 XTTS-сервер не отвечает — пропускаю HTTP, иду сразу в локальный XTTS')
    log('🎙 запускаю локальный XTTS v2 (3–6 минут при первой установке)…')
    if xtts_local_synth(PHRASE, voice_ref, want):
        save_state(phrase=PHRASE)
        PHRASE_AUD = want
        return want
    log('⚠️ фразу синтезировать не удалось — беру исходное аудио из видео')
    return None


def run_w2l(ckpt, base, aud, out_raw):
    """Официальный inference.py с батчами 8/4/2/1 и живым выводом.
    Перед стартом убивает зависшие старые рендеры и чистит видеопамять,
    чтобы не было OOM. True, если out_raw готов."""
    # добиваем старые зависшие рендеры (если остались от прошлых запусков)
    run_cmd('pkill -f w2l_infer.py || true', timeout=30)
    free_gpu()
    infer_py = os.path.join(BASE, 'w2l_infer.py')
    open(infer_py, 'w').write(HELPER_INFER)
    if os.path.isfile(out_raw):
        try:
            os.remove(out_raw)
        except OSError:
            pass
    for bs in (8, 4, 2, 1):
        beat('инференс, батч %d' % bs)
        log('🟢 рендер губ: батч %d' % bs)
        try:
            p = run_stream('cd "%s" && OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 "%s" "%s" "%s" "%s" "%s" "%s" %d'
                           % (W2L, sys.executable, infer_py, ckpt, base, aud, out_raw, bs), timeout=3600)
            errtail = p.stdout[-300:]
        except Exception as e:
            errtail = repr(e)
        if os.path.isfile(out_raw) and os.path.getsize(out_raw) > 200000:
            return True
        log('⚠️ батч %d не дал результата: %s' % (bs, errtail.replace('\n', ' ')))
        if os.path.isfile(out_raw):
            try:
                os.remove(out_raw)
            except OSError:
                pass
    return False


def make_preview(ckpt, tag):
    """Быстрый предпросмотр (первые 5 секунд), чтобы оценить результат СРАЗУ,
    не дожидаясь конца обучения. Сбой предпросмотра не останавливает обучение."""
    try:
        base_full = resolve_base()
        pb = os.path.join(BASE, 'preview_base.mp4')
        if not (os.path.isfile(pb) and os.path.getsize(pb) > 50000):
            p = run_cmd('"%s" -y -loglevel error -i "%s" -t 5 -c:v libx264 -pix_fmt yuv420p -an "%s"'
                        % (FF, base_full, pb), timeout=600)
            if p.returncode != 0 or not os.path.isfile(pb):
                log('⚠️ предпросмотр: не удалось подготовить 5-сек видео')
                return None
        # предпросмотр — СРАЗУ с тестовой фразой, если она готова
        aud_full = PHRASE_AUD if (PHRASE_AUD and os.path.isfile(PHRASE_AUD)) else resolve_audio()
        pa = os.path.join(BASE, 'preview_audio.wav')
        t_part = '' if aud_full == PHRASE_AUD and PHRASE_AUD else '-t 5 '
        p = run_cmd('"%s" -y -loglevel error -i "%s" %s-ar 16000 -ac 1 "%s"'
                    % (FF, aud_full, t_part, pa), timeout=300)
        if p.returncode != 0 or not os.path.isfile(pa):
            log('⚠️ предпросмотр: не удалось подготовить аудио')
            return None
        if aud_full != resolve_audio():
            log('👀 предпросмотр озвучен тестовой фразой')
        raw = os.path.join(BASE, 'preview_raw.mp4')
        if not run_w2l(ckpt, pb, pa, raw):
            log('⚠️ предпросмотр не получился — обучение продолжается')
            return None
        out = os.path.join(BASE, 'preview_%s.mp4' % tag)
        p = run_cmd('"%s" -y -loglevel error -i "%s" -vf "format=yuv420p" -c:v libx264 '
                    '-crf 18 -movflags +faststart -c:a aac -b:a 128k "%s"' % (FF, raw, out), timeout=900)
        if p.returncode != 0 or not os.path.isfile(out):
            log('⚠️ предпросмотр: сборка mp4 не удалась')
            return None
        shutil.copy(out, os.path.join(BASE, 'preview_latest.mp4'))
        log('👀 ПРЕДПРОСМОТР ГОТОВ (%s): %s — скачайте и посмотрите прямо сейчас' % (tag, out))
        return out
    except Exception as e:
        log('⚠️ предпросмотр: %s: %s — обучение продолжается' % (type(e).__name__, e))
        return None


def resolve_base():
    b = os.path.join(BASE, 'base_last.mp4')
    if os.path.isfile(b) and os.path.getsize(b) > 100000:
        return b
    ss = seg_list()
    if not ss:
        raise RuntimeError('нет сегментов для базового видео')
    out = os.path.join(BASE, 'base_auto.mp4')
    p = run_cmd('"%s" -y -loglevel error -i "%s" -c:v libx264 -pix_fmt yuv420p -an "%s"'
                % (FF, ss[0], out), timeout=600)
    if p.returncode != 0 or not os.path.isfile(out):
        raise RuntimeError('базовое видео не подготовлено')
    log('базовое видео: первый сегмент ' + os.path.basename(ss[0]))
    return out


def resolve_audio():
    for c in ('audio_last.wav', 'golos_iz_video.wav', 'audio_auto.wav'):
        p = os.path.join(BASE, c)
        if os.path.isfile(p) and os.path.getsize(p) > 10000:
            return p
    out = os.path.join(BASE, 'audio_auto.wav')
    src = SRCV if os.path.isfile(SRCV) else VIDEO
    p = run_cmd('"%s" -y -loglevel error -i "%s" -t 15 -vn -ar 16000 -ac 1 "%s"'
                % (FF, src, out), timeout=300)
    if p.returncode != 0 or not os.path.isfile(out):
        raise RuntimeError('аудио не подготовлено')
    log('аудио: первые 15 секунд исходного видео')
    return out


def gen_work():
    base = resolve_base()
    aud_raw = None
    if PHRASE:
        if PHRASE_AUD is None:
            resolve_phrase_audio()
        aud_raw = PHRASE_AUD
    if aud_raw is None:
        aud_raw = resolve_audio()
    # Длина финального видео в Wav2Lip = длине аудио. Если аудио длиннее базового
    # видео — лицо начнёт «заикаться» и рендер раздувается. Режем аудио под видео.
    bdur = media_duration(base)
    adur = media_duration(aud_raw)
    log('базовое видео: %.1f сек | аудио: %s сек'
        % (bdur if bdur else -1, ('%.1f' % adur) if adur else '?'))
    aud = aud_raw
    if bdur and adur and adur > bdur + 0.5:
        aud = os.path.join(BASE, 'gen_audio_trimmed.wav')
        p = run_cmd('"%s" -y -loglevel error -i "%s" -t %.2f -ar 16000 -ac 1 "%s"'
                    % (FF, aud_raw, bdur, aud), timeout=300)
        if p.returncode != 0 or not os.path.isfile(aud):
            raise RuntimeError('не удалось подрезать аудио под длину видео')
        log('аудио подрезано до %.1f сек (под длину базового видео)' % bdur)
    if not train_verify():
        raise RuntimeError('персональная модель не грузится — нужен этап train')
    if not run_w2l(CKPT, base, aud, RAWV):
        raise RuntimeError('синхронизация не получилась ни с одним батчем')
    if os.path.isfile(FINV):
        os.remove(FINV)
    p = run_cmd('"%s" -y -loglevel error -i "%s" -vf "format=yuv420p" -c:v libx264 '
                '-crf 18 -movflags +faststart -c:a aac -b:a 128k "%s"' % (FF, RAWV, FINV), timeout=1800)
    if p.returncode != 0 or not os.path.isfile(FINV):
        raise RuntimeError('сборка финального видео не удалась')
    if os.path.isfile(RAWV):
        try:
            os.remove(RAWV)
            log('💾 временный raw-файл удалён (освобождено место)')
        except OSError:
            pass
    save_state(phrase_used=PHRASE if (PHRASE and PHRASE_AUD) else '')
    log('🎬 финальное видео: ' + FINV)


def gen_done():
    if not os.path.isfile(FINV) or os.path.getsize(FINV) < 200000:
        return False
    if os.path.isfile(CKPT) and os.path.getmtime(FINV) < os.path.getmtime(CKPT):
        return False
    if PHRASE and state().get('phrase_used') != PHRASE:
        log('⚠️ финальное видео было без тестовой фразы — перегенерирую')
        return False
    return True


def gen_verify():
    if not (os.path.isfile(FINV) and os.path.getsize(FINV) > 200000):
        return False
    import cv2
    cap = cv2.VideoCapture(FINV)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n < 10:
        log('⚠️ финальное видео битое (кадров: %d)' % n)
        return False
    return True


# ─────────────────────────── главный запуск ───────────────────────────

def _devstr():
    try:
        import torch
        if torch.cuda.is_available():
            return 'GPU ' + torch.cuda.get_device_name(0)
        return 'CPU'
    except Exception:
        return '?'


def free_space_pre():
    """Освобождает место ДО всех этапов: лишние куски и временные файлы.
    Ничего нужного не трогает — только хвост сверх необходимого."""
    try:
        if os.path.isdir(SEGD):
            trim_segs(need_segs())
        if os.path.isfile(RAWV):
            try:
                os.remove(RAWV)
                log('💾 удалён временный raw-файл')
            except OSError:
                pass
        if os.path.isfile(FLAG) and os.path.isfile(LASTP):
            try:
                os.remove(LASTP)
                log('💾 удалён промежуточный чекпоинт')
            except OSError:
                pass
    except Exception as e:
        log('free_space: ' + repr(e))


def run_all():
    global MAIN_TID, HEART
    MAIN_TID = threading.get_ident()
    os.makedirs(BASE, exist_ok=True)
    if HEART is not None:
        HEART.stop()
    HEART = Heart(STALL_SEC)
    try:
        log('═' * 56)
        log('AVATAR STUDIO v2 — самопроверяющийся конвейер')
        log('рабочая папка: ' + BASE)
        log('видео: %s (существует: %s)' % (VIDEO, os.path.isfile(VIDEO)))
        log('кадров: %d | эпох: %d | кусков нужно: %d | watchdog: %d с | устройство: %s'
            % (MAX_FRAMES, EPOCHS, need_segs(), STALL_SEC, _devstr()))
        disk_report('старт')
        free_space_pre()
        disk_report('после очистки')
        res = {}
        res['setup'] = run_stage('setup', setup_verify, setup_work, setup_verify)
        if res['setup']:
            disk_report('перед датасетом')
        res['dataset'] = run_stage('dataset', dataset_done, frames_work, dataset_done) if res['setup'] else False
        if res['dataset'] and PHRASE:
            resolve_phrase_audio()   # фраза готовится ДО обучения — предпросмотры сразу с ней
        if res['dataset']:
            disk_report('перед обучением')
        try:
            res['train'] = run_stage('train', train_verify, train_work, train_verify) if res['dataset'] else False
        except PreviewStop:
            pv = os.path.join(BASE, 'preview_latest.mp4')
            log('═' * 56)
            log('⏸ ОСТАНОВКА ПО ЗАПРОСУ (PREVIEW_STOP): первый предпросмотр готов.')
            log('👀 Смотрите: ' + pv)
            log('   Скачать: Kaggle правая панель → Output → STUDIO → preview_latest.mp4')
            log('   Понравилось? Уберите PREVIEW_STOP=1 из ячейки и запустите её заново —')
            log('   обучение продолжится с того же места и дойдёт до финального видео.')
            return True
        if res['train']:
            # двойной чекпоинт больше не нужен — освобождаем ~416 МБ
            if os.path.isfile(FLAG) and os.path.isfile(LASTP):
                try:
                    os.remove(LASTP)
                    log('💾 промежуточный чекпоинт удалён (освобождено ~416 МБ)')
                except OSError:
                    pass
            disk_report('перед генерацией')
        res['generate'] = run_stage('generate', gen_done, gen_work, gen_verify) if res['train'] else False
        log('═' * 56)
        log('ИТОГ:')
        for k, v in res.items():
            log(('✅ ' if v else '🔴 ') + k)
        if res['generate']:
            log('🎬 ГОТОВОЕ ВИДЕО: ' + FINV)
            log('Скачать: в Kaggle правая панель → Output → STUDIO → personal_final.mp4')
            return True
        log('Конвейер не завершён. Запустите ячейку ЕЩЁ РАЗ — сделанные этапы пропустятся.')
        log('Полный журнал: ' + LOGF)
        return False
    finally:
        HEART.stop()


if __name__ == '__main__':
    ok = False
    try:
        ok = run_all()
    except Exception as e:
        log('КРИТИЧЕСКАЯ ОШИБКА: ' + repr(e))
        log(traceback.format_exc())
    if not ok:
        raise RuntimeError('конвейер не завершён — подробности в журнале выше; перезапустите ячейку')
