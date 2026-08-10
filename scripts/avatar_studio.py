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
"""
import os, sys, time, glob, json, shutil, subprocess, threading, traceback

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
SEG_SECONDS = 8       # длительность одного куска
SEG_FRAMES = 190      # сколько кадров в среднем даёт один кусок
FF = 'ffmpeg'

W2L_URL  = 'https://github.com/Rudrabha/Wav2Lip.git'
CKPT_URL = 'https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/wav2lip.pth'
S3FD_URL = 'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth'

MAIN_TID = None
HEART = None


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
    disk_ok(2000)
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
            'from face_detection import FaceAlignment; print("IMPORTS_OK")') % W2L
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

        def _crop_win(self, spec, frame_id):
            s0 = int(80.0 * (frame_id / 25.0))
            return spec[s0:s0 + 16, :]

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
                x[:, :, 48:, :] = 0.0
                return x

            xw = prep(range(i, i + T))
            y = np.asarray([cv2.imread(fs[j]) for j in range(i, i + T)]) / 255.0
            y = np.transpose(y, (3, 0, 1, 2))
            xw2 = prep(range(w, w + T))
            x = np.concatenate([xw, xw2], axis=0)
            spec = self.mels[c]
            mel = self._crop_win(spec, i)
            indiv = np.asarray([self._crop_win(spec, j - 2).T for j in range(i + 1, i + 1 + T)])
            return (torch.FloatTensor(x), torch.FloatTensor(indiv).unsqueeze(1),
                    torch.FloatTensor(mel.T).unsqueeze(0), torch.FloatTensor(y))

    prog = state().get('train', {})
    start_e = int(prog.get('epoch', 0))
    skip_it = int(prog.get('iter', 0))
    if not os.path.isfile(LASTP):
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
        torch.save({'state_dict': model.state_dict()}, CKPT)
        save_state(train={'epoch': e + 1, 'iter': 0})
        log('🟢 эпоха %d/%d завершена, loss=%.4f' % (e + 1, EPOCHS, last_loss))
    open(FLAG, 'w').write('ok')
    log('обучение завершено, модель сохранена: ' + CKPT)


def train_verify():
    if not (os.path.isfile(FLAG) and os.path.isfile(CKPT)):
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
    aud = resolve_audio()
    if not train_verify():
        raise RuntimeError('персональная модель не грузится — нужен этап train')
    infer_py = os.path.join(BASE, 'w2l_infer.py')
    open(infer_py, 'w').write(
        "import torch, runpy, sys, os\n"
        "sys.path.insert(0, os.getcwd())\n"
        "ckpt, face, audio, out, bs = sys.argv[1:6]\n"
        "_tl = torch.load\n"
        "torch.load = lambda *a, **k: _tl(*a, **{**k, 'weights_only': False})\n"
        "sys.argv = ['i', '--checkpoint_path', ckpt, '--face', face, '--audio', audio, "
        "'--outfile', out, '--pads', '0', '20', '0', '0', '--wav2lip_batch_size', bs]\n"
        "runpy.run_path('inference.py', run_name='__main__')\n")
    if os.path.isfile(RAWV):
        os.remove(RAWV)
    ok = False
    for bs in (16, 4, 1):
        beat('генерация, батч %d' % bs)
        log('🟢 синхронизация губ: батч %d' % bs)
        p = run_cmd('cd "%s" && OMP_NUM_THREADS=2 "%s" "%s" "%s" "%s" "%s" "%s" %d'
                    % (W2L, sys.executable, infer_py, CKPT, base, aud, RAWV, bs), timeout=3600)
        if os.path.isfile(RAWV) and os.path.getsize(RAWV) > 200000:
            ok = True
            break
        tail = (p.stderr or '')[-200:].replace('\n', ' ')
        log('⚠️ батч %d не дал результата: %s' % (bs, tail))
        if os.path.isfile(RAWV):
            os.remove(RAWV)
    if not ok:
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
    log('🎬 финальное видео: ' + FINV)


def gen_done():
    if not os.path.isfile(FINV) or os.path.getsize(FINV) < 200000:
        return False
    if os.path.isfile(CKPT) and os.path.getmtime(FINV) < os.path.getmtime(CKPT):
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
        res = {}
        res['setup'] = run_stage('setup', setup_verify, setup_work, setup_verify)
        if res['setup']:
            disk_report('перед датасетом')
        res['dataset'] = run_stage('dataset', dataset_done, frames_work, dataset_done) if res['setup'] else False
        if res['dataset']:
            disk_report('перед обучением')
        res['train'] = run_stage('train', train_verify, train_work, train_verify) if res['dataset'] else False
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
