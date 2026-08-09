# -*- coding: utf-8 -*-
"""AVATAR STUDIO — веб-сервис персонализации аватара (Colab/Kaggle/сервер).
Вкладка 1: исходное видео (кешируется, грузится один раз) + автоподготовка датасета.
Вкладка 2: дообучение Wav2Lip на видео Марии (персональные губы/зубы, T4 OK).
Вкладка 3: генерация видео (база + аудио) персональной моделью.
Прогресс операций выводится в интерфейс живыми строками."""
import os, glob, subprocess, shutil
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE = os.environ.get('STUDIO_DIR', '/content/STUDIO')
W2L = os.path.join(BASE, 'Wav2Lip')
FF = 'ffmpeg'

STANDARD = """СТАНДАРТ ИСХОДНОГО ВИДЕО (как у HeyGen):
1) 10–20 минут суммарно (для первой пробы достаточно 3–5).
2) Лицо крупно, фронтально, смотрит в камеру, не закрывается руками.
3) Ровный мягкий свет, фон статичный, камера неподвижна.
4) 25–30 fps, 720p/1080p, без склеек и зумов.
5) Спокойная непрерывная речь, рот виден; без музыки и второго голоса."""


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


def setup():
    os.makedirs(BASE, exist_ok=True)
    if not os.path.isdir(W2L):
        sh(f'git clone -q https://github.com/Rudrabha/Wav2Lip.git {W2L}')
    ck = os.path.join(W2L, 'checkpoints')
    os.makedirs(ck, exist_ok=True)
    if not os.path.isfile(os.path.join(ck, 'wav2lip.pth')):
        sh(f'wget -q https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/wav2lip.pth -O {ck}/wav2lip.pth')
    if not os.path.isfile(os.path.join(ck, 's3fd.pth')):
        sh(f'wget -q https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth -O {ck}/s3fd.pth')
    shutil.copy(os.path.join(ck, 's3fd.pth'), os.path.join(W2L, 'face_detection/detection/sfd/s3fd.pth'))
    ap = os.path.join(W2L, 'audio.py')
    src = open(ap, encoding='utf-8').read()
    if 'librosa.filters.mel(hp.sample_rate' in src:
        src = src.replace('librosa.filters.mel(hp.sample_rate, hp.n_fft,', 'librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft,')
        open(ap, 'w', encoding='utf-8').write(src)
    return 'База готова: Wav2Lip + веса скачаны.'


def preprocess(video, max_frames=6000):
    """Кеш: видео сохраняется в STUDIO/src_maria.mp4 и переиспользуется.
    Детекция лица — на GPU (если есть) и каждые 5 кадров (x5 быстрее).
    Прогресс выводится живыми строками."""
    dst = os.path.join(BASE, 'src_maria.mp4')
    if video is not None:
        src = video if isinstance(video, str) else video.name
        shutil.copy(src, dst)
    if not os.path.isfile(dst):
        yield 'Сначала загрузите видео Марии во вкладке 1.'
        return
    ds = os.path.join(BASE, 'dataset')
    os.makedirs(ds, exist_ok=True)
    segd = os.path.join(BASE, 'segs')
    os.makedirs(segd, exist_ok=True)
    if glob.glob(segd + '/seg_*.avi'):
        yield 'Сегменты уже нарезаны — пропускаю нарезку.'
    else:
        yield 'Режу на сегменты по 8 сек…'
        sh(f'{FF} -y -loglevel error -i "{dst}" -r 25 -c:v mjpeg -q:v 2 -f segment -segment_time 8 -reset_timestamps 1 {segd}/seg_%03d.avi')
    existing = sum(len(glob.glob(d + '/*.jpg')) for d in glob.glob(ds + '/*') if os.path.isdir(d))
    if existing >= int(max_frames):
        yield f'🟢 Датасет уже собран ({existing} кадров) — сразу к обучению.'
        return
    yield '🟢 СТАТУС: работаем — шаг 1/3 подготовка данных.'
    import sys
    sys.path.insert(0, W2L)
    from face_detection import FaceAlignment, LandmarksType
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    fa = FaceAlignment(LandmarksType._2D, flip_input=False, device=dev)
    total = 0
    report = []
    for seg in sorted(glob.glob(segd + '/seg_*.avi')):
        cap = cv2.VideoCapture(seg)
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        if len(frames) < 25:
            continue
        sid = os.path.splitext(os.path.basename(seg))[0]
        fdir = os.path.join(ds, sid)
        os.makedirs(fdir, exist_ok=True)
        have = len(glob.glob(fdir + '/*.jpg'))
        if have >= len(frames):
            yield f'🟢 Сегмент {sid} уже обработан ({have} кадров) — пропускаю.'
            if not os.path.isfile(os.path.join(fdir, 'audio.wav')):
                sh(f'{FF} -y -loglevel error -i "{seg}" -ar 16000 -ac 1 {fdir}/audio.wav')
            report.append(f'{sid}: {len(frames)} кадров (готово ранее)')
            total += len(frames)
            if total >= int(max_frames):
                break
            continue
        yield f'🟢 Обрабатываю сегмент {sid}…'
        last = None
        for i, fr in enumerate(frames):
            if total >= int(max_frames):
                break
            if i % 5 == 0 or last is None:
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
                y1 = max(0, y1 - pad); y2 = min(fr.shape[0], y2 + int(pad * 1.6))
                x1 = max(0, x1 - pad); x2 = min(fr.shape[1], x2 + pad)
                crop = cv2.resize(fr[y1:y2, x1:x2], (96, 96))
            else:
                h, w = fr.shape[:2]
                crop = cv2.resize(fr[int(h*0.1):int(h*0.7), int(w*0.25):int(w*0.75)], (96, 96))
            cv2.imwrite(os.path.join(fdir, f'{i:05d}.jpg'), crop)
            total += 1
            if i % 100 == 0:
                yield f'🟢 {sid}: кадр {i}/{len(frames)} (всего {total}/{int(max_frames)})'
        sh(f'{FF} -y -loglevel error -i "{seg}" -ar 16000 -ac 1 {fdir}/audio.wav')
        report.append(f'{sid}: {min(len(frames), int(max_frames))} кадров')
        if total >= int(max_frames):
            break
    yield 'Датасет готов:\n' + '\n'.join(report) + f'\nВсего кадров: {total}. Переходите во вкладку 2.'


class ClipDS(Dataset):
    def __init__(self, ds, T=5):
        self.T = T
        self.clips = [d for d in sorted(glob.glob(ds + '/*')) if os.path.isdir(d)]
        import sys
        sys.path.insert(0, W2L)
        import audio as A
        if not getattr(A, '_mel_kw', False):
            import librosa as _lb
            def _mb():
                return _lb.filters.mel(sr=A.hp.sample_rate, n_fft=A.hp.n_fft, n_mels=A.hp.num_mels, fmin=A.hp.fmin, fmax=A.hp.fmax)
            A._build_mel_basis = _mb
            A._mel_kw = True
        self.A = A
        self.mels = {}
        self.files = {}
        for c in self.clips:
            wav = A.load_wav(os.path.join(c, 'audio.wav'), 16000)
            self.mels[c] = A.melspectrogram(wav).T
            self.files[c] = sorted(glob.glob(c + '/*.jpg'))

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


def train(epochs, lr):
    flag = os.path.join(BASE, '.done_train')
    ckpt = os.path.join(BASE, 'personal_maria.pth')
    if os.path.isfile(flag) and os.path.isfile(ckpt):
        yield '🟢 Модель уже обучена — пропускаю обучение.'
        return
    import sys
    sys.path.insert(0, W2L)
    from models.wav2lip import Wav2Lip
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Wav2Lip().to(dev)
    ck = torch.load(os.path.join(W2L, 'checkpoints/wav2lip.pth'), map_location=dev, weights_only=False)
    sd = ck.get('state_dict', ck)
    if any(k.startswith('module.') for k in sd):
        sd = {k[7:]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    ds = ClipDS(os.path.join(BASE, 'dataset'))
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    log = []
    for e in range(int(epochs)):
        it = 0
        for x_t, indiv_t, mel_t, y_t in dl:
            x_t, indiv_t, y_t = x_t.to(dev), indiv_t.to(dev), y_t.to(dev)
            g = model(indiv_t, x_t)
            loss = nn.L1Loss()(g, y_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            it += 1
            if it % 100 == 0:
                yield f'🟢 epoch {e}: итерация {it}, loss={loss.item():.4f}'
        torch.save({'state_dict': model.state_dict()}, os.path.join(BASE, 'personal_maria.pth'))
        yield f'🟢 epoch {e} завершён: loss={loss.item():.4f}'
    open(os.path.join(BASE, '.done_train'), 'w').write('ok')


def generate(base_video, audio):
    fin0 = os.path.join(BASE, 'personal_final.mp4')
    ckpt0 = os.path.join(BASE, 'personal_maria.pth')
    if os.path.isfile(fin0) and os.path.isfile(ckpt0) and os.path.getmtime(fin0) >= os.path.getmtime(ckpt0):
        return f'🟢 Видео уже сгенерировано ранее: {fin0}'
    import sys
    cb, ca = os.path.join(BASE, 'base_last.mp4'), os.path.join(BASE, 'audio_last.wav')
    if base_video is not None:
        shutil.copy(base_video if isinstance(base_video, str) else base_video.name, cb)
    if audio is not None:
        src = audio if isinstance(audio, str) else audio.name
        sh(f'{FF} -y -loglevel error -i "{src}" -ar 16000 -ac 1 {ca}')
    if base_video is None and os.path.isfile(cb):
        base_video = cb
    if audio is None and os.path.isfile(ca):
        audio = ca
    if base_video is None or audio is None:
        return 'Загрузите базовое видео и аудио во вкладке 3 (один раз).'
    sys.path.insert(0, W2L)
    out = os.path.join(BASE, 'personal_raw.mp4')
    ckpt = os.path.join(BASE, 'personal_maria.pth')
    if not os.path.isfile(ckpt):
        base = os.path.join(W2L, 'checkpoints/wav2lip.pth')
        c = torch.load(base, map_location='cpu', weights_only=False)
        sd = c.get('state_dict', c)
        if any(k.startswith('module.') for k in sd):
            torch.save({'state_dict': {k[7:]: v for k, v in sd.items()}}, base + '.clean.pth')
            ckpt = base + '.clean.pth'
        else:
            ckpt = base
    infer_py = os.path.join(BASE, 'w2l_infer.py')
    open(infer_py, 'w').write(
        "import torch, runpy, sys, os\n"
        "sys.path.insert(0, os.getcwd())\n"
        "ckpt, face, audio, out, bs = sys.argv[1:6]\n"
        "_tl = torch.load\n"
        "torch.load = lambda *a, **k: _tl(*a, **{**k, 'weights_only': False})\n"
        "sys.argv = ['i', '--checkpoint_path', ckpt, '--face', face, '--audio', audio, '--outfile', out, '--pads', '0', '20', '0', '0', '--wav2lip_batch_size', bs]\n"
        "runpy.run_path('inference.py', run_name='__main__')\n")
    for bs in (16, 4, 1):
        sh(f'cd {W2L} && OMP_NUM_THREADS=2 PATH=$HOME/.local/bin:$PATH python {infer_py} {ckpt} {base_video} {audio} {out} {bs}')
        if os.path.isfile(out):
            break
    fin = os.path.join(BASE, 'personal_final.mp4')
    sh(f'{FF} -y -loglevel error -i {out} -vf "format=yuv420p" -c:v libx264 -crf 18 -movflags +faststart -c:a aac -b:a 128k {fin}')
    return fin
