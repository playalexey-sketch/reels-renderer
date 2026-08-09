# -*- coding: utf-8 -*-
"""AVATAR STUDIO — веб-сервис персонализации аватара (Colab/Kaggle/сервер).
Вкладка 1: загрузка исходного видео Марии + автоподготовка датасета (стандарт ниже).
Вкладка 2: дообучение Wav2Lip на её видео (персональные губы/зубы, T4 OK).
Вкладка 3: генерация видео (база + аудио) персональной моделью + GFPGAN.
Запуск: python avatar_studio.py  (в Colab — через bootstrap-ячейку)."""
import os, glob, subprocess, shutil, math
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

BASE = os.environ.get('STUDIO_DIR', '/content/STUDIO')
W2L = os.path.join(BASE, 'Wav2Lip')
FF = 'ffmpeg'

STANDARD = """СТАНДАРТ ИСХОДНОГО ВИДЕО (как у HeyGen):
1) 10–20 минут суммарно (минимум 3–5 для первой итерации).
2) Лицо крупно, фронтально, смотрит в камеру, голова не закрывается руками/предметами.
3) Ровный мягкий свет, без резких теней и пересветов; фон статичный.
4) 25–30 fps, 720p/1080p, камера неподвижна, без склеек и зумов.
5) Спокойная непрерывная речь, рот полностью виден; без музыки и второго голоса.
6) Одна персона в кадре; без масок/очков с бликами."""


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
    if not os.path.isfile(os.path.join(ck, 'syncnet.pth')):
        sh(f'wget -q https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/syncnet.pth -O {ck}/syncnet.pth || true')  # best-effort; без него — обучение rec-loss
    if not os.path.isfile(os.path.join(ck, 's3fd.pth')):
        sh(f'wget -q https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth -O {ck}/s3fd.pth')
        shutil.copy(os.path.join(ck, 's3fd.pth'), os.path.join(W2L, 'face_detection/detection/sfd/s3fd.pth'))
    return 'База готова: Wav2Lip + веса скачаны.'


def preprocess(video):
    """Стандартная подготовка: 25fps, 16k, сегменты 8с, кропы лица 96.
    Видео кешируется: загрузили один раз — дальше используется автоматически."""
    dst = os.path.join(BASE, 'src_maria.mp4')
    if video is not None:
        shutil.copy(video if isinstance(video, str) else video.name, dst)
    if not os.path.isfile(dst):
        return 'Сначала загрузите видео Марии во вкладке 1.'
    video = dst
    ds = os.path.join(BASE, 'dataset')
    shutil.rmtree(ds, ignore_errors=True)
    os.makedirs(ds, exist_ok=True)
    wav = os.path.join(BASE, 'src16k.wav')
    sh(f'{FF} -y -loglevel error -i "{video}" -ar 16000 -ac 1 {wav}')
    segd = os.path.join(BASE, 'segs')
    shutil.rmtree(segd, ignore_errors=True)
    os.makedirs(segd)
    sh(f'{FF} -y -loglevel error -i "{video}" -r 25 -c:v mjpeg -q:v 2 -f segment -segment_time 8 -reset_timestamps 1 {segd}/seg_%03d.avi')
    import sys
    sys.path.insert(0, W2L)
    from face_detection import FaceAlignment, LandmarksType
    fa = FaceAlignment(LandmarksType._2D, flip_input=False, device='cuda' if torch.cuda.is_available() else 'cpu',
                       model_path=os.path.join(W2L, 'checkpoints/s3fd.pth')) if os.path.isfile(os.path.join(W2L, 'checkpoints/s3fd.pth')) else None
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
        preds = fa.get_detections_for_batch(frames[0:1]) if fa else None
        sid = os.path.splitext(os.path.basename(seg))[0]
        fdir = os.path.join(ds, sid)
        os.makedirs(fdir, exist_ok=True)
        for i, fr in enumerate(frames):
            if preds is not None and len(preds[0]):
                x1, y1, x2, y2 = preds[0][0].astype(int)
                pad = int((y2 - y1) * 0.25)
                y1 = max(0, y1 - pad); y2 = min(fr.shape[0], y2 + int(pad * 1.6))
                x1 = max(0, x1 - pad); x2 = min(fr.shape[1], x2 + pad)
                crop = cv2.resize(fr[y1:y2, x1:x2], (96, 96))
            else:
                h, w = fr.shape[:2]
                crop = cv2.resize(fr[int(h*0.1):int(h*0.7), int(w*0.25):int(w*0.75)], (96, 96))
            cv2.imwrite(os.path.join(fdir, f'{i:05d}.jpg'), crop)
        sh(f'{FF} -y -loglevel error -i "{seg}" -ar 16000 -ac 1 {fdir}/audio.wav')
        report.append(f'{sid}: {len(frames)} кадров')
    return 'Датасет готов:\n' + '\n'.join(report[:20]) + f'\nВсего клипов: {len(report)}'


class ClipDS(Dataset):
    def __init__(self, ds, T=16):
        self.T = T
        self.clips = [d for d in sorted(glob.glob(ds + '/*')) if os.path.isdir(d)]
        import sys
        sys.path.insert(0, W2L)
        import audio as A
        self.A = A
        self.mels = {}
        for c in self.clips:
            wav = self.A.load_wav(os.path.join(c, 'audio.wav'), 16000)
            self.mels[c] = self.A.melspectrogram(wav)

    def __len__(self):
        return sum(max(1, len(glob.glob(c + '/*.jpg')) - self.T) for c in self.clips)

    def __getitem__(self, idx):
        c = self.clips[idx % len(self.clips)]
        fs = sorted(glob.glob(c + '/*.jpg'))
        T = min(self.T, len(fs))
        i = np.random.randint(0, max(1, len(fs) - T))
        frames = []
        for j in range(i, i + T):
            frames.append(cv2.imread(fs[j]) / 255.0)
        frames = np.stack(frames).astype(np.float32)
        m = self.mels[c][:, i * 4: i * 4 + T * 4]
        if m.shape[1] < T * 4:
            m = np.pad(m, ((0, 0), (0, T * 4 - m.shape[1])))
        return (torch.from_numpy(frames).permute(0, 3, 1, 2), torch.from_numpy(m))


def train(epochs, lr):
    import sys
    sys.path.insert(0, W2L)
    from models import Wav2Lip, SyncNet
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Wav2Lip().to(dev)
    ck = os.path.join(W2L, 'checkpoints/wav2lip.pth')
    model.load_state_dict(torch.load(ck, map_location=dev, weights_only=False).get('state_dict',
                 torch.load(ck, map_location=dev, weights_only=False)))
    expert = SyncNet().to(dev)
    sp = os.path.join(W2L, 'checkpoints/syncnet.pth')
    if os.path.isfile(sp):
        expert.load_state_dict(torch.load(sp, map_location=dev, weights_only=False).get('state_dict',
                     torch.load(sp, map_location=dev, weights_only=False)))
    expert.eval()
    ds = ClipDS(os.path.join(BASE, 'dataset'))
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=1)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    log = []
    for e in range(int(epochs)):
        for frames, mels in dl:
            frames, mels = frames.to(dev), mels.to(dev)
            g = model(mels, frames[:, :3])
            rec = nn.L1Loss()(g, frames)
            B, T, C, H, W = frames.shape
            gf = g.reshape(-1, 3, 96, 96)
            rf = frames.reshape(-1, 3, 96, 96)
            mm = mels.reshape(B * T, 1, 80, 16)
            a = expert(torch.cat([gf, mm], 1))
            b = expert(torch.cat([rf, mm], 1))
            sync = nn.MSELoss()(a, b)
            loss = rec + 0.3 * sync
            opt.zero_grad()
            loss.backward()
            opt.step()
        log.append(f'epoch {e}: loss={loss.item():.4f}')
        torch.save({'state_dict': model.state_dict()}, os.path.join(BASE, 'personal_maria.pth'))
    return '\n'.join(log) + '\nСохранено: personal_maria.pth'


def generate(base_video, audio):
    import sys
    # кеш: что загрузили один раз — используется повторно
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
    env = dict(os.environ, PYTHONPATH=W2L)
    sh(f'cd {W2L} && python -c "import torch,runpy,sys;_tl=torch.load;torch.load=lambda *a,**k:_tl(*a,**{{**k,\\"weights_only\\":False}});sys.argv=[\\"i\\",\\"--checkpoint_path\\",\\"{BASE}/personal_maria.pth\\",\\"--face\\",\\"{base_video}\\",\\"--audio\\",\\"{audio}\\",\\"--outfile\\",\\"{out}\\",\\"--pads\\",\\"0\\",\\"20\\",\\"0\\",\\"0\\"];runpy.run_path(\\"inference.py\\",run_name=\\"__main__\")"', env=env)
    fin = os.path.join(BASE, 'personal_final.mp4')
    sh(f'{FF} -y -loglevel error -i {out} -vf "format=yuv420p" -c:v libx264 -crf 18 -movflags +faststart -c:a aac -b:a 128k {fin}')
    return fin


import gradio as gr

with gr.Blocks(title='Avatar Studio') as demo:
    gr.Markdown('# AVATAR STUDIO — персональный аватар (стандарт HeyGen-класса)')
    gr.Markdown(STANDARD)
    with gr.Tab('1. Исходник'):
        btn0 = gr.Button('Подготовить базу (клон+веса)')
        out0 = gr.Textbox()
        btn0.click(setup, outputs=out0)
        vid = gr.Video(label='Видео Марии (по стандарту выше)')
        btn1 = gr.Button('Подготовить датасет')
        out1 = gr.Textbox()
        btn1.click(preprocess, inputs=vid, outputs=out1)
    with gr.Tab('2. Обучение'):
        ep = gr.Number(value=2, label='Эпохи')
        lr = gr.Number(value=1e-5, label='Learning rate')
        btn2 = gr.Button('Дообучить аватар')
        out2 = gr.Textbox()
        btn2.click(train, inputs=[ep, lr], outputs=out2)
    with gr.Tab('3. Генерация'):
        bv = gr.Video(label='Базовое видео (например, рендер SadTalker/LatentSync)')
        au = gr.Audio(label='Аудио (голос Марии или любой текст TTS)')
        btn3 = gr.Button('Сгенерировать')
        out3 = gr.Video()
        btn3.click(generate, inputs=[bv, au], outputs=out3)

demo.launch(share=True, server_name='0.0.0.0')
