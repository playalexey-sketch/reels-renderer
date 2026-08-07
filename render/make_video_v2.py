#!/usr/bin/env python3
"""Talking avatar v2: 3 visemes (closed/slight/open) switched by audio envelope.
Encodes a maximally compatible 5s 720x1280 mp4 (H.264 main, yuv420p, faststart)."""
import subprocess, sys, wave, os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FF = subprocess.check_output([sys.executable, "-c",
    "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())"]).decode().strip()

W, H, FPS, DUR = 720, 1280, 24, 5.0
# mouth region in OUTPUT coords (from diff-analysis of viseme pairs), incl. margin
MY0, MY1, MX0, MX1 = 480, 675, 250, 510

def decode_pcm(mp3):
    wav = os.path.join(HERE, "voice.wav")
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", mp3, "-ar", "16000", "-ac", "1", wav],
                   check=True, capture_output=True)
    with wave.open(wav) as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

def opening_track(pcm, sr=16000, hop=0.010):
    hop_n = int(sr * hop)
    n = len(pcm) // hop_n
    rms = np.array([np.sqrt(np.mean(pcm[i*hop_n:(i+1)*hop_n]**2) + 1e-9) for i in range(n)])
    k = 3  # 30ms smoothing
    rms = np.convolve(rms, np.ones(k)/k, mode="same")
    floor, peak = np.percentile(rms, 20), np.percentile(rms, 98)
    o = np.clip((rms - floor) / max(peak - floor, 1e-6), 0, 1)
    return hop, o

def states_for_frames(o, hop):
    total = int(DUR * FPS)
    st = []
    for f in range(total):
        t = f / FPS
        v = o[min(int(t / hop), len(o) - 1)]
        if t < 0.12 or t > 4.15:
            s = 0
        elif v < 0.15:
            s = 0
        elif v < 0.50:
            s = 1
        else:
            s = 2
        st.append(s)
    # articulation realism: break long 'open' runs with a 'slight'
    i = 0
    while i < total:
        if st[i] == 2:
            j = i
            while j < total and st[j] == 2:
                j += 1
            if j - i > 5:
                st[(i + j) // 2] = 1
            i = j
        else:
            i += 1
    # smooth articulation: every closed<->open transition passes through 'slight'
    for i in range(1, total):
        if abs(st[i] - st[i - 1]) == 2:
            st[i] = 1
    return st

def fit(img):
    s = max(W / img.width, H / img.height)
    iw, ih = int(img.width * s), int(img.height * s)
    img = img.resize((iw, ih), Image.LANCZOS)
    l, t = (iw - W) // 2, (ih - H) // 2
    return img.crop((l, t, l + W, t + H))

def zoomed(img_fit, t):
    z = 1.0 + 0.05 * (t / DUR)
    iw, ih = int(W * z), int(H * z)
    img = img_fit.resize((iw, ih), Image.LANCZOS)
    l, t0 = (iw - W) // 2, (ih - H) // 2
    return img.crop((l, t0, l + W, t0 + H))

def make_mask():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cy, cx = (MY0 + MY1) / 2, (MX0 + MX1) / 2
    ry, rx = (MY1 - MY0) / 2, (MX1 - MX0) / 2
    d = np.sqrt(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2)
    m = np.clip((1.15 - d) / 0.35, 0, 1)  # feathered ellipse
    return m[..., None]

def main():
    pcm = decode_pcm(os.path.join(HERE, "voice.mp3"))
    hop, o = opening_track(pcm)
    states = states_for_frames(o, hop)

    imgs = {s: fit(Image.open(os.path.join(HERE, n)).convert("RGB"))
            for s, n in [(0, "avatar_closed.jpg"), (1, "avatar_slight.jpg"), (2, "avatar_open.jpg")]}
    mask = make_mask()

    frames_dir = os.path.join(HERE, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for f in range(int(DUR * FPS)):
        t = f / FPS
        base = np.asarray(zoomed(imgs[0], t), dtype=np.float32)
        if states[f] == 0:
            out = base
        else:
            vis = np.asarray(zoomed(imgs[states[f]], t), dtype=np.float32)
            out = base * (1 - mask) + vis * mask
        Image.fromarray(out.astype(np.uint8)).save(os.path.join(frames_dir, f"frame_{f:03d}.png"))

    out = os.path.join(HERE, "talking_avatar_5s.mp4")
    cmd = [FF, "-y", "-loglevel", "error", "-framerate", str(FPS),
           "-i", os.path.join(frames_dir, "frame_%03d.png"),
           "-i", os.path.join(HERE, "voice.mp3"),
           "-filter_complex", "[1:a]apad=whole_dur=5,aformat=channel_layouts=stereo[a]",
           "-map", "0:v", "-map", "[a]",
           "-c:v", "libx264", "-profile:v", "main", "-level", "4.0",
           "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "medium",
           "-movflags", "+faststart",
           "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-t", str(DUR), out]
    subprocess.run(cmd, check=True, capture_output=True)
    np.save(os.path.join(HERE, "states.npy"), np.array(states, dtype=np.int8))
    print("states:", [states.count(i) for i in range(3)], "->", out)

if __name__ == "__main__":
    main()
