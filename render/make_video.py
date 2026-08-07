#!/usr/bin/env python3
"""Build a 5s 'talking avatar' reel: image pair + RU voice -> faux lip-sync via audio envelope."""
import subprocess, sys, wave, math, os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FF = subprocess.check_output([sys.executable, "-c",
    "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())"]).decode().strip()

W, H, FPS, DUR = 720, 1280, 24, 5.0

def decode_pcm(mp3):
    wav = os.path.join(HERE, "voice.wav")
    subprocess.run([FF, "-y", "-i", mp3, "-ar", "16000", "-ac", "1", wav],
                   check=True, capture_output=True)
    with wave.open(wav) as w:
        n = w.getnframes()
        raw = w.readframes(n)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

def envelope(pcm, sr=16000, hop=0.02):
    hop_n = int(sr * hop)
    n = len(pcm) // hop_n
    rms = np.array([np.sqrt(np.mean(pcm[i*hop_n:(i+1)*hop_n]**2) + 1e-9) for i in range(n)])
    return hop, rms

def mouth_state(t, hop, rms, thr):
    i = int(t / hop)
    return "open" if i < len(rms) and rms[i] > thr else "closed"

def main():
    pcm = decode_pcm(os.path.join(HERE, "voice.mp3"))
    hop, rms = envelope(pcm)
    thr = 0.22 * rms.max()

    closed = Image.open(os.path.join(HERE, "avatar_closed.jpg")).convert("RGB")
    opened = Image.open(os.path.join(HERE, "avatar_open.jpg")).convert("RGB")

    def fit(img):
        # cover-crop to W,H
        s = max(W / img.width, H / img.height)
        iw, ih = int(img.width * s), int(img.height * s)
        img = img.resize((iw, ih), Image.LANCZOS)
        l, t = (iw - W) // 2, (ih - H) // 2
        return img.crop((l, t, l + W, t + H))

    frames_dir = os.path.join(HERE, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    total = int(DUR * FPS)
    states = []
    for f in range(total):
        t = f / FPS
        st = mouth_state(t, hop, rms, thr) if t < 4.05 else "closed"
        states.append(st)
    # merge runs shorter than 2 frames into previous state
    for f in range(1, total - 1):
        if states[f] != states[f-1] and states[f] != states[f+1]:
            states[f] = states[f-1]

    for f in range(total):
        t = f / FPS
        z = 1.0 + 0.06 * (t / DUR)            # slow push-in
        src = opened if states[f] == "open" else closed
        img = fit(src)
        iw, ih = int(W * z), int(H * z)
        img = img.resize((iw, ih), Image.LANCZOS)
        l, tpos = (iw - W) // 2, (ih - H) // 2
        img = img.crop((l, tpos, l + W, tpos + H))
        img.save(os.path.join(frames_dir, f"frame_{f:03d}.png"))

    out = os.path.join(HERE, "talking_avatar_5s.mp4")
    cmd = [FF, "-y", "-framerate", str(FPS), "-i", os.path.join(frames_dir, "frame_%03d.png"),
           "-i", os.path.join(HERE, "voice.mp3"),
           "-filter_complex", "[1:a]apad=whole_dur=5[a]",
           "-map", "0:v", "-map", "[a]",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           "-c:a", "aac", "-b:a", "128k", "-t", str(DUR), out]
    subprocess.run(cmd, check=True, capture_output=True)
    print("OPEN frames:", states.count("open"), "->", out)

if __name__ == "__main__":
    main()
