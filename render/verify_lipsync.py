#!/usr/bin/env python3
"""Self-check of talking_avatar_5s.mp4: decode the REAL mp4, classify viseme per frame,
correlate with audio envelope; container/playability checks; montage for eyeball."""
import subprocess, sys, wave, os, glob
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FF = subprocess.check_output([sys.executable, "-c",
    "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())"]).decode().strip()
W, H, FPS, DUR = 720, 1280, 24, 5.0
MY0, MY1, MX0, MX1 = 480, 675, 250, 510

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
    return img.crop((l, t0, l + W, t0 + H)).convert("RGB")

def main():
    mp4 = os.path.join(HERE, "talking_avatar_5s.mp4")

    # 1) container checks
    data = open(mp4, "rb").read()
    faststart = data.find(b"moov") < data.find(b"mdat")
    probe = subprocess.run([FF, "-i", mp4], capture_output=True, text=True).stderr
    probe_l = probe.lower()
    ok_profile = "h264" in probe_l and "(main)" in probe_l and "yuv420p" in probe_l
    dec = subprocess.run([FF, "-v", "error", "-i", mp4, "-f", "null", "-"],
                         capture_output=True, text=True)
    decode_clean = dec.returncode == 0 and dec.stderr.strip() == ""

    # 2) decode real frames
    vd = os.path.join(HERE, "vframes")
    os.makedirs(vd, exist_ok=True)
    for p in glob.glob(vd + "/*.png"):
        os.remove(p)
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", mp4, os.path.join(vd, "f_%03d.png")],
                   check=True, capture_output=True)
    frames = sorted(glob.glob(vd + "/f_*.png"))

    refs = {s: fit(Image.open(os.path.join(HERE, n)).convert("RGB"))
            for s, n in [(0, "avatar_closed.jpg"), (1, "avatar_slight.jpg"), (2, "avatar_open.jpg")]}

    def classify(f):
        t = f / FPS
        patch_f = np.asarray(Image.open(frames[f]), dtype=np.float32)[MY0:MY1, MX0:MX1]
        best, bs = 1e18, -1
        for s in range(3):
            r = np.asarray(zoomed(refs[s], t), dtype=np.float32)[MY0:MY1, MX0:MX1]
            d = np.mean((patch_f - r) ** 2)
            if d < best:
                best, bs = d, s
        return bs

    video_states = np.array([classify(f) for f in range(len(frames))], dtype=np.int8)
    intended = np.load(os.path.join(HERE, "states.npy"))

    # 3) audio-derived truth (independent recompute)
    wav = os.path.join(HERE, "voice.wav")
    with wave.open(wav) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    hop_n = 160
    n = len(pcm) // hop_n
    rms = np.array([np.sqrt(np.mean(pcm[i*hop_n:(i+1)*hop_n]**2) + 1e-9) for i in range(n)])
    rms = np.convolve(rms, np.ones(3)/3, mode="same")
    floor, peak = np.percentile(rms, 20), np.percentile(rms, 98)
    o = np.clip((rms - floor) / max(peak - floor, 1e-6), 0, 1)
    audio_open = np.array([o[min(int((f/FPS)/0.010), len(o)-1)] for f in range(len(frames))])
    audio_talk = (audio_open > 0.15).astype(np.int8)
    video_talk = (video_states > 0).astype(np.int8)

    # best shift alignment +-3 frames
    accs = {}
    for sh in range(-3, 4):
        a, v = audio_talk, np.roll(video_talk, sh)
        accs[sh] = float(np.mean(a == v))
    best_sh = max(accs, key=accs.get)

    talk_acc = accs[best_sh]
    head_closed = all(video_states[:3] == 0)
    tail_closed = all(video_states[-21:] == 0)  # last ~0.9s smile
    # no impossible jumps closed<->open in one frame
    jumps = int(np.sum(np.abs(np.diff(video_states.astype(int))) == 2))

    # 4) montage for eyeball
    idx = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 96, 110]
    cells = [Image.open(frames[i]).resize((180, 320)) for i in idx]
    mon = Image.new("RGB", (180 * len(cells), 320))
    for i, c in enumerate(cells):
        mon.paste(c, (i * 180, 0))
    mon.save(os.path.join(HERE, "montage.png"))

    print(f"faststart={faststart} profile_ok={ok_profile} decode_clean={decode_clean}")
    print(f"frames={len(frames)} video_states={np.bincount(video_states, minlength=3).tolist()}")
    print(f"talk-vs-audio accuracy by shift: { {k: round(v,3) for k,v in accs.items()} }")
    print(f"best_shift={best_sh} talk_acc={talk_acc:.3f} head_closed={head_closed} tail_closed={tail_closed} closed<->open jumps={jumps}")
    ok = faststart and ok_profile and decode_clean and talk_acc >= 0.90 and head_closed and tail_closed and jumps == 0
    print("VERDICT:", "PASS" if ok else "FAIL")

if __name__ == "__main__":
    main()
