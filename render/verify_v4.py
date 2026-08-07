#!/usr/bin/env python3
"""Самопроверка v4: контейнер, encode-верность, синхронность o(t) с аудио,
гладкость движения рта в декодированном видео, монтажи."""
import subprocess, sys, wave, os, glob, importlib.util
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
spec = importlib.util.spec_from_file_location("mv4", os.path.join(HERE, "make_video_v4.py"))
mv4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(mv4)
FF = mv4.FF
W, H, FPS, DUR = mv4.W, mv4.H, mv4.FPS, mv4.DUR
MY0, MY1, MX0, MX1 = mv4.MY0, mv4.MY1, mv4.MX0, mv4.MX1


def main():
    mp4 = os.path.join(HERE, "talking_avatar_5s.mp4")
    data = open(mp4, "rb").read()
    faststart = data.find(b"moov") < data.find(b"mdat")
    probe = subprocess.run([FF, "-i", mp4], capture_output=True, text=True).stderr.lower()
    ok_profile = "h264" in probe and "(main)" in probe and "yuv420p" in probe
    dec = subprocess.run([FF, "-v", "error", "-i", mp4, "-f", "null", "-"], capture_output=True, text=True)
    decode_clean = dec.returncode == 0 and dec.stderr.strip() == ""

    vd = os.path.join(HERE, "vframes")
    os.makedirs(vd, exist_ok=True)
    for p in glob.glob(vd + "/*.png"):
        os.remove(p)
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", mp4, os.path.join(vd, "f_%03d.png")],
                   check=True, capture_output=True)
    frames = sorted(glob.glob(vd + "/f_*.png"))
    total = len(frames)

    planz = np.load(os.path.join(HERE, "plan.npz"))
    P, blink = planz["P"], planz["blink"]

    names = ["avatar_closed.jpg", "avatar_open.jpg", "avatar_wide.jpg",
             "avatar_round.jpg", "avatar_blink.jpg"]
    L = [mv4.fit(Image.open(os.path.join(HERE, n)).convert("RGB")) for n in names]
    m_mouth = mv4.ellipse_mask(MY0, MY1, MX0, MX1)
    m_eye = mv4.ellipse_mask(mv4.EY0, mv4.EY1, mv4.EX0, mv4.EX1)

    diffs = []
    for f in range(0, total, 5):
        exp = mv4.render_frame(f, P, blink[f], L, m_mouth, m_eye)
        got = np.asarray(Image.open(frames[f]).convert("RGB"), dtype=np.uint8)
        diffs.append(np.abs(exp.astype(int) - got.astype(int)).mean())
    enc_mad = float(np.mean(diffs))

    with wave.open(os.path.join(HERE, "voice.wav")) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)/32768.0
    hop, v = mv4.envelope(pcm)
    vt = np.array([v[min(int((f/FPS)/hop), len(v)-1)] for f in range(total)])
    o = P[:total, 0]
    def first_true(a): return int(np.argmax(a)) if a.any() else -1
    def last_true(a): return len(a)-1-int(np.argmax(a[::-1])) if a.any() else -1
    on_v, on_o = first_true(vt > 0.15), first_true(o > 0.15)
    off_v, off_o = last_true(vt > 0.15), last_true(o > 0.15)
    onset_ok = abs(on_v - on_o) <= 6
    offset_ok = abs(off_v - off_o) <= 9
    corr = float(np.corrcoef(o, vt)[0, 1])

    mouth = np.stack([np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)[MY0:MY1, MX0:MX1]
                      for p in frames])
    d1 = np.abs(np.diff(mouth, axis=0)).mean(axis=(1, 2, 3))
    smooth_ok = float(d1.max()) < 12 and float(np.median(d1)) < 4

    idx = list(range(0, 90, 6)) + [110, 140]
    cells = [Image.open(frames[i]).resize((160, 284)) for i in idx]
    mon = Image.new("RGB", (160*len(cells), 284))
    for i, c in enumerate(cells):
        mon.paste(c, (i*160, 0))
    mon.save(os.path.join(HERE, "montage.png"))
    cells = [Image.open(frames[i]).crop((250, 450, 510, 700)).resize((260, 250)) for i in range(12, 60, 3)]
    mon2 = Image.new("RGB", (260*len(cells), 250))
    for i, c in enumerate(cells):
        mon2.paste(c, (i*260, 0))
    mon2.save(os.path.join(HERE, "mouth_closeup.png"))

    print(f"faststart={faststart} profile_ok={ok_profile} decode_clean={decode_clean} frames={total}")
    print(f"encode_mad={enc_mad:.2f} onset_ok={onset_ok}({on_v}/{on_o}) offset_ok={offset_ok}({off_v}/{off_o}) sync_corr={corr:.3f}")
    print(f"mouth_d1 max={float(d1.max()):.2f} median={float(np.median(d1)):.2f} smooth_ok={smooth_ok}")
    ok = faststart and ok_profile and decode_clean and enc_mad < 6 and onset_ok and offset_ok and smooth_ok
    print("VERDICT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
