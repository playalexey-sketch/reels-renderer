#!/usr/bin/env python3
"""Talking avatar v3: непрерывный морфинг рта (уровни closed→slight→half→{open,wide}),
артикуляционная динамика (быстрое открытие/медленное закрытие), чередование форм по слогам,
покачивания головы, моргания. 30 fps, совместимый encode."""
import subprocess, sys, wave, os, math
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FF = subprocess.check_output([sys.executable, "-c",
    "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())"]).decode().strip()

W, H, FPS, DUR = 720, 1280, 30, 5.0
MY0, MY1, MX0, MX1 = 480, 675, 250, 510      # рот (output coords)
EY0, EY1, EX0, EX1 = 372, 428, 255, 505      # глаза (output coords)


def decode_pcm(mp3):
    wav = os.path.join(HERE, "voice.wav")
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", mp3, "-ar", "16000", "-ac", "1", wav],
                   check=True, capture_output=True)
    with wave.open(wav) as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def envelope(pcm, sr=16000, hop=0.010):
    hop_n = int(sr * hop)
    n = len(pcm) // hop_n
    rms = np.array([np.sqrt(np.mean(pcm[i*hop_n:(i+1)*hop_n]**2) + 1e-9) for i in range(n)])
    rms = np.convolve(rms, np.ones(5)/5, mode="same")
    floor, peak = np.percentile(rms, 20), np.percentile(rms, 98)
    return hop, np.clip((rms - floor) / max(peak - floor, 1e-6), 0, 1)


def plan(total, v, hop):
    """Непрерывные o(t), s(t) (форма а/э), blink(t) + дискретные события."""
    vt = np.array([v[min(int((f/FPS)/hop), len(v)-1)] for f in range(total)])
    if len(vt) > 4.2*FPS:  # хвост тишины
        vt[int(4.15*FPS):] = 0
    vt[:int(0.12*FPS)] = 0
    # артикуляционная динамика: attack 35ms, release 90ms
    o = np.zeros(total)
    for f in range(1, total):
        dt = 1/FPS
        target = vt[f]
        tau = 0.035 if target > o[f-1] else 0.090
        o[f] = o[f-1] + (target - o[f-1]) * (1 - math.exp(-dt/tau))
    # слоги: чередование формы s (0=open 'а', 1=wide 'э')
    voiced = vt > 0.25
    s = np.zeros(total)
    seg_id, cur = np.zeros(total, int), -1
    for f in range(total):
        if voiced[f] and (f == 0 or not voiced[f-1]):
            cur += 1
        seg_id[f] = max(cur, 0)
    for f in range(total):
        s[f] = (seg_id[f] % 2)
    # сгладить s (кроссфейд 70ms)
    k = max(2, int(0.07*FPS))
    s = np.convolve(s, np.ones(k)/k, mode="same")
    # моргания
    blink = np.zeros(total)
    for t0 in (1.35, 3.20):
        d = 0.16
        i0, i1 = int(t0*FPS), int((t0+d)*FPS)
        for f in range(max(i0, 0), min(i1, total)):
            blink[f] = math.sin(math.pi*(f/FPS - t0)/d)**2
    return vt, o, s, blink


def fit(img):
    s = max(W/img.width, H/img.height)
    iw, ih = int(img.width*s), int(img.height*s)
    img = img.resize((iw, ih), Image.LANCZOS)
    l, t = (iw-W)//2, (ih-H)//2
    return np.asarray(img.crop((l, t, l+W, t+H)), dtype=np.float32)


def ellipse_mask(y0, y1, x0, x1, feather=0.35):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cy, cx = (y0+y1)/2, (x0+x1)/2
    ry, rx = (y1-y0)/2, (x1-x0)/2
    d = np.sqrt(((yy-cy)/ry)**2 + ((xx-cx)/rx)**2)
    return np.clip((1.15-d)/feather*0.5, 0, 1)[..., None]


def mouth_texture(o, s, L):
    if o <= 0.35:
        a, lo, hi = o/0.35, L[0], L[1]
    elif o <= 0.65:
        a, lo, hi = (o-0.35)/0.30, L[1], L[2]
    else:
        a, lo = (o-0.65)/0.35, L[2]
        hi = L[3]*(1-s) + L[4]*s
    return lo*(1-a) + hi*a


def render_frame(f, o, s, blink, L, m_mouth, m_eye):
    t = f/FPS
    tex = mouth_texture(min(o, 1.0), s, L)
    frame = L[0].copy()
    frame = frame*(1-m_mouth) + tex*m_mouth
    if blink > 0.01:
        frame = frame*(1-m_eye*blink) + L[5]*(m_eye*blink)
    z = 1.02 + 0.04*(t/DUR)
    iw, ih = int(W*z)+8, int(H*z)+8
    img = Image.fromarray(frame.astype(np.uint8)).resize((iw, ih), Image.LANCZOS)
    dx = int(round(2*math.sin(2*math.pi*0.45*t)))
    dy = int(round(1.2*math.sin(2*math.pi*0.30*t+1.0) + 2.0*o))
    l = (iw-W)//2 + dx
    t0 = (ih-H)//2 + dy
    l = max(0, min(l, iw-W)); t0 = max(0, min(t0, ih-H))
    return np.asarray(img.crop((l, t0, l+W, t0+H)), dtype=np.uint8)


def main():
    pcm = decode_pcm(os.path.join(HERE, "voice.mp3"))
    hop, v = envelope(pcm)
    total = int(DUR*FPS)
    vt, o, s, blink = plan(total, v, hop)

    names = ["avatar_closed.jpg", "avatar_slight.jpg", "avatar_half.jpg",
             "avatar_open.jpg", "avatar_wide.jpg", "avatar_blink.jpg"]
    L = [fit(Image.open(os.path.join(HERE, n)).convert("RGB")) for n in names]
    m_mouth = ellipse_mask(MY0, MY1, MX0, MX1)
    m_eye = ellipse_mask(EY0, EY1, EX0, EX1)

    frames_dir = os.path.join(HERE, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for f in range(total):
        Image.fromarray(render_frame(f, o[f], s[f], blink[f], L, m_mouth, m_eye)) \
            .save(os.path.join(frames_dir, f"frame_{f:03d}.png"))
    np.savez(os.path.join(HERE, "plan.npz"), o=o, s=s, blink=blink)

    out = os.path.join(HERE, "talking_avatar_5s.mp4")
    cmd = [FF, "-y", "-loglevel", "error", "-framerate", str(FPS),
           "-i", os.path.join(frames_dir, "frame_%03d.png"),
           "-i", os.path.join(HERE, "voice.mp3"),
           "-filter_complex", "[1:a]apad=whole_dur=5,aformat=channel_layouts=stereo[a]",
           "-map", "0:v", "-map", "[a]",
           "-c:v", "libx264", "-profile:v", "main", "-level", "4.0",
           "-pix_fmt", "yuv420p", "-crf", "21", "-preset", "medium",
           "-movflags", "+faststart",
           "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-t", str(DUR), out]
    subprocess.run(cmd, check=True, capture_output=True)
    print("v3 rendered ->", out)


if __name__ == "__main__":
    main()
