#!/usr/bin/env python3
"""Talking avatar v4: фонемно-точный липсинк.
Rhubarb Lip Sync (open source) даёт таймлайн визем по аудио;
непрерывная деформация рта риг-ом губ (контрольные точки + IDW-поле):
сильное, гладкое, естественное движение, синхронное фонемам."""
import subprocess, sys, wave, os, math, json
import numpy as np
import cv2
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FF = subprocess.check_output([sys.executable, "-c",
    "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())"]).decode().strip()
RHUBARB = os.path.expanduser("~/third_party/rhubarb/.tools/rhubarb-Lip-Sync-1.13.0-Linux/rhubarb")

W, H, FPS, DUR = 720, 1280, 30, 5.0
MY0, MY1, MX0, MX1 = 460, 700, 230, 530
EY0, EY1, EX0, EX1 = 372, 428, 255, 505

# визема -> (открытие o, ширина w, округлость r)
VISEME = {
    "X": (0.00, 0.00, 0.00), "A": (0.06, 0.00, 0.00), "B": (0.35, 0.25, 0.00),
    "C": (0.60, 0.35, 0.00), "D": (1.00, 0.45, 0.00), "E": (0.40, 0.15, 0.15),
    "F": (0.30, 0.00, 0.85), "G": (0.65, 0.00, 0.75), "H": (0.25, 0.20, 0.25),
}

# контрольные точки рта (fit-координаты)
PTS = np.array([
    [296, 512], [452, 512],          # уголки
    [375, 503], [375, 527],          # верх/низ губ центр
    [375, 560],                      # под нижней губой
    [335, 507], [415, 507],          # верхняя губа бока
    [335, 522], [415, 522],          # нижняя губа бока
    [315, 565], [435, 565],          # челюсть бока
], np.float32)
# нулевые якоря по периметру, чтобы поле затухало
ANCH = np.array([
    [230, 450], [530, 450], [230, 700], [530, 700],
    [380, 440], [380, 700], [240, 580], [520, 580],
], np.float32)


def offsets(o, w, r):
    d = np.zeros_like(PTS)
    d[0] = (-3*o - 14*w + 12*r, 6*o)
    d[1] = (3*o + 14*w - 12*r, 6*o)
    d[2] = (0, -4*o + 4*r)
    d[3] = (0, 26*o - 4*w + 10*r)
    d[4] = (0, 18*o)
    d[5] = (0, -2*o)
    d[6] = (0, -2*o)
    d[7] = (-2*o - 6*w, 20*o)
    d[8] = (2*o + 6*w, 20*o)
    d[9] = (0, 12*o)
    d[10] = (0, 12*o)
    return d


def idw_field(o, w, r):
    disp = offsets(o, w, r)
    P = np.vstack([PTS, ANCH])
    D = np.vstack([disp, np.zeros_like(ANCH)])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    X = np.stack([xx, yy], -1)                       # (H,W,2)
    diff = X[..., None, :] - P[None, None, :, :]     # (H,W,N,2)
    dist2 = (diff**2).sum(-1) + 1e-4
    wgt = 1.0 / dist2**1.5
    field = (wgt[..., None] * D[None, None, :, :]).sum(-2) / wgt.sum(-1)[..., None]
    return field


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


def rhubarb_timeline(wav_path):
    out = os.path.join(HERE, "rhubarb.json")
    subprocess.run([RHUBARB, "-f", "json", "-o", out, wav_path],
                   check=True, capture_output=True)
    return json.load(open(out))["mouthCues"]


def targets_per_frame(cues, total):
    tgt = np.zeros((total, 3))
    for c in cues:
        i0, i1 = int(c["start"]*FPS), min(int(c["end"]*FPS)+1, total)
        tgt[i0:i1] = VISEME.get(c["value"], (0.3, 0.2, 0.0))
    P = np.zeros((total, 3))
    for f in range(1, total):
        dt = 1/FPS
        for k in range(3):
            tau = 0.025 if tgt[f, k] > P[f-1, k] else 0.060
            P[f, k] = P[f-1, k] + (tgt[f, k] - P[f-1, k]) * (1 - math.exp(-dt/tau))
    return P


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


def smoothstep(a, b, x):
    t = np.clip((x-a)/(b-a), 0, 1)
    return t*t*(3-2*t)


def render_frame(f, P, blink, L, m_mouth, m_eye):
    o, w, r = P[f]
    field = idw_field(o, w, r)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    warped = cv2.remap(L[0], (xx - field[..., 0]).astype(np.float32),
                       (yy - field[..., 1]).astype(np.float32),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    a_open = smoothstep(0.25, 0.70, o)
    a_wide = 0.5*w*smoothstep(0.2, 0.6, w)
    a_round = 0.6*r*smoothstep(0.2, 0.6, r)
    tex = warped
    tex = tex*(1-a_open*m_mouth) + L[1]*(a_open*m_mouth)
    tex = tex*(1-a_wide*m_mouth) + L[2]*(a_wide*m_mouth)
    tex = tex*(1-a_round*m_mouth) + L[3]*(a_round*m_mouth)
    frame = L[0]*(1-m_mouth) + tex*m_mouth
    if blink > 0.01:
        frame = frame*(1-m_eye*blink) + L[4]*(m_eye*blink)
    t = f/FPS
    z = 1.02 + 0.04*(t/DUR)
    iw, ih = int(W*z)+8, int(H*z)+8
    img = Image.fromarray(frame.astype(np.uint8)).resize((iw, ih), Image.LANCZOS)
    ddx = int(round(2*math.sin(2*math.pi*0.45*t)))
    ddy = int(round(1.2*math.sin(2*math.pi*0.30*t+1.0) + 2.0*o))
    l = max(0, min((iw-W)//2 + ddx, iw-W))
    t0 = max(0, min((ih-H)//2 + ddy, ih-H))
    return np.asarray(img.crop((l, t0, l+W, t0+H)), dtype=np.uint8)


def main():
    cues = rhubarb_timeline(os.path.join(HERE, "voice.wav"))
    decode_pcm(os.path.join(HERE, "voice.mp3"))
    total = int(DUR*FPS)
    P = targets_per_frame(cues, total)
    P[int(4.15*FPS):] *= 0.0
    blink = np.zeros(total)
    for t0 in (1.35, 3.20):
        d = 0.16
        for f in range(int(t0*FPS), min(int((t0+d)*FPS), total)):
            blink[f] = math.sin(math.pi*(f/FPS - t0)/d)**2

    names = ["avatar_closed.jpg", "avatar_open.jpg", "avatar_wide.jpg",
             "avatar_round.jpg", "avatar_blink.jpg"]
    L = [fit(Image.open(os.path.join(HERE, n)).convert("RGB")) for n in names]
    m_mouth = ellipse_mask(MY0, MY1, MX0, MX1)
    m_eye = ellipse_mask(EY0, EY1, EX0, EX1)

    frames_dir = os.path.join(HERE, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for f in range(total):
        Image.fromarray(render_frame(f, P, blink[f], L, m_mouth, m_eye)) \
            .save(os.path.join(frames_dir, f"frame_{f:03d}.png"))
    np.savez(os.path.join(HERE, "plan.npz"), P=P, blink=blink)

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
    print("v4 rendered ->", out, "| visemes:", len(cues))


if __name__ == "__main__":
    main()
