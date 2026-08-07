#!/usr/bin/env python3
"""Органичный апгрейд wav2lip-видео: GFPGAN-резкость (без facexlib-детектора) +
естественные моргания + лёгкие покачивания/наклоны головы (живое лицо)."""
import subprocess, sys, os, glob, math
import numpy as np
import cv2
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FF = subprocess.check_output([sys.executable, "-c",
    "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())"]).decode().strip()
GFPGAN_PTH = os.path.expanduser("~/third_party/GFPGANv1.4.pth")
FX0, FY0, FS = 34, 130, 700
# глаза в исходных координатах 768x1376
EY0, EY1, EX0, EX1 = 395, 470, 265, 540


def load_gfpgan():
    from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
    model = GFPGANv1Clean(out_size=512, num_style_feat=512, channel_multiplier=2,
                          decoder_load_path=None, fix_decoder=False, num_mlp=8,
                          input_is_latent=True, different_w=True, narrow=1, sft_half=True)
    ckpt = torch.load(GFPGAN_PTH, map_location="cpu", weights_only=False)
    sd = ckpt.get("params_ema", ckpt.get("params", ckpt))
    model.load_state_dict(sd)
    model.eval()
    return model


def enhance_frame(model, img, feather):
    crop = img[FY0:FY0+FS, FX0:FX0+FS]
    small = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LINEAR)
    x = (small.astype(np.float32)/255.0 - 0.5)/0.5
    t = torch.from_numpy(x.transpose(2, 0, 1))[None]
    with torch.no_grad():
        out = model(t)[0]
    enh = out[0].numpy().transpose(1, 2, 0)
    enh = np.clip((enh*0.5 + 0.5)*255, 0, 255).astype(np.uint8)
    enh = cv2.resize(enh, (FS, FS), interpolation=cv2.INTER_LINEAR)
    blended = (crop.astype(np.float32)*0.25 + enh.astype(np.float32)*0.75)
    img2 = img.copy()
    img2[FY0:FY0+FS, FX0:FX0+FS] = (img2[FY0:FY0+FS, FX0:FX0+FS].astype(np.float32)*(1-feather)
                                    + blended*feather).astype(np.uint8)
    return img2


def eye_feather():
    m = np.zeros((1376, 768), np.float32)
    yy, xx = np.mgrid[0:1376, 0:768].astype(np.float32)
    cy, cx = (EY0+EY1)/2, (EX0+EX1)/2
    ry, rx = (EY1-EY0)/2+8, (EX1-EX0)/2+10
    d = np.sqrt(((yy-cy)/ry)**2 + ((xx-cx)/rx)**2)
    return np.clip((1.1-d)/0.35, 0, 1)[..., None]


def face_feather():
    yy, xx = np.mgrid[0:FS, 0:FS].astype(np.float32)
    d = np.sqrt(((yy-FS/2)/(FS/2))**2 + ((xx-FS/2)/(FS/2))**2)
    return np.clip((1.05-d)/0.30, 0, 1)[..., None]


def head_transform(t, env):
    ang = 0.9*math.sin(2*math.pi*0.21*t + 0.7) + 0.45*math.sin(2*math.pi*0.43*t)
    tx = 3.0*math.sin(2*math.pi*0.17*t + 1.3)
    ty = 2.2*math.sin(2*math.pi*0.25*t) + 2.5*env
    sc = 1.0 + 0.004*math.sin(2*math.pi*0.13*t + 2.0)
    return ang, tx, ty, sc


def main(src, dst, blink_img_path):
    model = load_gfpgan()
    ff = face_feather()
    ef = eye_feather()
    blink = cv2.imread(blink_img_path)

    vd = os.path.join(ROOT, "render", "gframes")
    os.makedirs(vd, exist_ok=True)
    for p in glob.glob(vd + "/*.png"):
        os.remove(p)
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", src, os.path.join(vd, "f_%03d.png")],
                   check=True, capture_output=True)
    frames = sorted(glob.glob(vd + "/f_*.png"))
    n = len(frames)
    fps = 25

    # огибающая для кивка
    import wave
    wav = os.path.join(ROOT, "render", "voice.wav")
    if not os.path.exists(wav):
        subprocess.run([FF, "-y", "-loglevel", "error", "-i",
                        os.path.join(ROOT, "render", "voice.mp3"), "-ar", "16000", "-ac", "1", wav],
                       check=True, capture_output=True)
    with wave.open(wav) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)/32768.0
    hop = 160
    rms = np.array([np.sqrt(np.mean(pcm[i*hop:(i+1)*hop]**2)+1e-9) for i in range(len(pcm)//hop)])
    rms = np.convolve(rms, np.ones(8)/8, mode="same")
    rms = np.clip(rms/max(rms.max(), 1e-6), 0, 1)

    for i, p in enumerate(frames):
        t = i/fps
        img = cv2.imread(p)
        img = enhance_frame(model, img, ff)
        # моргания
        for t0 in (1.15, 2.75, 4.05):
            d = 0.15
            if t0 <= t <= t0+d:
                b = math.sin(math.pi*(t-t0)/d)**2
                img = (img.astype(np.float32)*(1-ef*b) + blink.astype(np.float32)*(ef*b)).astype(np.uint8)
        # живая голова
        ang, tx, ty, sc = head_transform(t, float(rms[min(int(t*100), len(rms)-1)]))
        Hh, Ww = img.shape[:2]
        M = cv2.getRotationMatrix2D((Ww/2, 500), ang, sc)
        M[0, 2] += tx
        M[1, 2] += ty
        img = cv2.warpAffine(img, M, (Ww, Hh), borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite(p, img)

    cmd = [FF, "-y", "-loglevel", "error", "-framerate", str(fps),
           "-i", os.path.join(vd, "f_%03d.png"), "-i", src,
           "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-profile:v", "main",
           "-level", "4.0", "-pix_fmt", "yuv420p", "-crf", "20",
           "-movflags", "+faststart", "-c:a", "aac", "-b:a", "128k", "-shortest", dst]
    subprocess.run(cmd, check=True, capture_output=True)
    print("organic+gfpgan ->", dst)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "render/matrix/wav2lip-pytorch_raw.mp4"),
         sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "render/preview_organic.mp4"),
         sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, "render/avatar_blink.jpg"))
