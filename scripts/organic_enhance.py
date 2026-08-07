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


def enhance_frame(model, img, feather, orig=None):
    crop = img[FY0:FY0+FS, FX0:FX0+FS]
    small = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LINEAR)
    x = (small.astype(np.float32)/255.0 - 0.5)/0.5
    t = torch.from_numpy(x.transpose(2, 0, 1))[None]
    with torch.no_grad():
        out = model(t)[0]
    enh = out[0].numpy().transpose(1, 2, 0)
    enh = np.clip((enh*0.5 + 0.5)*255, 0, 255).astype(np.uint8)
    # детали берём с исходного РЕЗКОГО фото, если оно передано (апскейл-случай)
    src = orig[FY0:FY0+FS, FX0:FX0+FS] if orig is not None else crop
    src512 = cv2.resize(src, (512, 512), interpolation=cv2.INTER_LINEAR)
    detail = src512.astype(np.float32) - cv2.GaussianBlur(src512, (0, 0), 2.0).astype(np.float32)
    enh = np.clip(enh.astype(np.float32) + detail*0.6, 0, 255).astype(np.uint8)
    enh = cv2.resize(enh, (FS, FS), interpolation=cv2.INTER_LINEAR)
    blended = enh.astype(np.float32)
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
    """Маска в координатах кропа FSxFS: GFPGAN только на рту/нижней трети лица,
    остальное лицо остаётся исходным (резким)."""
    yy, xx = np.mgrid[0:FS, 0:FS].astype(np.float32)
    # рот в координатах кропа: центр (366, 494), радиусы
    cy, cx = 620 - FY0, 400 - FX0
    d = np.sqrt(((yy-cy)/150.0)**2 + ((xx-cx)/170.0)**2)
    mouth = np.clip((1.15-d)/0.35, 0, 1)
    # лёгкая общая доводка нижней половины лица
    d2 = np.sqrt(((yy-(680-FY0))/260.0)**2 + ((xx-cx)/240.0)**2)
    low = 0.3*np.clip((1.1-d2)/0.4, 0, 1)
    return np.clip(np.maximum(mouth, low), 0, 1)[..., None]


def person_mask(img0):
    mask = np.zeros(img0.shape[:2], np.uint8)
    rect = (50, 60, 668, 1316)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(img0, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    m = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.float32)
    m[1000:, :] = 1.0                      # торс целиком
    m[:, :12] = 0; m[:, -12:] = 0; m[:12, :] = 0
    m = cv2.GaussianBlur(m, (0, 0), 6)      # перо
    return m[..., None]


def head_transform(t, env):
    ang = 0.8*math.sin(2*math.pi*0.19*t + 0.7) + 0.35*math.sin(2*math.pi*0.41*t + 2.1)
    tx = 2.4*math.sin(2*math.pi*0.16*t + 1.3) + 0.8*math.sin(2*math.pi*0.37*t)
    ty = 2.6*env + 1.4*math.sin(2*math.pi*0.24*t) + 0.6*math.sin(2*math.pi*0.52*t + 1.0)
    sc = 1.0 + 0.003*math.sin(2*math.pi*0.12*t + 2.0)
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
    pm = person_mask(cv2.imread(frames[0]))

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
        # живая голова: движется только человек, фон статичен
        ang, tx, ty, sc = head_transform(t, float(rms[min(int(t*100), len(rms)-1)]))
        Hh, Ww = img.shape[:2]
        M = cv2.getRotationMatrix2D((Ww/2, 500), ang, sc)
        M[0, 2] += tx
        M[1, 2] += ty
        warped = cv2.warpAffine(img, M, (Ww, Hh), borderMode=cv2.BORDER_REPLICATE)
        img = (warped*pm + img*(1-pm)).astype(np.uint8)
        # финальный unsharp — хрусткость деталей
        blur = cv2.GaussianBlur(img, (0, 0), 1.2)
        img = cv2.addWeighted(img, 1.35, blur, -0.35, 0)
        cv2.imwrite(p, img)

    cmd = [FF, "-y", "-loglevel", "error", "-framerate", str(fps),
           "-i", os.path.join(vd, "f_%03d.png"), "-i", src,
           "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-profile:v", "main",
           "-level", "4.0", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow",
           "-movflags", "+faststart", "-c:a", "aac", "-b:a", "128k", "-shortest", dst]
    subprocess.run(cmd, check=True, capture_output=True)
    print("органик+gfpgan готов ->", dst)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "render/matrix/wav2lip-pytorch_raw.mp4"),
         sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "render/preview_organic.mp4"),
         sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, "render/avatar_blink.jpg"))
