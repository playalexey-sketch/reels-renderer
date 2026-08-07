#!/usr/bin/env python3
"""GFPGAN-доводка лица в wav2lip-видео: резкие губы/зубы/кожа.
Без facexlib-детекции (лицо фронтальное, кроп фиксированный) — CPU-OK."""
import subprocess, sys, os, glob
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FF = subprocess.check_output([sys.executable, "-c",
    "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())"]).decode().strip()
GFPGAN_PTH = os.path.expanduser("~/third_party/GFPGANv1.4.pth")
# кроп лица в координатах 768x1376 (квадрат 700)
FX0, FY0, FS = 34, 130, 700


def main(src, dst):
    from gfpgan import GFPGANer
    restorer = GFPGANer(model_path=GFPGAN_PTH, upscale=1, arch="clean", channel_multiplier=2)

    vd = os.path.join(ROOT, "render", "gframes")
    os.makedirs(vd, exist_ok=True)
    for p in glob.glob(vd + "/*.png"):
        os.remove(p)
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", src, os.path.join(vd, "f_%03d.png")],
                   check=True, capture_output=True)
    frames = sorted(glob.glob(vd + "/f_*.png"))
    # перо для вклейки лица обратно
    yy, xx = np.mgrid[0:FS, 0:FS].astype(np.float32)
    d = np.sqrt(((yy - FS/2)/(FS/2))**2 + ((xx - FS/2)/(FS/2))**2)
    feather = np.clip((1.05 - d)/0.30, 0, 1)[..., None]

    for p in frames:
        img = cv2.imread(p)
        crop = img[FY0:FY0+FS, FX0:FX0+FS]
        small = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LINEAR)
        _, _, enh = restorer.enhance(small, has_aligned=False, only_center_face=True,
                                     paste_back=True)
        enh = cv2.resize(enh, (FS, FS), interpolation=cv2.INTER_LINEAR)
        img[FY0:FY0+FS, FX0:FX0+FS] = (crop*(1-feather) + enh*feather).astype(np.uint8)
        cv2.imwrite(p, img)

    cmd = [FF, "-y", "-loglevel", "error", "-framerate", "25",
           "-i", os.path.join(vd, "f_%03d.png"), "-i", src,
           "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-profile:v", "main",
           "-level", "4.0", "-pix_fmt", "yuv420p", "-crf", "20",
           "-movflags", "+faststart", "-c:a", "aac", "-b:a", "128k", "-shortest", dst]
    subprocess.run(cmd, check=True, capture_output=True)
    print("enhanced ->", dst)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "render/matrix/wav2lip-pytorch_raw.mp4"),
         sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "render/talking_avatar_5s.mp4"))
