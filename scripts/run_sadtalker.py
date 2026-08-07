#!/usr/bin/env python3
"""SadTalker (natural head motion from photo+audio) + GFPGAN-резкость без детектора.
Движение — целиком от нейросети (никакой синтетики): голова, моргания, мимика."""
import subprocess, sys, os, glob
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TP = os.path.expanduser(os.environ.get("THIRDPARTY", "~/third_party"))
FF = subprocess.check_output([sys.executable, "-c",
    "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())"]).decode().strip()
sys.path.insert(0, HERE)
import organic_enhance as oe


def main(audio, out):
    st = os.path.join(TP, "SadTalker")
    env = dict(os.environ,
               TORCH_HOME=os.path.join(TP, "torchhome"),
               PATH=os.path.expanduser("~/.local/bin") + os.pathsep + os.environ["PATH"])
    res_dir = os.path.join(st, "results_reels")
    cmd = [sys.executable, "-c",
           "import torch,runpy,sys,numpy as np;"
           "_tl=torch.load;"
           "torch.load=lambda *a,**k:_tl(*((a[0],torch.device('cpu'))+a[2:]) if len(a)>1 else a,**{**k,'weights_only':False,'map_location':torch.device('cpu')});"
           "np.VisibleDeprecationWarning=getattr(np,'VisibleDeprecationWarning',None) or __import__('numpy').exceptions.VisibleDeprecationWarning;"
           "np.float=float;np.int=int;np.complex=complex;"
           f"sys.path.insert(0,'{HERE}');import st_patch;"
           "sys.argv=['inference.py',"
           f"'--driven_audio','{audio}',"
           f"'--source_image','{ROOT}/render/avatar_closed.jpg',"
           "'--checkpoint_dir','checkpoints',"
           f"'--result_dir','{res_dir}',"
           "'--preprocess','full',"
           "'--cpu',"
           "'--batch_size','1'];"
           "runpy.run_path('inference.py',run_name='__main__')"]
    subprocess.run(cmd, check=True, cwd=st, env=env)
    vids = sorted(glob.glob(res_dir + "/*/*.mp4"))
    assert vids, "SadTalker не выдал видео"
    raw = vids[-1]

    # пост-резкость: upscale + GFPGAN (зона рта) + детали с исходного фото
    vd = os.path.join(ROOT, "render", "stframes")
    os.makedirs(vd, exist_ok=True)
    for p in glob.glob(vd + "/*.png"):
        os.remove(p)
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", raw, os.path.join(vd, "f_%03d.png")],
                   check=True, capture_output=True)
    frames = sorted(glob.glob(vd + "/f_*.png"))
    model = oe.load_gfpgan()
    ff = oe.face_feather()
    orig = cv2.imread(os.path.join(ROOT, "render", "avatar_closed.jpg"))
    for p in frames:
        img = cv2.imread(p)
        img = cv2.resize(img, (768, 1376), interpolation=cv2.INTER_LANCZOS4)
        img = oe.enhance_frame(model, img, ff, orig=orig)
        blur = cv2.GaussianBlur(img, (0, 0), 1.1)
        img = cv2.addWeighted(img, 1.25, blur, -0.25, 0)
        cv2.imwrite(p, img)
    cmd = [FF, "-y", "-loglevel", "error", "-framerate", "25",
           "-i", os.path.join(vd, "f_%03d.png"), "-i", audio,
           "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-profile:v", "main",
           "-level", "4.0", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow",
           "-movflags", "+faststart", "-c:a", "aac", "-b:a", "128k", "-shortest", out]
    subprocess.run(cmd, check=True, capture_output=True)
    print("sadtalker готов ->", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "render", "voice5.wav"),
         sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "render", "preview_sadtalker.mp4"))
