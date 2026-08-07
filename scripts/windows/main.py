# -*- coding: utf-8 -*-
"""Reels Renderer (Windows, один клик через run.bat).
SadTalker (живая голова) + Wav2Lip (точный липсинк) + GFPGAN (резкость) + 50fps.
Свои файлы: положите рядом face.jpg и voice.mp3 — иначе возьмутся примеры из репо."""
import os, sys, glob, subprocess, shutil, urllib.request, zipfile, io

BASE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(BASE, "work")
GPU = "--cpu" not in sys.argv
FPS50 = "--fps50" in sys.argv

def log(*a): print("[рендер]", *a, flush=True)

def get(url, dst):
    if os.path.isfile(dst) and os.path.getsize(dst) > 100000:
        return
    log("скачиваю", os.path.basename(dst))
    with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)

def get_zip(url, dst_dir):
    flag = os.path.join(dst_dir, ".done")
    if os.path.isfile(flag):
        return
    log("скачиваю и распаковываю", os.path.basename(dst_dir))
    with urllib.request.urlopen(url) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
        z.extractall(dst_dir)
    open(flag, "w").write("ok")

def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()

def main():
    os.makedirs(WORK, exist_ok=True)
    FF = ffmpeg()
    face = os.path.join(BASE, "face.jpg")
    if not os.path.isfile(face):
        face = os.path.join(BASE, "..", "..", "render", "avatar_closed.jpg")
    audio = os.path.join(BASE, "voice.mp3")
    if not os.path.isfile(audio):
        audio = os.path.join(BASE, "..", "..", "render", "voice5.wav")
    audio = os.path.abspath(audio); face = os.path.abspath(face)

    # --- код моделей ---
    get_zip("https://codeload.github.com/OpenTalker/SadTalker/zip/refs/heads/master", os.path.join(WORK, "st"))
    get_zip("https://codeload.github.com/Rudrabha/Wav2Lip/zip/refs/heads/master", os.path.join(WORK, "w2l"))
    st = glob.glob(os.path.join(WORK, "st", "SadTalker-*"))[0]
    w2l = glob.glob(os.path.join(WORK, "w2l", "Wav2Lip-*"))[0]

    # --- веса (официальные, напрямую) ---
    ck = os.path.join(st, "checkpoints")
    get("https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/SadTalker_V0.0.2_256.safetensors", os.path.join(ck, "SadTalker_V0.0.2_256.safetensors"))
    get("https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/mapping_00109-model.pth.tar", os.path.join(ck, "mapping_00109-model.pth.tar"))
    get("https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/wav2lip.pth", os.path.join(ck, "wav2lip.pth"))
    get("https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/shape_predictor_68_face_landmarks.dat", os.path.join(ck, "shape_predictor_68_face_landmarks.dat"))
    get_zip("https://github.com/Winfredy/SadTalker/releases/download/v0.0.2/BFM_Fitting.zip", ck)
    gw = os.path.join(st, "gfpgan", "weights")
    os.makedirs(gw, exist_ok=True)
    get("https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth", os.path.join(gw, "alignment_WFLW_4HG.pth"))
    get("https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth", os.path.join(gw, "detection_Resnet50_Final.pth"))
    get("https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth", os.path.join(gw, "parsing_parsenet.pth"))
    get("https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth", os.path.join(gw, "GFPGANv1.4.pth"))

    env = dict(os.environ, PATH=os.path.dirname(FF) + os.pathsep + os.environ["PATH"])

    # --- шимы + инференс SadTalker ---
    shim = ("import numpy as np,sys,types;"
            "np.VisibleDeprecationWarning=getattr(np,'exceptions',np).VisibleDeprecationWarning;"
            "setattr(np,'float',float);setattr(np,'int',int);setattr(np,'complex',complex);"
            "import torchvision.transforms.functional as _F;"
            "_m=types.ModuleType('torchvision.transforms.functional_tensor');_m.rgb_to_grayscale=_F.rgb_to_grayscale;"
            "sys.modules.setdefault('torchvision.transforms.functional_tensor',_m);")
    dev = "" if GPU else "'--cpu',"
    cmd = [sys.executable, "-c", shim +
           "import runpy,sys,os;os.chdir(r'%s');sys.path.insert(0,'.');" % st +
           "sys.argv=['inference.py','--driven_audio',r'%s','--source_image',r'%s',"
           "'--checkpoint_dir','checkpoints','--result_dir','results','--preprocess','full',"
           "'--enhancer','none','--batch_size','8',%s];"
           "runpy.run_path('inference.py',run_name='__main__')" % (audio, face, dev)]
    log("SadTalker: генерация движения...")
    subprocess.run(cmd, check=True, env=env)
    raw = sorted(glob.glob(os.path.join(st, "results", "**", "*.mp4"), recursive=True))[-1]

    # --- Wav2Lip: точный липсинк поверх ---
    log("Wav2Lip: точная синхронизация рта...")
    cmd = [sys.executable, "-c",
           "import torch,runpy,sys,os;os.chdir(r'%s');sys.path.insert(0,'.');" % w2l +
           "sys.argv=['inference.py','--checkpoint_path','checkpoints/wav2lip.pth',"
           "'--face',r'%s','--audio',r'%s','--outfile',r'%s','--pads','0','20','0','0'];"
           "runpy.run_path('inference.py',run_name='__main__')" % (raw, audio, os.path.join(WORK, "sync.mp4"))]
    subprocess.run(cmd, check=True, env=env)
    synced = os.path.join(WORK, "sync.mp4")

    # --- GFPGAN: резкость лица ---
    log("GFPGAN: резкость...")
    vd = os.path.join(WORK, "fr")
    os.makedirs(vd, exist_ok=True)
    for p in glob.glob(vd + "/*.png"): os.remove(p)
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", synced, vd + "/f_%04d.png"], check=True)
    sharpen_py = r'''
import sys, glob, os, cv2, numpy as np
sys.path.insert(0, os.path.join(r"%s", "..", ".."))  # заглушка, не используется
import torch
from gfpgan import GFPGANer
restorer = GFPGANer(model_path=r"%s", upscale=1, arch="clean", channel_multiplier=2, device="cuda" if torch.cuda.is_available() else "cpu")
FY0, FY1, FX0, FX1 = 470, 690, 240, 520
yy, xx = np.mgrid[0:1376, 0:768].astype(np.float32)
d = np.sqrt(((yy-580)/130.0)**2 + ((xx-380)/150.0)**2)
mask = np.clip((1.15-d)/0.35*0.5, 0, 1)[..., None]
orig = cv2.imread(r"%s")
for p in sorted(glob.glob(r"%s" + "/f_*.png")):
    img = cv2.imread(p)
    img = cv2.resize(img, (768, 1376), interpolation=cv2.INTER_LANCZOS4)
    crop = img[FY0:FY1, FX0:FX1]
    small = cv2.resize(crop, (512, 512))
    _, _, enh = restorer.enhance(small, has_aligned=False, only_center_face=True, paste_back=True)
    src = cv2.resize(cv2.imread(r"%s")[FY0:FY1, FX0:FX1], (512, 512))
    det = src.astype(np.float32) - cv2.GaussianBlur(src, (0,0), 2.0).astype(np.float32)
    enh = np.clip(enh.astype(np.float32) + det*0.6, 0, 255).astype(np.uint8)
    enh = cv2.resize(enh, (FX1-FX0, FY1-FY0))
    img[FY0:FY1, FX0:FX1] = (img[FY0:FY1, FX0:FX1].astype(np.float32)*(1-mask) + enh*mask).astype(np.uint8)
    b = cv2.GaussianBlur(img, (0,0), 1.1)
    img = cv2.addWeighted(img, 1.25, b, -0.25, 0)
    cv2.imwrite(p, img)
print("резкость готова")
''' % (BASE, os.path.join(gw, "GFPGANv1.4.pth"), face, vd, face)
    subprocess.run([sys.executable, "-c", sharpen_py], check=True, env=env)

    out = os.path.join(BASE, "result.mp4")
    vf = "minterpolate=fps=50:mi_mode=mci:mc_mode=aobmc:vsbmc=1," if FPS50 else ""
    subprocess.run([FF, "-y", "-loglevel", "error", "-framerate", "25", "-i", vd + "/f_%04d.png",
                    "-i", audio, "-map", "0:v", "-map", "1:a",
                    "-vf", vf + "format=yuv420p", "-c:v", "libx264", "-crf", "18",
                    "-movflags", "+faststart", "-c:a", "aac", "-b:a", "128k", "-shortest", out],
                   check=True)
    log("ГОТОВО:", out)

if __name__ == "__main__":
    main()
