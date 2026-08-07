#!/usr/bin/env python3
"""Прогон исходных аудио+кадр через все доступные open-source движки липсинка.
Результаты в render/matrix/<engine>.mp4 + STATUS.md. Недоступные движки помечаются SKIP с причиной."""
import os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TP = Path(os.environ.get("THIRDPARTY", Path.home() / "third_party"))
FF = subprocess.check_output([sys.executable, "-c",
    "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())"]).decode().strip()
MATRIX = ROOT / "render" / "matrix"
FACE = ROOT / "render" / "avatar_closed.jpg"
AUDIO = next(p for p in (ROOT / "render" / "voice5.wav", ROOT / "render" / "voice_xtts.wav",
                         ROOT / "render" / "voice.mp3") if p.exists())


def run_local_viseme():
    subprocess.run([sys.executable, str(ROOT / "render" / "make_video_v4.py")], check=True)
    shutil.copy(ROOT / "render" / "talking_avatar_5s.mp4", MATRIX / "local-phoneme-v4.mp4")


def run_wav2lip_pytorch():
    w = TP / "Wav2Lip"
    need = [w / "checkpoints" / "wav2lip.pth", w / "face_detection" / "detection" / "sfd" / "s3fd.pth"]
    miss = [p.name for p in need if not p.exists() or p.stat().st_size < 1000000]
    if miss:
        return False, f"нет весов: {miss} (GitHub LFS/HF заблокированы эгрессом песочницы)"
    env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + os.pathsep + os.environ["PATH"])
    cmd = [sys.executable, "-c",
           "import torch,runpy,sys;_tl=torch.load;torch.load=lambda *a,**k:_tl(*a,**{**k,'weights_only':False});"
           f"sys.argv=['inference.py','--checkpoint_path','{need[0]}','--face','{FACE}',"
           f"'--audio','{AUDIO}','--outfile','{MATRIX}/wav2lip-pytorch.mp4','--pads','0','20','0','0','--static','True'];"
           f"runpy.run_path('{w}/inference.py',run_name='__main__')"]
    subprocess.run(cmd, check=True, cwd=w, env=env)
    return True, "ok"


def run_wav2lip_onnx_hq():
    w = TP / "w2lonnx"
    models = w / "src" / "lipsync" / "models" / "wav2lip" / "models"
    miss = [p.name for p in models.glob("*.onnx")] and \
           [p.name for p in models.glob("*.onnx") if p.stat().st_size < 1000000] or []
    if not list(models.glob("*.onnx")) or miss:
        return False, "нет ONNX-весов (LFS заблокирован)"
    cmd = [sys.executable, "-m", "lipsync.scripts.wav2lip_inference",
           "--video", str(FACE), "--audio", str(AUDIO), "--output", str(MATRIX / "wav2lip-onnx-hq.mp4")]
    subprocess.run(cmd, check=True, cwd=w / "src")
    return True, "ok"


def main():
    MATRIX.mkdir(parents=True, exist_ok=True)
    engines = [
        ("local-phoneme-v4", "фонемный риг губ v4 (Rhubarb + IDW-деформация)", run_local_viseme),
        ("wav2lip-pytorch", "Wav2Lip (PyTorch, Rudrabha) — эталон точности синка", run_wav2lip_pytorch),
        ("wav2lip-onnx-hq", "Wav2Lip ONNX + GFPGAN (instant-high/guzmanvitar)", run_wav2lip_onnx_hq),
    ]
    rows = []
    for name, desc, fn in engines:
        try:
            res = fn()
            ok, info = (True, "") if res is None else res
            rows.append((name, desc, "DONE" if ok else "SKIP", info))
        except Exception as e:
            rows.append((name, desc, "ERROR", str(e)[:200]))
    md = ["# Матрица липсинк-движков (open source)", "",
          f"Исходники: кадр `{FACE.relative_to(ROOT)}`, аудио `{AUDIO.relative_to(ROOT)}`", ""]
    md.append("| Движок | Описание | Статус | Примечание |")
    md.append("|---|---|---|---|")
    for name, desc, st, info in rows:
        md.append(f"| {name} | {desc} | {st} | {info} |")
    (MATRIX / "STATUS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
