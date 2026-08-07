#!/usr/bin/env python3
"""Конвейер улучшения: жду/перезапускаю рендер Марии, затем варианты
(V1 резкость, V2 +Wav2Lip-синхронизация рта, V3 +50fps), выбираю лучший по метрикам
(синхронность/плавность/резкость), то же для RU-версии, коммичу превью и ОТЧЁТ в git."""
import os, sys, glob, subprocess, time, shutil
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import organic_enhance as oe

FF = os.path.expanduser("~/.local/bin/ffmpeg")
TP = os.path.expanduser("~/third_party")
LOG = lambda *a: print("[конвейер]", *a, flush=True)


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def extract(mp4, vd):
    os.makedirs(vd, exist_ok=True)
    for p in glob.glob(vd + "/*.png"):
        os.remove(p)
    run([FF, "-y", "-loglevel", "error", "-i", mp4, vd + "/f_%04d.png"], check=True)
    return sorted(glob.glob(vd + "/f_*.png"))


def encode(frames_dir, audio, out, extra_vf=None):
    cmd = [FF, "-y", "-loglevel", "error", "-framerate", "25", "-i", frames_dir + "/f_%04d.png"]
    if extra_vf:
        cmd += ["-vf", extra_vf]
    cmd += ["-i", audio, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-profile:v", "main",
            "-level", "4.0", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
            "-movflags", "+faststart", "-c:a", "aac", "-b:a", "128k", "-shortest", out]
    run(cmd, check=True)


def upscale_dir(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for p in glob.glob(dst_dir + "/*.png"):
        os.remove(p)
    for p in sorted(glob.glob(src_dir + "/f_*.png")):
        img = cv2.imread(p)
        img = cv2.resize(img, (768, 1376), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(os.path.join(dst_dir, os.path.basename(p)), img)


def sharpen_dir(frames, orig_path):
    model = oe.load_gfpgan()
    ff = oe.face_feather()
    orig = cv2.imread(orig_path)
    for p in frames:
        img = cv2.imread(p)
        img = oe.enhance_frame(model, img, ff, orig=orig)
        b = cv2.GaussianBlur(img, (0, 0), 1.1)
        img = cv2.addWeighted(img, 1.25, b, -0.25, 0)
        cv2.imwrite(p, img)


def wav2lip_pass(video, audio, out):
    env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + os.pathsep + os.environ["PATH"])
    cmd = [sys.executable, "-c",
           "import torch,runpy,sys;_tl=torch.load;"
           "torch.load=lambda *a,**k:_tl(*a,**{**k,'weights_only':False,'map_location':torch.device('cpu')});"
           "sys.argv=['inference.py','--checkpoint_path','checkpoints/wav2lip.pth',"
           f"'--face','{video}','--audio','{audio}','--outfile','{out}','--pads','0','20','0','0'];"
           "runpy.run_path('inference.py',run_name='__main__')"]
    r = run(cmd, cwd=os.path.join(TP, "Wav2Lip"), env=env)
    if r.returncode != 0:
        LOG("Wav2Lip ошибка:", r.stderr[-600:])
        return False
    return True


def envelope_of(audio):
    run([FF, "-y", "-loglevel", "error", "-i", audio, "-ar", "16000", "-ac", "1", "/tmp/env.wav"])
    import wave
    with wave.open("/tmp/env.wav") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
    hop = 160
    rms = np.array([np.sqrt(np.mean(pcm[i*hop:(i+1)*hop] ** 2) + 1e-9) for i in range(len(pcm)//hop)])
    return np.convolve(rms, np.ones(5)/5, mode="same")


def metrics(video, audio):
    vd = "/tmp/mtr"
    fs = extract(video, vd)
    mo, sharp = [], []
    for p in fs:
        g = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY).astype(np.float32)
        H, W = g.shape
        mo.append(g)
        sharp.append(cv2.Laplacian(g[int(H*0.1):int(H*0.6), int(W*0.15):int(W*0.85)].astype(np.uint8), cv2.CV_64F).var())
    en = np.array([np.abs(mo[i+1]-mo[i]).mean() for i in range(len(mo)-1)])
    env = envelope_of(audio)
    n = min(len(en), len(env)//4)
    e = np.array([env[i*4:i*4+4].mean() for i in range(n)])
    sync = float(np.corrcoef(en[:n], e)[0, 1]) if n > 10 else 0.0
    smooth = float(1.0 / (1.0 + np.median(np.abs(np.diff(en)))*30))
    sharp_v = float(np.mean(sharp[::5]))
    return {"sync": round(sync, 3), "smooth": round(smooth, 3), "sharp": round(sharp_v, 1)}


def score(m):
    return 0.45*m["sync"] + 0.25*m["smooth"] + 0.30*min(m["sharp"]/120.0, 1.5)


def wait_or_render_maria(target, audio15):
    for _ in range(200):
        if os.path.isfile(target):
            return True
        time.sleep(50)
    LOG("рендер Марии не появился — запускаю сам")
    r = run([sys.executable, os.path.join(ROOT, "scripts", "run_sadtalker.py"), audio15, target],
            env=dict(os.environ, RR_BATCH="1"))
    return r.returncode == 0 and os.path.isfile(target)


def build_best(tag, src_video, audio, orig_img, out_best):
    work = f"/tmp/night_{tag}"
    os.makedirs(work, exist_ok=True)
    extract(src_video, work + "/raw")
    upscale_dir(work + "/raw", work + "/up")
    v1 = work + "/v1"
    if os.path.isdir(v1):
        shutil.rmtree(v1)
    shutil.copytree(work + "/up", v1)
    sharpen_dir(sorted(glob.glob(v1 + "/f_*.png")), orig_img)
    encode(v1, audio, work + "/v1.mp4")
    m1 = metrics(work + "/v1.mp4", audio)
    LOG(tag, "V1:", m1)
    best, bm = "v1", m1
    if wav2lip_pass(work + "/v1.mp4", audio, work + "/w2l.mp4"):
        v2 = extract(work + "/w2l.mp4", work + "/v2")
        sharpen_dir(v2, orig_img)
        encode(work + "/v2", audio, work + "/v2.mp4")
        m2 = metrics(work + "/v2.mp4", audio)
        LOG(tag, "V2:", m2)
        if score(m2) > score(bm):
            best, bm = "v2", m2
        r = run([FF, "-y", "-loglevel", "error", "-i", work + f"/{best}.mp4",
                 "-vf", "minterpolate=fps=50:mi_mode=mci:mc_mode=aobmc:vsbmc=1",
                 "-c:v", "libx264", "-profile:v", "main", "-pix_fmt", "yuv420p", "-crf", "18",
                 "-movflags", "+faststart", "-c:a", "aac", "-b:a", "128k", work + "/v3.mp4"])
        if r.returncode == 0 and os.path.isfile(work + "/v3.mp4"):
            m3 = metrics(work + "/v3.mp4", audio)
            LOG(tag, "V3:", m3)
            if score(m3) > score(bm):
                best, bm = "v3", m3
    shutil.copy(work + f"/{best}.mp4", out_best)
    LOG(tag, "ЛУЧШИЙ:", best, bm)
    return best, bm


def main():
    maria_src = os.path.join(ROOT, "render", "preview_maria.mp4")
    maria15 = os.path.join(ROOT, "render", "voices", "maria15.wav")
    ru_src = os.path.join(ROOT, "render", "preview_sadtalker.mp4")
    voice5 = os.path.join(ROOT, "render", "voice5.wav")
    orig = os.path.join(ROOT, "render", "avatar_closed.jpg")

    if not wait_or_render_maria(maria_src, maria15):
        LOG("НЕ УДАЛОСЬ получить рендер Марии")
        return
    rep = {"maria": build_best("maria", maria_src, maria15, orig,
                               os.path.join(ROOT, "render", "preview_maria_best.mp4")),
           "ru": build_best("ru", ru_src, voice5, orig,
                            os.path.join(ROOT, "render", "preview_ru_best.mp4"))}
    with open(os.path.join(ROOT, "render", "NIGHT_REPORT.md"), "w") as f:
        f.write("# Отчёт авто-улучшения\n\n| Версия | вариант | синхрон | плавность | резкость |\n|---|---|---|---|---|\n")
        for k, (b, m) in rep.items():
            f.write(f"| {k} | {b} | {m['sync']} | {m['smooth']} | {m['sharp']} |\n")
        f.write("\nV1=резкость GFPGAN; V2=+Wav2Lip; V3=+50fps.\n")
    LOG("отчёт готов")
    run(["git", "add", "-A"], cwd=ROOT)
    run(["git", "commit", "-q", "-m", "авто-улучшение: SadTalker+Wav2Lip+GFPGAN(+50fps), выбор по метрикам"], cwd=ROOT)
    r = run(["git", "push", "-q", "origin", "arena/019fdb1c-reels-renderer"], cwd=ROOT)
    LOG("пуш:", r.returncode)


if __name__ == "__main__":
    main()
