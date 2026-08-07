#!/usr/bin/env python3
"""Клиент проверенного сервиса Coqui XTTS v2 (клонирование голоса, ru).

Сервис пользователя (по умолчанию): http://195.209.214.155:9090
Использование:
  python3 scripts/xtts_client.py health
  python3 scripts/xtts_client.py upload --src my_voice.wav [--name ivan]
  python3 scripts/xtts_client.py clone --text "Привет! ..." --voice render/voices/ivan.wav --out render/voice_xtts.wav
"""
import argparse, json, os, shutil, sys, time, uuid, urllib.request, urllib.error
from pathlib import Path

DEFAULT_URL = os.environ.get("XTTS_URL", "http://195.209.214.155:9090")
ROOT = Path(__file__).resolve().parent.parent
VOICES = ROOT / "render" / "voices"


def health(url=DEFAULT_URL, timeout=8):
    for probe in ("/", "/docs", "/xtts_ru.wav"):
        try:
            with urllib.request.urlopen(url + probe, timeout=timeout) as r:
                if r.status == 200:
                    return True
        except Exception:
            continue
    return False


def upload_voice(src: str, name=None) -> Path:
    VOICES.mkdir(parents=True, exist_ok=True)
    name = name or f"voice_{int(time.time())}"
    dst = VOICES / f"{name}{os.path.splitext(src)[1] or '.wav'}"
    shutil.copy(src, dst)
    print(json.dumps({"uploaded": str(dst)}))
    return dst


def _post_json(url, payload, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def _post_multipart(url, fields, filepath, timeout):
    boundary = uuid.uuid4().hex
    body = b""
    for k, v in fields.items():
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    fn = os.path.basename(filepath)
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"voice_file\"; "
             f"filename=\"{fn}\"\r\nContent-Type: audio/wav\r\n\r\n").encode()
    body += Path(filepath).read_bytes() + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def _looks_audio(b: bytes) -> bool:
    return len(b) > 8000 and (b[:4] == b"RIFF" or b[:3] == b"ID3" or b[:2] == b"\xff\xfb")


def clone_speech(text: str, voice_path: str, out: str, url=DEFAULT_URL, timeout=600) -> str:
    """Прогон текста через XTTS v2 с клонированием голоса. Пробует несколько API-форматов."""
    strategies = [
        ("multipart /tts_with_voice_clone",
         lambda: _post_multipart(f"{url}/tts_with_voice_clone",
                                 {"text": text, "language": "ru"}, voice_path, timeout)),
        ("json /tts_with_voice_clone",
         lambda: _post_json(f"{url}/tts_with_voice_clone",
                            {"text": text, "language": "ru", "voice_file": str(voice_path)}, timeout)),
        ("json /v1/tts_with_voice_clone",
         lambda: _post_json(f"{url}/v1/tts_with_voice_clone",
                            {"text": text, "language": "ru", "voice_file": str(voice_path)}, timeout)),
        ("json /tts_to_speaker",
         lambda: _post_json(f"{url}/tts_to_speaker",
                            {"text": text, "language": "ru", "speaker_id": "xtts_ru"}, timeout)),
    ]
    errors = []
    for name, fn in strategies:
        try:
            status, body = fn()
            if status == 200 and _looks_audio(body):
                Path(out).parent.mkdir(parents=True, exist_ok=True)
                Path(out).write_bytes(body)
                return name
            if status == 200:  # возможно, json с путем/ base64
                try:
                    d = json.loads(body)
                    if isinstance(d, dict) and "audio" in d:
                        import base64
                        Path(out).write_bytes(base64.b64decode(d["audio"]))
                        return name + " (json/base64)"
                except Exception:
                    pass
            errors.append(f"{name}: status={status}, not-audio")
        except Exception as e:
            errors.append(f"{name}: {e}")
    raise RuntimeError("XTTS clone failed:\n" + "\n".join(errors))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("health")
    h.add_argument("--url", default=DEFAULT_URL)
    u = sub.add_parser("upload")
    u.add_argument("--src", required=True)
    u.add_argument("--name")
    c = sub.add_parser("clone")
    c.add_argument("--text", required=True)
    c.add_argument("--voice", required=True)
    c.add_argument("--out", default="render/voice_xtts.wav")
    c.add_argument("--url", default=DEFAULT_URL)
    args = ap.parse_args()

    if args.cmd == "health":
        ok = health(args.url)
        print(json.dumps({"url": args.url, "reachable": ok}))
        sys.exit(0 if ok else 3)
    if args.cmd == "upload":
        upload_voice(args.src, args.name)
        return
    if args.cmd == "clone":
        used = clone_speech(args.text, args.voice, args.out, args.url)
        print(json.dumps({"out": args.out, "strategy": used}))


if __name__ == "__main__":
    main()
