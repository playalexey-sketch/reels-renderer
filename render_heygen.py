#!/usr/bin/env python3
import json, os, sys, time, datetime, urllib.request, urllib.error

API = "https://api.heygen.com"
KEY = os.environ["HEYGEN_API_KEY"]

def status(stage, **kw):
    d = {"stage": stage, "updated": datetime.datetime.utcnow().isoformat() + "Z"}
    d.update(kw)
    json.dump(d, open("status.json", "w"), ensure_ascii=False, indent=2)
    print("STATUS:", json.dumps(d, ensure_ascii=False), flush=True)

def req(method, url, data=None):
    r = urllib.request.Request(url, method=method,
        data=json.dumps(data).encode() if data else None,
        headers={"X-Api-Key": KEY, "Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=120) as resp:
        return json.loads(resp.read())

status("queued")
body = json.load(open("task.json", encoding="utf-8"))

try:
    res = req("POST", f"{API}/v2/video/generate", body)
except urllib.error.HTTPError as e:
    status("error", detail=f"HTTP {e.code}: {e.read().decode()[:800]}")
    sys.exit(1)
except Exception as e:
    status("error", detail=repr(e))
    sys.exit(1)

vid = res["data"]["video_id"]
status("rendering", video_id=vid)

deadline = time.time() + 1500
url = None
while time.time() < deadline:
    time.sleep(20)
    try:
        st = req("GET", f"{API}/v1/video_status.get?video_id={vid}")["data"]
    except Exception as e:
        print("poll error:", repr(e), flush=True)
        continue
    status("rendering", video_id=vid, heygen_status=st["status"])
    if st["status"] == "completed":
        url = st["video_url"]
        break
    if st["status"] in ("failed", "error"):
        status("error", video_id=vid, detail=json.dumps(st)[:800])
        sys.exit(1)

if not url:
    status("error", video_id=vid, detail="timeout 25min")
    sys.exit(1)

urllib.request.urlretrieve(url, "reels1_raw.mp4")
status("downloaded", video_id=vid)
