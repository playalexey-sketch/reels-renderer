# Кладите этот файл в папку с тремя скачанными файлами и запускайте: python split_for_upload.py
import pathlib
JOBS = [
    ("SadTalker_V0.0.2_256.safetensors", "sadt256"),
    ("BFM_Fitting.zip", "bfmfit"),
    ("resnet50-19c8e357.pth", "resnet50"),
]
for f, p in JOBS:
    path = pathlib.Path(f)
    if not path.exists():
        print("НЕ НАЙДЕН ФАЙЛ:", f)
        continue
    d = path.read_bytes()
    n = len(d) // 24000000 + 1
    for i in range(n):
        pathlib.Path("%s.part_%02d" % (p, i)).write_bytes(d[i*24000000:(i+1)*24000000])
    print(f, "-> chunks:", n)
print("ГОТОВО: загружите все файлы .part_ на GitHub")
