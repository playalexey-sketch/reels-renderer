# Матрица липсинк-движков (open source)

Исходники: кадр `render/avatar_closed.jpg`, аудио `render/voice.mp3`

| Движок | Описание | Статус | Примечание |
|---|---|---|---|
| local-phoneme-v4 | фонемный риг губ v4 (Rhubarb + IDW-деформация) | DONE |  |
| wav2lip-pytorch | Wav2Lip (PyTorch, Rudrabha) — эталон точности синка | SKIP | нет весов: ['wav2lip.pth', 's3fd.pth'] (GitHub LFS/HF заблокированы эгрессом песочницы) |
| wav2lip-onnx-hq | Wav2Lip ONNX + GFPGAN (instant-high/guzmanvitar) | SKIP | нет ONNX-весов (LFS заблокирован) |
