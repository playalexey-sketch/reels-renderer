# Матрица липсинк-движков (open source)

Исходники: кадр `render/avatar_closed.jpg`, аудио `render/voice5.wav`

| Движок | Описание | Статус | Примечание |
|---|---|---|---|
| local-phoneme-v4 | фонемный риг губ v4 (Rhubarb + IDW-деформация) | DONE |  |
| wav2lip-pytorch | Wav2Lip (PyTorch, Rudrabha) — эталон точности синка | DONE | ok |
| wav2lip-onnx-hq | Wav2Lip ONNX + GFPGAN (instant-high/guzmanvitar) | SKIP | нет ONNX-весов (LFS заблокирован) |

| wav2lip-gfpgan-organic | Wav2Lip + GFPGAN (зона рта) + моргания/живая голова + unsharp | DONE | рабочая версия working-v2 |

| sadtalker-gfpgan | SadTalker (нейродвижение головы/мимика) + GFPGAN-резкость | DONE | рабочая версия working-v3 |
