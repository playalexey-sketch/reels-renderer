# Паспорт версий видео (reels-renderer)

| Версия | Файл у пользователя | Пайплайн | Статус |
|---|---|---|---|
| v1 | reels_v1_sharp.mp4 | SadTalker 256 + GFPGAN (рот) | база |
| v2 | (kaggle, ранее) | + Wav2Lip | промежуточная |
| v3 | reels_v3_512_facesharp.mp4 | SadTalker 512 + GFPGAN upscale=2 всё лицо | хорошая |
| **v4** | **reels_v4_512_w2l_50fps.mp4** | **v3 + Wav2Lip + 50 fps** | **текущая лучшая (база для улучшений)** |
| v5 | (следующая) | v4 + анти-фликер + текстура кожи + микро-резкость рта | в работе |

Эталонный кадр v4: `v4_quality_reference_frame.png` (кадр той же сцены).
Генератор версий: `scripts/colab_final.ipynb`; теги: `v4-approved`, далее `v5-...`.
