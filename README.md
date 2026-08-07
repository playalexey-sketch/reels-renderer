# reels-renderer
Рендер-исполнитель рилсов: HeyGen API + ffmpeg монтаж (субтитры, SFX, CTA)

## Делевери результатов
Готовые результаты (mp4, аватары, озвучка) кладутся в репозиторий и пушатся
в рабочую ветку; в ответе всегда сразу даётся ссылка на файл на GitHub
(`https://github.com/playalexey-sketch/reels-renderer/blob/<ветка>/<путь>`).
Промежуточные артефакты (кадры, wav, check-скрины) игнорируются через `.gitignore`.

## Голос аватара — Coqui XTTS v2 (клонирование, ru)
Проверенный сервис: `http://195.209.214.155:9090` (переопределяется `XTTS_URL`).

**Загрузка вашего голоса в чат:** прикрепите аудиофайл (10–30 сек чистой речи, wav/mp3)
к сообщению — агент сохранит его в `render/voices/` и прогонит клонирование:
```bash
python3 scripts/xtts_client.py upload --src <ваш_файл> --name <имя>
python3 scripts/xtts_client.py clone  --text "Привет! Я твой цифровой аватар..." \
    --voice render/voices/<имя>.wav --out render/voice_xtts.wav
python3 scripts/xtts_client.py health   # проверка доступности сервиса
```
Результат — `render/voice_xtts.wav` (клонированный голос на тексте аватара),
далее он автоматически подхватывается матрицей липсинка.

### Сетевой регламент песочницы
Эгресс песочницы разрешает только PyPI / GitHub (git+API) / npm.
Прямые обращения к `195.209.214.155:9090`, Google Drive, HuggingFace, GitHub LFS
из песочницы **заблокированы** (проверено). Поэтому клонирование запускается:
- из песочницы — когда сервис доступен по разрешённому адресу, либо
- на вашей машине — той же командой `scripts/xtts_client.py clone ...`
  (результат просто прикрепите в чат или запушьте в `render/`).

## Матрица липсинк-движков (open source)
`python3 scripts/run_lipsync_matrix.py` → `render/matrix/*.mp4` + `STATUS.md`.
Движки: local-viseme-v2 (всегда), Wav2Lip PyTorch, Wav2Lip ONNX+GFPGAN.
Нейродвижки требуют весов; статус — в `render/matrix/STATUS.md`.

## Доставка весов в песочницу
`bash scripts/fetch_weights.sh`:
- s3fd и любые файлы <100 МБ — обычными blob через git;
- wav2lip.pth (435 МБ) — чанками: на вашей машине
  `split -b 90M wav2lip.pth wav2lip.pth.part_`, запушьте в любое ваше репо и
  `WEIGHTS_REPO=<url> bash scripts/fetch_weights.sh` — песочница соберёт файл.
