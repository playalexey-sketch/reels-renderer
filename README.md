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
- s3fd (89.8 МБ) — тянется сам (обычный blob в sahilg06/EmoGen);
- wav2lip.pth (435 МБ) — чанками от пользователя (эгресс песочницы режет LFS/HF/Drive):
  ```bash
  # на вашей машине (сеть открыта): скачать wav2lip.pth с официального Drive
  # (папка https://drive.google.com/drive/folders/153HLrqlBNxzZcHi17PEvP09kkAfzRshM),
  # затем:
  gh repo create wav2lip-weights --private   # или любой ваш репо
  git clone https://github.com/<вы>/wav2lip-weights && cd wav2lip-weights
  split -b 90M -d wav2lip.pth wav2lip.pth.part_
  git add -A && git commit -m chunks && git push
  # в чате написать «старт» (+ URL репо, если не playalexey-sketch/wav2lip-weights)
  ```
  Песочница соберёт файл, проверит sha256
  `b78b681b…63c37` и прогонит матрицу (Wav2Lip PyTorch).
