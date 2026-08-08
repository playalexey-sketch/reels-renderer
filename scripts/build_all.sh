#!/usr/bin/env bash
# Полная сборка всех видео одним запуском (устойчиво к сбросам):
# среда -> веса -> рендер Марии -> конвейер улучшения (V1-V3, метрики, отчёт, пуш).
set -e
RR="$(cd "$(dirname "$0")/.." && pwd)"
bash "$RR/scripts/bootstrap_env.sh"
RR_BATCH=1 python3 "$RR/scripts/run_sadtalker.py" "$RR/render/voices/maria15.wav" "$RR/render/preview_maria.mp4"
python3 "$RR/scripts/overnight_pipeline.py"
echo "ВСЁ ГОТОВО: preview_maria_best.mp4 / preview_ru_best.mp4 / NIGHT_REPORT.md"
