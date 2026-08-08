#!/usr/bin/env bash
# УСТОЙЧИВОЕ восстановление среды после сброса песочницы.
# Безопасно запускать повторно: каждый шаг идемпотентен.
# Рабочий код рендеров НЕ трогает — только окружение и веса.
set -e
PIP="pip install --user --break-system-packages --no-cache-dir -q"

python3 -c "import torch" 2>/dev/null || $PIP torch==2.5.1 torchvision==0.20.1
python3 -c "import kornia, gfpgan, librosa" 2>/dev/null || $PIP kornia==0.7.3 gfpgan facexlib basicsr librosa==0.10.2 safetensors yacs pydub scipy soundfile imageio-ffmpeg
python3 -c "import cv2" 2>/dev/null || { pip uninstall -q -y --break-system-packages opencv-python opencv-contrib-python 2>/dev/null || true; $PIP --force-reinstall opencv-python-headless; }

B=~/.local/lib/python3.11/site-packages/basicsr/data/degradations.py
[ -f "$B" ] && sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' "$B" || true

mkdir -p ~/.local/bin
ln -sf "$(python3 -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')" ~/.local/bin/ffmpeg 2>/dev/null || true

RR="$(cd "$(dirname "$0")/.." && pwd)"
bash "$RR/scripts/fetch_weights.sh"
bash "$RR/scripts/fetch_sadtalker.sh"
echo "BOOTSTRAP OK"
