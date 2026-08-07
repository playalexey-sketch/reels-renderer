#!/usr/bin/env bash
# SadTalker: код + веса. Мелочи тяну сам (реальные blob'ы), крупное — из чанков пользователя.
set -e
TP="${THIRDPARTY:-$HOME/third_party}"
RR="$(cd "$(dirname "$0")/.." && pwd)"
WEIGHTS_REPO="${WEIGHTS_REPO:-https://github.com/playalexey-sketch/wav2lip-weights.git}"
mkdir -p "$TP"
cd "$TP"

[ -d SadTalker ] || git clone -q https://github.com/OpenTalker/SadTalker.git SadTalker
cd SadTalker
mkdir -p checkpoints gfpgan/weights

# чанки пользователя
rm -rf "$TP/weights_chunks"
git clone -q "$WEIGHTS_REPO" "$TP/weights_chunks" || true
CH="$TP/weights_chunks"
mkdir -p "$TP/torchhome/hub/checkpoints"
assemble() { # prefix target
  if ls "$CH/$1.part_"* >/dev/null 2>&1; then
    cat $(ls "$CH/$1.part_"* | sort -V) > "$2"
    echo "собрано: $2 ($(stat -c%s "$2") bytes)"
  else
    echo "НЕТ чанков $1 для $2"
  fi
}
assemble sadt256 checkpoints/SadTalker_V0.0.2_256.safetensors
assemble bfmfit  checkpoints/BFM_Fitting.zip
assemble resnet50 "$TP/torchhome/hub/checkpoints/resnet50-19c8e357.pth"
if [ -f checkpoints/BFM_Fitting.zip ]; then
  (cd checkpoints && unzip -oq BFM_Fitting.zip)
fi

# shape_predictor_68 (реальный blob из numz/wav2lip_uhq)
if [ ! -f checkpoints/shape_predictor_68_face_landmarks.dat ]; then
  rm -rf "$TP/dlib_repo"
  git clone -q --filter=blob:none --no-checkout https://github.com/numz/wav2lip_uhq "$TP/dlib_repo"
  (cd "$TP/dlib_repo" && git sparse-checkout set predicator && git checkout -q)
  cp -f "$TP/dlib_repo/predicator/shape_predictor_68_face_landmarks.dat" checkpoints/
  echo "shape_predictor OK"
fi

# имеющиеся веса
[ -f "$TP/Wav2Lip/checkpoints/wav2lip.pth" ] && cp -f "$TP/Wav2Lip/checkpoints/wav2lip.pth" checkpoints/ || true
[ -f "$TP/GFPGANv1.4.pth" ] && cp -f "$TP/GFPGANv1.4.pth" gfpgan/weights/ || true
mkdir -p "$TP/torchhome/hub/checkpoints"
echo "SadTalker готов к запуску"
