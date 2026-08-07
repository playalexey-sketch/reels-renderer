#!/usr/bin/env bash
# Доставка весов Wav2Lip в песочницу через разрешённый эгресс (github.com).
# 1) s3fd.pth (89.8MB, обычный blob) — тянется сам из sahilg06/EmoGen.
# 2) wav2lip.pth (435MB) — собирается из чанков <100MB, запушенных пользователем
#    в WEIGHTS_REPO (по умолчанию playalexey-sketch/wav2lip-weights).
set -e
TP="${THIRDPARTY:-$HOME/third_party}"
WEIGHTS_REPO="${WEIGHTS_REPO:-https://github.com/playalexey-sketch/wav2lip-weights.git}"
SHA_W2L=b78b681b68ad9fe6c6fb1debc6ff43ad05834a8af8a62ffc4167b7b34ef63c37
mkdir -p "$TP"
cd "$TP"

# код Wav2Lip
[ -d Wav2Lip ] || git clone -q https://github.com/Rudrabha/Wav2Lip.git Wav2Lip

# s3fd (реальный blob)
if [ ! -f Wav2Lip/face_detection/detection/sfd/s3fd.pth ] || [ $(stat -c%s Wav2Lip/face_detection/detection/sfd/s3fd.pth) -lt 1000000 ]; then
  rm -rf s3fd_repo
  git clone -q --filter=blob:none --no-checkout https://github.com/sahilg06/EmoGen s3fd_repo
  (cd s3fd_repo && git sparse-checkout set face_detection/detection/sfd && git checkout -q)
  mkdir -p Wav2Lip/face_detection/detection/sfd
  cp -f s3fd_repo/face_detection/detection/sfd/s3fd.pth Wav2Lip/face_detection/detection/sfd/s3fd.pth
  echo "s3fd.pth OK: $(stat -c%s Wav2Lip/face_detection/detection/sfd/s3fd.pth) bytes"
fi

# wav2lip.pth из чанков
if [ -f Wav2Lip/checkpoints/wav2lip.pth ] && [ "$(sha256sum Wav2Lip/checkpoints/wav2lip.pth | cut -d' ' -f1)" = "$SHA_W2L" ]; then
  echo "wav2lip.pth уже собран и верифицирован"
else
  rm -rf weights_chunks
  if git clone -q "$WEIGHTS_REPO" weights_chunks 2>/dev/null && ls weights_chunks/wav2lip.pth.part_* >/dev/null 2>&1; then
    cat $(ls weights_chunks/wav2lip.pth.part_* | sort -V) > Wav2Lip/checkpoints/wav2lip.pth
    if [ "$(sha256sum Wav2Lip/checkpoints/wav2lip.pth | cut -d' ' -f1)" = "$SHA_W2L" ]; then
      echo "wav2lip.pth собран и верифицирован: $(stat -c%s Wav2Lip/checkpoints/wav2lip.pth) bytes"
    else
      echo "ОШИБКА: sha256 не совпадает — чанки повреждены"; exit 2
    fi
  else
    echo "wav2lip.pth: чанки не найдены в $WEIGHTS_REPO (см. README «Доставка весов»)"
  fi
fi
