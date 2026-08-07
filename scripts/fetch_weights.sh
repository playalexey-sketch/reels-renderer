#!/usr/bin/env bash
# Доставка весов в песочницу через разрешённый эгресс (github.com).
# - s3fd.pth (89.8MB) — сам, обычный blob из sahilg06/EmoGen.
# - wav2lip.pth (435MB) и GFPGANv1.4.pth (~64MB) — из чанков <=25MB,
#   загруженных пользователем в WEIGHTS_REPO (веб-аплоадом).
# - voice_*.* — аудио пользователя для клонирования голоса → render/voices/.
set -e
TP="${THIRDPARTY:-$HOME/third_party}"
RR="$(cd "$(dirname "$0")/.." && pwd)"
WEIGHTS_REPO="${WEIGHTS_REPO:-https://github.com/playalexey-sketch/wav2lip-weights.git}"
SHA_W2L=b78b681b68ad9fe6c6fb1debc6ff43ad05834a8af8a62ffc4167b7b34ef63c37
mkdir -p "$TP"
cd "$TP"

[ -d Wav2Lip ] || git clone -q https://github.com/Rudrabha/Wav2Lip.git Wav2Lip

# s3fd (реальный blob)
if [ ! -f Wav2Lip/face_detection/detection/sfd/s3fd.pth ] || [ $(stat -c%s Wav2Lip/face_detection/detection/sfd/s3fd.pth) -lt 1000000 ]; then
  rm -rf s3fd_repo
  git clone -q --filter=blob:none --no-checkout https://github.com/sahilg06/EmoGen s3fd_repo
  (cd s3fd_repo && git sparse-checkout set face_detection/detection/sfd && git checkout -q)
  mkdir -p Wav2Lip/face_detection/detection/sfd
  cp -f s3fd_repo/face_detection/detection/sfd/s3fd.pth Wav2Lip/face_detection/detection/sfd/s3fd.pth
  echo "s3fd.pth OK"
fi

w2l_ok=0
[ -f Wav2Lip/checkpoints/wav2lip.pth ] && [ "$(sha256sum Wav2Lip/checkpoints/wav2lip.pth | cut -d' ' -f1)" = "$SHA_W2L" ] && w2l_ok=1
g_ok=0
[ -f "$TP/GFPGANv1.4.pth" ] && [ $(stat -c%s "$TP/GFPGANv1.4.pth") -gt 1000000 ] && g_ok=1

if [ $w2l_ok -eq 0 ] || [ $g_ok -eq 0 ]; then
  rm -rf weights_chunks
  git clone -q "$WEIGHTS_REPO" weights_chunks || true
fi

if [ $w2l_ok -eq 0 ]; then
  if ls weights_chunks/wav2lip.pth.part_* >/dev/null 2>&1; then
    cat $(ls weights_chunks/wav2lip.pth.part_* | sort -V) > Wav2Lip/checkpoints/wav2lip.pth
    [ "$(sha256sum Wav2Lip/checkpoints/wav2lip.pth | cut -d' ' -f1)" = "$SHA_W2L" ] \
      && echo "wav2lip.pth собран и верифицирован" || { echo "sha256 wav2lip НЕ совпадает"; exit 2; }
  else
    echo "wav2lip.pth: чанки не найдены в $WEIGHTS_REPO"
  fi
fi

if [ $g_ok -eq 0 ]; then
  if ls weights_chunks/GFPGANv1.4.pth.part_* >/dev/null 2>&1; then
    cat $(ls weights_chunks/GFPGANv1.4.pth.part_* | sort -V) > "$TP/GFPGANv1.4.pth"
    echo "GFPGANv1.4.pth собран: $(stat -c%s "$TP/GFPGANv1.4.pth") bytes"
  else
    echo "GFPGANv1.4.pth: чанки не найдены"
  fi
fi

if ls weights_chunks/voice_*.* >/dev/null 2>&1; then
  mkdir -p "$RR/render/voices"
  cp -f weights_chunks/voice_*.* "$RR/render/voices/"
  echo "аудио пользователя скопировано в render/voices/"
fi
