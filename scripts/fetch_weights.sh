#!/usr/bin/bash
# Доставка весов в песочницу через разрешённый эгресс (github.com, обычные blob <100MB).
# 1) s3fd (face detection, 89.8MB) — лежит обычным blob, тянется сам:
set -e
cd "${THIRDPARTY:-$HOME/third_party}"
if [ ! -s s3fd_repo/face_detection/detection/sfd/s3fd.pth ] || [ $(stat -c%s s3fd_repo/face_detection/detection/sfd/s3fd.pth 2>/dev/null || echo 0) -lt 1000000 ]; then
  rm -rf s3fd_repo
  git clone -q --filter=blob:none --no-checkout https://github.com/camenduru/video-dubbing-hf s3fd_repo
  cd s3fd_repo && git sparse-checkout set Wav2Lip/face_detection/detection/sfd && git checkout -q
  cd ..
  mkdir -p Wav2Lip/face_detection/detection/sfd
  cp -f s3fd_repo/Wav2Lip/face_detection/detection/sfd/s3fd.pth Wav2Lip/face_detection/detection/sfd/s3fd.pth
  echo "s3fd.pth OK: $(stat -c%s Wav2Lip/face_detection/detection/sfd/s3fd.pth) bytes"
fi
# 2) wav2lip.pth (435MB) — не влезает в обычный blob; ждём чанки <100MB из репо пользователя:
#    пользователь на своей машине: split -b 90M wav2lip.pth wav2lip.pth.part_ && git push в WEIGHTS_REPO
if [ -n "$WEIGHTS_REPO" ]; then
  rm -rf weights_chunks
  git clone -q "$WEIGHTS_REPO" weights_chunks
  if ls weights_chunks/wav2lip.pth.part_* >/dev/null 2>&1; then
    cat weights_chunks/wav2lip.pth.part_* > Wav2Lip/checkpoints/wav2lip.pth
    echo "wav2lip.pth собран: $(stat -c%s Wav2Lip/checkpoints/wav2lip.pth) bytes"
  fi
else
  echo "wav2lip.pth: нужен WEIGHTS_REPO с чанками (см. README «Доставка весов»)"
fi
