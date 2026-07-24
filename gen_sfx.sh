#!/usr/bin/env bash
set -e
mkdir -p sfx

# whoosh на хуке
ffmpeg -y -loglevel error -f lavfi -i "anoisesrc=color=pink:duration=1.0:amplitude=0.7" \
  -af "asetrate=44100*(0.5+2.5*t),aresample=44100,highpass=f=250,lowpass=f=9000,afade=t=in:st=0:d=0.08,afade=t=out:st=0.5:d=0.5,volume=1.6" \
  sfx/whoosh.mp3

# бас-акцент
ffmpeg -y -loglevel error -f lavfi -i "sine=frequency=60:duration=1.5" \
  -af "volume='min(1\,exp(-2.5*t))':eval=frame,lowpass=f=180,volume=2.5" \
  sfx/bass.mp3

# тиканье под блоком проблемы (6 сек)
ffmpeg -y -loglevel error -f lavfi -i "aevalsrc='sin(2*PI*1400*t)*lt(mod(t\,0.5)\,0.04)':d=6:s=44100" \
  -af "volume=0.5" sfx/ticking.mp3

# ding на шагах демо
ffmpeg -y -loglevel error -f lavfi -i "aevalsrc='(sin(2*PI*880*t)+0.6*sin(2*PI*1760*t))*exp(-6*t)':d=1:s=44100" \
  -af "volume=0.8" sfx/ding.mp3

# финальный акцент под CTA
ffmpeg -y -loglevel error -f lavfi -i "aevalsrc='(sin(2*PI*523*t)+0.8*sin(2*PI*659*t)+0.6*sin(2*PI*784*t))*exp(-2*t)':d=2.5:s=44100" \
  -af "volume=0.7" sfx/final.mp3

ls -la sfx/
