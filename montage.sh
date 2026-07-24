#!/usr/bin/env bash
set -e
FONT="$HOME/.fonts/Montserrat-ExtraBold.ttf"
[ -f "$FONT" ] || FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
printf 'ЧЕК-ЛИСТ: 30 ТЕМ ДЛЯ РИЛСОВ' > cta1.txt
printf '→ t.me/Tvoj_MASSHTAB' > cta2.txt

ffmpeg -y -loglevel error -i reels1_raw.mp4 \
  -i sfx/whoosh.mp3 -i sfx/bass.mp3 -i sfx/ticking.mp3 -i sfx/ding.mp3 -i sfx/final.mp3 \
  -filter_complex "\
[0:v]scale=1152:2048,zoompan=z='1+0.0014*in':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1080x1920,ass=subs.ass,\
drawbox=x=60:y='ih-560':w='iw-120':h=150:color=0x0c1220@0.88:t=fill:enable='between(t,38,43)',\
drawbox=x=60:y='ih-560':w=6:h=150:color=0xF5C451:t=fill:enable='between(t,38,43)',\
drawtext=fontfile=$FONT:textfile=cta1.txt:fontcolor=0xF5C451:fontsize=44:x=95:y='h-540':enable='between(t,38,43)',\
drawtext=fontfile=$FONT:textfile=cta2.txt:fontcolor=white:fontsize=40:x=95:y='h-478':enable='between(t,38,43)'[vout];\
[1:a]adelay=0|0,volume=0.9[s1];\
[2:a]adelay=1500|1500,volume=0.8[s2];\
[3:a]adelay=6000|6000,volume=0.16,afade=t=out:st=5:d=1[s3];\
[4:a]asplit=2[d1][d2];\
[d1]adelay=20500|20500,volume=0.7[s4a];\
[d2]adelay=25800|25800,volume=0.7[s4b];\
[5:a]adelay=38000|38000,volume=0.85[s5];\
[0:a][s1][s2][s3][s4a][s4b][s5]amix=inputs=7:duration=first:normalize=0[aout]" \
  -map "[vout]" -map "[aout]" -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p \
  -c:a aac -b:a 192k -movflags +faststart reels1_final_1080x1920.mp4

python3 - <<'EOF'
import json, datetime
json.dump({"stage": "done", "updated": datetime.datetime.utcnow().isoformat() + "Z"},
          open("status.json", "w"), ensure_ascii=False, indent=2)
EOF
echo ">> Готово: reels1_final_1080x1920.mp4"
