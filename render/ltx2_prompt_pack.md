# LTX-2 (ltx-2-19b) — говорящий аватар, RU, 5 сек

Готовые ассеты в этой папке (вне git, см. `.gitignore`):
- `avatar_closed.jpg` — стартовый кадр (img2video driving image), 9:16
- `voice.mp3` — русская реплика (≈4.1 c): «Привет! Я твой цифровой аватар. Пять секунд эфира — и мы уже знакомы.»
- `talking_avatar_5s.mp4` — локальный фолбэк-рендер (псевдо-липсинк по огибающей аудио), если LTX недоступен

## Рецепт (image + audio → video)

| Параметр | Значение |
|---|---|
| Модель | `ltx-2-19b` |
| Режим | image-to-video + audio conditioning (lip-sync) |
| Стартовое изображение | `avatar_closed.jpg` |
| Аудио | `voice.mp3` |
| Длительность | 5 s (аудио 4.1 s + улыбка в конце) |
| Разрешение | 720×1280 (9:16) |
| FPS | 24 |
| Steps / guidance | steps ≈ 30–40, cfg ≈ 3–4 (по дефолтам плейграунда) |
| Seed | зафиксировать для итераций |

## Motion prompt (EN, для ltx-2-19b)

```
A photorealistic young woman presenter in a green blazer speaks directly to camera
in Russian, natural conversational lip sync matching the provided audio track,
expressive mouth movements and subtle head nods, gentle blinks, soft smile at the
end. Static medium close-up shot, head and shoulders centered, very slow almost
imperceptible push-in, dark teal studio background, soft warm key light, sharp
focus on eyes, realistic skin texture.
```

## Негатив (опционально)

```
text, watermark, logo, distorted mouth, extra teeth, warped face, jitter,
camera shake, zoom artifacts, cropped head, changing background
```

## Проверка выхода
- Рот синхронен аудио на «Привет! … знакомы.»
- Последние ~0.9 c — закрытый рот, лёгкая улыбка
- Без артефактов на лице при 24 fps
