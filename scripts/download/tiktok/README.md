# Kolektor wideo z psami z TikToka

Zbiera wideo po hashtagach, odsiewa treść oznaczoną jako AI-generated,
filtruje po widoczności mordy psa (reużywa modeli bbox+keypoints projektu)
i wysyła zaakceptowane wideo na Google Drive.

## Instalacja

```bash
pip install -e ".[tiktok]"
python -m playwright install chromium
```

## Konfiguracja przed pierwszym uruchomieniem

1. **`secrets/gdrive_credentials.json`** — OAuth client (Desktop app) z Google
   Cloud Console. Już umieszczony w repo lokalnie (poza git, patrz `.gitignore`).
2. **Zmienna środowiskowa `TIKTOK_MS_TOKEN`** — wartość cookie `ms_token`
   z zalogowanej sesji TikTok w przeglądarce:
   - Zaloguj się na tiktok.com pod swoim kontem.
   - Otwórz DevTools (F12) → zakładka **Application/Storage → Cookies**
     → `https://www.tiktok.com` → skopiuj wartość cookie `ms_token`.
   - Ustaw: `set TIKTOK_MS_TOKEN=<wartość>` (PowerShell: `$env:TIKTOK_MS_TOKEN="<wartość>"`).
   - Token wygasa po pewnym czasie (zwykle dni) — jeśli skrypt zacznie dostawać
     puste wyniki, trzeba go odświeżyć tym samym sposobem.
3. Lista hashtagów i progi filtrów — `config.py` (`DEFAULT_HASHTAGS`,
   `AI_CONTENT_MARKERS`, `FACE_KEYPOINT_CONFIDENCE_THRESHOLD` itd.).

## Uruchomienie

```bash
python -m scripts.download.tiktok.collect
```

Przy pierwszym uruchomieniu otworzy się przeglądarka z prośbą o zalogowanie
do Google Drive (tylko raz — token zapisze się w `secrets/token.json`).

Skrypt można bezpiecznie przerwać (Ctrl+C) i uruchomić ponownie — postęp
(przetworzone ID wideo, licznik zaakceptowanych) jest zapisywany do
`data/tiktok_state.json` po każdym wideo.

## Uwagi

- Bez proxy tempo jest celowo ograniczone (`MIN/MAX_REQUEST_DELAY_SECONDS`
  w `config.py`), żeby zmniejszyć ryzyko tymczasowego rate-limitu na koncie/IP.
  Realistyczny czas zebrania 2000 wideo: kilka dni pracy w tle.
- Filtr AI-generated jest heurystyczny (słowa kluczowe w opisie/hashtagach)
  — nie gwarantuje 100% skuteczności, warto wyrywkowo zweryfikować ręcznie.
- Filtr widoczności mordy używa aktualnych wag `models/yolov8m.pt` i
  `models/keypoints_dogflw.pt` — jakość filtrowania zależy od jakości tych modeli.
