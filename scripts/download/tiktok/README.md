# Kolektor wideo z psami (TikTok / YouTube)

Zbiera wideo po hashtagach (TikTok) i/lub zapytaniach (YouTube), odsiewa
treść oznaczoną jako AI-generated, klasyfikuje emocję (keypoints -> neutral
frame -> delta AU -> reguły DogFACS) i wysyła zaakceptowane wideo na
wspólny Google Drive, posortowane do podfolderów per emocja.

## Instalacja

```bash
pip install -e ".[tiktok,download]"
python -m playwright install chromium   # tylko jeśli używasz źródła "tiktok"
```

## Konfiguracja przed pierwszym uruchomieniem

Wszystkie zmienne środowiskowe wpisz do pliku `.env` w korzeniu repo
(nie commituj go - jest w `.gitignore`).

### 1. Google Drive - `secrets/gdrive_credentials.json` (WSPÓLNY dla całego zespołu)

To identyfikator aplikacji (OAuth client), nie osobisty sekret - dostań go
od kogoś, kto już go ma, i wklej pod `secrets/gdrive_credentials.json`
(utwórz folder `secrets/`, jeśli go nie ma).

Przy pierwszym uruchomieniu otworzy się przeglądarka z prośbą o zalogowanie
do Google **pod TWOIM własnym kontem** - to tworzy TWÓJ osobisty
`secrets/token.json` (tego już NIE przesyłać dalej, jest twój).

### 2. YouTube Data API - `YOUTUBE_API_KEY` (OSOBNY dla każdej osoby)

Dzienna kwota API liczy się PER PROJEKT Google Cloud, nie per klucz - jeśli
kilka osób użyje tego samego klucza/projektu, kwota (100 wyszukiwań/dzień)
skończy się wielokrotnie szybciej. Każdy zakłada więc swój własny projekt:

1. `console.cloud.google.com/projectcreate` - nowy projekt (dowolna nazwa)
2. `console.cloud.google.com/apis/library/youtube.googleapis.com` - wybierz
   swój nowy projekt u góry -> **Enable**
3. `console.cloud.google.com/apis/credentials` -> **+ Create Credentials**
   -> **API key** -> skopiuj
4. W swoim `.env`:
   ```
   YOUTUBE_API_KEY=<twój_klucz>
   ```

Bez klucza kolektor działa dalej, ale wraca do scrapowanego `yt-dlp
ytsearch` (wolniejsze, podatne na tymczasowe blokady wyszukiwania).

### 3. TikTok - `TIKTOK_MS_TOKEN` (tylko jeśli używasz źródła "tiktok")

Wartość cookie `ms_token` z zalogowanej sesji TikTok w przeglądarce:
- Zaloguj się na tiktok.com pod swoim kontem.
- Otwórz DevTools (F12) → zakładka **Application/Storage → Cookies**
  → `https://www.tiktok.com` → skopiuj wartość cookie `ms_token`.
- W `.env`: `TIKTOK_MS_TOKEN=<wartość>`.
- Token wygasa po pewnym czasie (zwykle dni) — jeśli skrypt zacznie dostawać
  puste wyniki, trzeba go odświeżyć tym samym sposobem.
- TikTok pokazuje captchę do ręcznego rozwiązania w widocznym oknie
  przeglądarki - to źródło NIE jest w pełni bezobsługowe.

### 4. Listy zapytań i progi filtrów

`config.py` (`DEFAULT_HASHTAGS`, `DEFAULT_YOUTUBE_QUERIES`,
`AI_CONTENT_MARKERS`, progi `EMOTION_*` itd.).

## Uruchomienie

```bash
python -m scripts.download.tiktok.collect --source youtube   # tylko YouTube (bez captchy)
python -m scripts.download.tiktok.collect --source tiktok    # tylko TikTok
python -m scripts.download.tiktok.collect --source all       # oba na raz (domyślne)
```

Skrypt można bezpiecznie przerwać (Ctrl+C) i uruchomić ponownie — postęp
(przetworzone ID wideo, licznik zaakceptowanych per emocja) jest zapisywany
do `data/tiktok_state.json` po każdym wideo.

## Duplikaty treści

Przy starcie kolektor pobiera MD5 wszystkich plików już obecnych we
wspólnych podfolderach emocji na Dysku (Drive liczy je sam przy uploadzie)
i dolicza je do lokalnego stanu. Dzięki temu ta sama treść wgrana wcześniej
przez INNĄ osobę/maszynę nie zostanie wgrana ponownie — nie tylko w ramach
jednego uruchomienia na jednej maszynie.

## Uwagi

- Bez proxy tempo TikToka jest celowo ograniczone (`MIN/MAX_REQUEST_DELAY_SECONDS`
  w `config.py`), żeby zmniejszyć ryzyko tymczasowego rate-limitu na koncie/IP.
- Filtr AI-generated jest heurystyczny (słowa kluczowe w opisie/hashtagach)
  — nie gwarantuje 100% skuteczności, warto wyrywkowo zweryfikować ręcznie.
- Klasyfikacja emocji reużywa modeli projektu (`models/yolov8m.pt`,
  `models/keypoints_dogflw.pt`, `models/dogface_yolo.pt`) — jakość zależy
  od jakości tych wag. Klasyfikacja rasy jest celowo wyłączona (niepotrzebna
  tu, kosztowałaby dodatkowe inference na każdy peak frame).
