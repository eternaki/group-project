# CLAUDE.md

Wytyczne dla Claude Code i innych agentów AI pracujących w tym repozytorium.

## Przegląd Projektu

**Dog FACS Dataset** — pipeline do automatycznej anotacji emocji psów (AI). Tworzy dataset w formacie COCO: bounding boxes, klasyfikacja ras, punkty kluczowe twarzy i etykiety emocji z filmów.

**Projekt grupowy** — Politechnika Gdańska (WETI), 1. semestr.

---

## Zasady obowiązkowe

### Język
- **Dokumentacja i komentarze**: po polsku
- **Nazwy zmiennych/funkcji**: po angielsku (standard)
- **Commit messages**: po polsku

### Styl kodu
- Type hints zawsze; format `ruff` (PEP 8); f-strings
- Funkcja robi jedną rzecz (≤50 linii, złożoność ≤10), max 5-6 parametrów
- Bez mutable default args, bez bare `except`, bez hardcodowanych ścieżek/magic numbers
- Docstringi publicznego API po polsku
- Zasady: SOLID, DRY, KISS, YAGNI

### Git Workflow

```
feature/<nazwa>  →  develop (integracja)  →  main (aktualna baza)
```

1. Nowa praca → gałąź `feature/<nazwa>` od `develop`
2. Gotowa funkcja → merge do `develop`
3. Stabilna integracja → merge `develop` do `main`
4. **Nie commituj bezpośrednio do `main`/`develop`** — używaj gałęzi `feature/*` i merge
5. `main` jest aktualną bazą odniesienia

### Commity
Format: `[PREFIX] Krótki opis po polsku`, gdzie PREFIX to `[SPRINT-X][STORY-Y.Z]`, `[SPRINT-X][TASK]` lub `[TASK]`.

Stopka co-author:
```
Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

Commity atomowe (jedna zmiana = jeden commit). Po ukończeniu zadania: testy (`pytest`), linter (`ruff check .`), aktualizacja statusu w `docs/sprints/` i Linear.

### GitHub / Linear
- Repo origin: `eternaki/group-project`. Aktywne konto sprawdź: `gh auth status`
- Linear: zespół **Dogs-ai (DOG)** — https://linear.app/team/DOG. Aktualny status zadań żyje w Linear, nie w tym pliku.

---

## Komendy Deweloperskie

```bash
# Instalacja (po zmianach w strukturze pakietów uruchom ponownie — patrz Gotchas)
pip install -e ".[dev,download,notebooks]"

uvicorn apps.webapp.backend.main:app --reload   # backend (FastAPI)
cd apps/webapp/frontend && npm run dev           # frontend (React + Vite)

pytest                                           # testy
ruff check . --fix                               # linter
mypy packages/                                    # typy
```

---

## Architektura

### Monorepo
- `apps/webapp/backend/` — FastAPI (SessionStore, REST API)
- `apps/webapp/frontend/` — React + TypeScript (Vite, Zustand, Tailwind)
- `packages/models/` — modele AI (bbox, breed, keypoints, AU, emocje)
- `packages/pipeline/` — zunifikowany pipeline inference (`inference.py`)
- `packages/data/` — odczyt/zapis COCO (`schemas.py`: 46 keypoints DogFLW + klasy emocji)
- `scripts/` — trening, pobieranie, batch annotation (`scripts/annotation/`, `scripts/debug/`)
- `notebooks/`, `docs/plans/`, `docs/sprints/`

### Pipeline AI
```
Obraz/Klatka → BBox (YOLOv8) → Crop → Rasa
                                     → Detektor mordy → Keypoints → AU → Emocje
→ COCO JSON
```

### Modele
| Model | Architektura | Cel | Wynik |
|-------|--------------|-----|-------|
| BBox | YOLOv8m | Detekcja psów | — |
| Rasa | EfficientNet-B4 @380 | Klasyfikacja rasy (120 klas) | Top-1 91.5% |
| Morda | YOLOv8n | Kadrowanie mordy przed keypoints | mAP50 0.99 |
| Keypoints | HRNet-W48 (320→80) | 46 punktów twarzy (DogFLW) | NME_iod 0.091, PCK 0.748 |
| AU | DeltaActionUnitsExtractor | 21 AU (DogFACS), geometria vs klatka neutralna | brak metryki (brak danych GT) |
| Emocje | Rule-based na AU | 9 emocji (DogFACS) | brak metryki (brak danych GT) |

### Katalogi danych
`data/raw/`, `data/frames/`, `data/annotations/` — w `.gitignore` (duże pliki lokalne). Tymczasowe podglądy w korzeniu `data/` też ignorowane.

---

## Gotchas (niełatwe do odgadnięcia)

- **Import shadowing**: zagnieżdżona kopia `Dog-Emotion-Classification/` może przechwytywać import `packages` w skryptach. Po zmianach w strukturze pakietów: `pip install -e .` ponownie.
- **pyproject**: `pythonpath = [".", "apps/webapp/backend"]` — testy importują backend.
- **Testy API**: `starlette.testclient.TestClient` niekompatybilny z httpx ≥0.28. Używaj `httpx.AsyncClient(transport=ASGITransport(app=app))` + `@pytest.mark.anyio`.
- **Środowisko uruchomienia**: `.venv` (Python 3.12), `node` dla yt-dlp, `ffmpeg` w PATH.
- **Keypoints**: pipeline robi square-crop przed Resize (fix zniekształceń) i kadruje mordę osobnym detektorem. Trudne pozostają profil i wiszące uszy.
- **AU w COCO**: pole `au_analysis` to `{ratio, is_active, confidence}` — samo `ratio` nie odróżnia realnej aktywacji od klamrowanego (niewiarygodnego) pomiaru. Odczyt starych zbiorów (samo `ratio` jako liczba) przez `packages.data.coco.au_ratio()`.

---

## Deliverables (1. semestr)
DPP (proces projektowania), Specyfikacja Oprogramowania (funkcje, interfejs, kod, wyniki), Raport Roczny (szablon WETI), Prezentacja przed komisją. Dokumenty po polsku w `docs/`.

---

## Sprinty

18 sprintów: narzędzia i modele → dane → weryfikacja → sieć neuronowa AU. Szczegóły i aktualny status w `docs/sprints/` oraz Linear.

| # | Sprint | Status |
|---|--------|--------|
| 1-7 | Setup, Detection, Breed, Keypoints, Emotion, Pipeline, Webapp | Done |
| 8-12 | Ulepszenia modeli (Detection/Breed/Keypoints/AU/Emotion) | Done (AU/emocje bez metryki — brak GT) |
| 13-14 | Data Collection, Batch Annotation | Done — 1491 wideo → 2826 peak → 549 czystych klatek |
| 15 | Manual Verification (webapp) | Blokada: brak importu COCO do sesji i eksportu CSV |
| 16 | AU Neural Network (MLP 138→21 AU) | Planowane |
| 17-18 | Dataset Finalization, Statistics & Reporting | Planowane |
