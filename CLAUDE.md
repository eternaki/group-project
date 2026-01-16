# CLAUDE.md

Ten plik zawiera wytyczne dla Claude Code (claude.ai/code) oraz innych agentów AI podczas pracy z kodem w tym repozytorium.

## Przegląd Projektu

**Dog FACS Dataset** - Pipeline do automatycznej anotacji emocji psów z wykorzystaniem AI. Tworzy dataset w formacie COCO z bounding boxes, klasyfikacją ras, punktami kluczowymi twarzy i etykietami emocji z filmów YouTube.

**Projekt grupowy** dla Politechniki Gdańskiej (WETI) - 1 semestr.

---

## ZASADY OBOWIĄZKOWE DLA WSZYSTKICH AGENTÓW

### Język

- **Dokumentacja**: Wszystkie dokumenty MUSZĄ być napisane po polsku
- **Komentarze w kodzie**: Po polsku
- **Nazwy zmiennych/funkcji**: Po angielsku (standard programistyczny)
- **Commit messages**: Po polsku z prefiksem typu zadania

### Czysty Kod

- Przestrzegaj zasad SOLID i DRY
- Każda funkcja robi jedną rzecz
- Nazwy zmiennych i funkcji muszą być opisowe
- Maksymalna długość funkcji: 50 linii
- Dokumentuj publiczne API (docstrings po polsku)

### Git Workflow

```
main (produkcja)
  ↑ merge po zakończeniu semestru
develop (integracja)
  ↑ merge po zakończeniu sprintu
sprint-X (np. sprint-1, sprint-2)
  ↑ commity podczas pracy
```

**Zasady:**
1. NIGDY nie pushuj bezpośrednio do `main` lub `develop`
2. Pracuj na gałęzi sprintu: `sprint-1`, `sprint-2`, itd.
3. Po zakończeniu sprintu → merge do `develop`
4. Po zakończeniu semestru → merge `develop` do `main`
5. Przed rozpoczęciem nowego sprintu → pull z `develop`

### Commity

**Format commit message:**
```
[SPRINT-X][STORY-Y.Z] Krótki opis zmiany

Szczegółowy opis (opcjonalnie)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

**Przykłady:**
- `[SPRINT-1][STORY-1.1] Konfiguracja repozytorium i struktury projektu`
- `[SPRINT-2][STORY-2.2] Fine-tuning modelu YOLOv8 do detekcji psów`
- `[SPRINT-1][TASK] Aktualizacja dokumentacji`

**Zasady commitów:**
1. Commit po każdej ukończonej story/task
2. Najpierw commity lokalne
3. Push do remote po zakończeniu logicznego bloku pracy
4. Atomowe commity - jedna zmiana = jeden commit

### Konto GitHub

- **Użytkownik eternaki**: Zawsze używaj `gh auth switch --user eternaki` przed operacjami git
- Sprawdź aktywne konto: `gh auth status`

### Struktura Gałęzi

```bash
# Tworzenie nowej gałęzi sprintu
git checkout develop
git pull origin develop
git checkout -b sprint-X

# Merge po zakończeniu sprintu
git checkout develop
git merge sprint-X
git push origin develop

# Finalna wersja na main
git checkout main
git merge develop
git push origin main
```

---

## Komendy Deweloperskie

```bash
# Instalacja zależności
pip install -e .
pip install -e ".[dev,download,notebooks]"

# Uruchomienie demo
streamlit run apps/demo/app.py

# Uruchomienie testów
pytest

# Linter
ruff check .
ruff check . --fix

# Sprawdzanie typów
mypy packages/
```

## Architektura

### Struktura Monorepo
- `apps/demo/` - Aplikacja demonstracyjna Streamlit
- `packages/models/` - Modele AI (YOLOv8 bbox, ViT breed, HRNet keypoints, klasyfikator emocji)
- `packages/pipeline/` - Zunifikowany pipeline inference orkiestrujący wszystkie modele
- `packages/data/` - Narzędzia do odczytu/zapisu formatu COCO
- `scripts/` - Skrypty do treningu, pobierania, batch annotation
- `notebooks/` - Statystyki i analiza datasetu
- `docs/plans/` - Dokumenty projektowe dla każdej fazy implementacji
- `docs/sprints/` - Definicje sprintów i stories

### Pipeline AI
```
Obraz/Klatka → Detekcja BBox → Crop → Klasyfikacja Rasy
                                    → Detekcja Keypoints → Klasyfikacja Emocji

Wyjście: COCO JSON ze wszystkimi anotacjami
```

### Kluczowe Modele
| Model | Architektura | Cel |
|-------|--------------|-----|
| BBox | YOLOv8 | Detekcja psów |
| Rasa | ViT/EfficientNet | Klasyfikacja rasy |
| Keypoints | HRNet | Punkty kluczowe twarzy (20+ punktów) |
| Emocje | Klasyfikator na keypoints | Etykiety emocji DogFACS |

---

## Deliverables 1. Semestr

### DPP (Dokumentacja Procesu Projektowania)
- Informacja o projekcie (temat, cel, zakres, zespół)
- Podział zadań i ról
- Specyfikacja wymagań
- Harmonogram prac

### Specyfikacja Oprogramowania
- Charakterystyka funkcjonalna
- Opis interfejsu
- Opis oprogramowania (kod, algorytmy, API)
- Wyniki (metryki, przykłady działania)

### Raport Roczny
- Wykonawcy
- Główne zadania
- Osiągnięte wyniki
- Format według szablonu WETI

### Prezentacja
- Demonstracja przed komisją
- Produkt + proces + dokumentacja

---

## Konwencja Dokumentacji

Po implementacji każdej funkcjonalności, utwórz/zaktualizuj dokument w `docs/plans/` według wzorca:
`YYYY-MM-DD-<nazwa-funkcjonalności>.md`

## Katalogi Danych

`data/` jest w .gitignore. Struktura:
- `data/raw/` - Pobrane filmy
- `data/frames/` - Wyekstrahowane klatki
- `data/annotations/` - Wyjścia COCO JSON

---

## Sprinty

Projekt podzielony na 12 sprintów:

| Sprint | Nazwa | Opis |
|--------|-------|------|
| 1 | Project Setup | Konfiguracja repo, research |
| 2 | Dog Detection | Model YOLOv8 |
| 3 | Breed Classification | Klasyfikacja ras |
| 4 | Keypoint Detection | HRNet keypoints |
| 5 | Emotion Classification | Klasyfikator emocji |
| 6 | Inference Pipeline | Zunifikowany pipeline |
| 7 | Demo Application | Aplikacja Streamlit |
| 8 | Data Collection | Zbieranie danych |
| 9 | Batch Annotation | Batch processing |
| 10 | Manual Verification | Weryfikacja manualna |
| 11 | Dataset Finalization | Finalizacja datasetu |
| 12 | Statistics & Reporting | Statystyki i raport |

Szczegóły każdego sprintu w `docs/sprints/`
