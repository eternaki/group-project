# Szablon Zadań Linear - Dog FACS Dataset

Ten plik zawiera listę zadań do utworzenia w Linear po podłączeniu MCP.

---

## Projekt: Dog FACS Dataset

**Opis**: Pipeline do automatycznej anotacji emocji psów z wykorzystaniem AI. Projekt grupowy dla Politechniki Gdańskiej (WETI).

---

## Epiki (Cycles/Projects w Linear)

### 1. Dokumentacja 1. Semestr
**Priorytet**: Wysoki
**Deadline**: Koniec semestru

Zadania:
- [ ] **DPP - Dokumentacja Procesu Projektowania**
  - Informacja o projekcie (temat, cel, zakres, zespół)
  - Podział zadań i ról
  - Specyfikacja wymagań
  - Harmonogram prac

- [ ] **Specyfikacja Oprogramowania**
  - Charakterystyka funkcjonalna
  - Opis interfejsu
  - Opis oprogramowania (kod, algorytmy, API)
  - Wyniki (metryki, przykłady działania)

- [ ] **Raport Roczny**
  - Wykonawcy
  - Główne zadania
  - Osiągnięte wyniki
  - Format według szablonu WETI

- [ ] **Prezentacja**
  - Przygotowanie slajdów
  - Demo aplikacji
  - Próby prezentacji

---

## Sprinty jako Milestones

### Sprint 1: Project Setup
**Czas**: 2 tygodnie

- [ ] 1.1 Konfiguracja repozytorium
- [ ] 1.2 Research DogFACS i COCO format
- [ ] 1.3 Analiza istniejących datasetów
- [ ] 1.4 Research architektury modeli
- [ ] 1.5 Finalizacja tech stacku

### Sprint 2: Dog Detection
**Czas**: 3 tygodnie

- [ ] 2.1 Przygotowanie danych treningowych
- [ ] 2.2 Fine-tuning YOLOv8
- [ ] 2.3 Ewaluacja modelu
- [ ] 2.4 Integracja z pipeline

### Sprint 3: Breed Classification
**Czas**: 3 tygodnie

- [ ] 3.1 Przygotowanie datasetu ras
- [ ] 3.2 Fine-tuning klasyfikatora
- [ ] 3.3 Ewaluacja modelu
- [ ] 3.4 Integracja z pipeline

### Sprint 4: Keypoint Detection
**Czas**: 4 tygodnie

- [ ] 4.1 Przygotowanie danych keypoints
- [ ] 4.2 Definicja schematu keypoints
- [ ] 4.3 Trening HRNet
- [ ] 4.4 Ewaluacja modelu
- [ ] 4.5 Integracja z pipeline

### Sprint 5: Emotion Classification
**Czas**: 3 tygodnie

- [ ] 5.1 Przygotowanie danych emocji
- [ ] 5.2 Mapowanie DogFACS
- [ ] 5.3 Trening klasyfikatora
- [ ] 5.4 Ewaluacja modelu
- [ ] 5.5 Integracja z pipeline

### Sprint 6: Inference Pipeline
**Czas**: 2 tygodnie

- [ ] 6.1 Architektura pipeline
- [ ] 6.2 Inference pojedynczej klatki
- [ ] 6.3 Przetwarzanie wideo
- [ ] 6.4 Eksport COCO

### Sprint 7: Demo Application
**Czas**: 2 tygodnie

- [ ] 7.1 Scaffold aplikacji Streamlit
- [ ] 7.2 Upload obrazów
- [ ] 7.3 Upload wideo
- [ ] 7.4 Wizualizacja wyników
- [ ] 7.5 Funkcja eksportu

### Sprint 8: Data Collection
**Czas**: 2 tygodnie

- [ ] 8.1 Strategia wyszukiwania filmów
- [ ] 8.2 Skrypt pobierania
- [ ] 8.3 Preprocessing filmów
- [ ] 8.4 Śledzenie kolekcji

### Sprint 9: Batch Annotation
**Czas**: 2 tygodnie

- [ ] 9.1 Skrypt batch processing
- [ ] 9.2 Optymalizacja GPU
- [ ] 9.3 Monitoring jakości

### Sprint 10: Manual Verification
**Czas**: 2 tygodnie

- [ ] 10.1 Setup narzędzia weryfikacji
- [ ] 10.2 Wybór próbek
- [ ] 10.3 Przeprowadzenie weryfikacji

### Sprint 11: Dataset Finalization
**Czas**: 1 tydzień

- [ ] 11.1 Merge anotacji
- [ ] 11.2 Walidacja COCO
- [ ] 11.3 Eksport datasetu

### Sprint 12: Statistics & Reporting
**Czas**: 1 tydzień

- [ ] 12.1 Notebook statystyk
- [ ] 12.2 Ocena jakości
- [ ] 12.3 Raport końcowy
- [ ] 12.4 Prezentacja

---

## Labels do użycia w Linear

- `documentation` - zadania dokumentacyjne
- `model` - zadania związane z modelami AI
- `pipeline` - zadania pipeline'u
- `data` - zadania związane z danymi
- `demo` - zadania aplikacji demo
- `research` - zadania badawcze
- `bug` - błędy
- `enhancement` - ulepszenia

## Priorytety

- `Urgent` - Blokuje innych
- `High` - Wymagane na ten sprint
- `Medium` - Do zrobienia
- `Low` - Nice to have

---

## Polecenia do utworzenia zadań (po podłączeniu Linear MCP)

```
# Przykład tworzenia zadania przez MCP:
linear_create_issue(
  title="[SPRINT-1][1.1] Konfiguracja repozytorium",
  description="Konfiguracja struktury projektu, CI/CD, linter, testy",
  priority=2,  # High
  labels=["documentation", "research"]
)
```
