# DPP - Dokumentacja Procesu Projektowania

> Skopiuj zawartość do szablonu: `template/PG_WETI_DPP_wer. 1.00.doc`

---

## 1. Informacja o projekcie

### 1.1 Temat projektu

**Dog FACS Dataset** - Pipeline do automatycznej anotacji emocji psów z wykorzystaniem sztucznej inteligencji.

### 1.2 Cel projektu

Stworzenie datasetu w formacie COCO zawierającego:
- Bounding boxes psów na obrazach/klatkach wideo
- Klasyfikację ras psów
- Punkty kluczowe twarzy (facial keypoints)
- Etykiety emocji według systemu DogFACS

### 1.3 Zakres projektu

1. Opracowanie pipeline'u AI do automatycznej detekcji i anotacji
2. Zbieranie danych z filmów YouTube
3. Weryfikacja manualna anotacji
4. Eksport datasetu w formacie COCO
5. Aplikacja demonstracyjna (Streamlit)

### 1.4 Zespół projektowy

| Rola | Imię i nazwisko | Numer albumu |
|------|-----------------|--------------|
| TODO | TODO | TODO |
| TODO | TODO | TODO |
| TODO | TODO | TODO |
| TODO | TODO | TODO |

**Opiekun projektu**: TODO

---

## 2. Podział zadań i ról

### 2.1 Role w zespole

| Osoba | Odpowiedzialność |
|-------|------------------|
| TODO | Model detekcji psów (YOLOv8) |
| TODO | Model klasyfikacji ras |
| TODO | Model keypoints (HRNet) |
| TODO | Klasyfikator emocji, pipeline |

### 2.2 Harmonogram spotkań

- Spotkania zespołu: TODO (dzień tygodnia, godzina)
- Spotkania z opiekunem: TODO

---

## 3. Specyfikacja wymagań

### 3.1 Wymagania funkcjonalne

| ID | Wymaganie | Priorytet |
|----|-----------|-----------|
| WF01 | System wykrywa psy na obrazach/wideo | Wysoki |
| WF02 | System klasyfikuje rasę psa | Wysoki |
| WF03 | System wykrywa punkty kluczowe twarzy | Wysoki |
| WF04 | System klasyfikuje emocje psa | Wysoki |
| WF05 | System eksportuje dane w formacie COCO | Wysoki |
| WF06 | Aplikacja demo umożliwia upload obrazów | Średni |
| WF07 | Aplikacja demo umożliwia upload wideo | Średni |
| WF08 | System wspiera batch processing | Średni |

### 3.2 Wymagania niefunkcjonalne

| ID | Wymaganie | Wartość |
|----|-----------|---------|
| WN01 | Dokładność detekcji psów (mAP) | > 0.85 |
| WN02 | Dokładność klasyfikacji ras (Top-5) | > 0.90 |
| WN03 | Błąd keypoints (PCK@0.2) | < 0.15 |
| WN04 | Czas inference na GPU | < 500ms/obraz |
| WN05 | Format wyjściowy | COCO JSON |

---

## 4. Harmonogram prac

### 4.1 Sprinty

| Sprint | Nazwa | Czas | Status |
|--------|-------|------|--------|
| 1 | Project Setup | 2 tyg. | TODO |
| 2 | Dog Detection | 3 tyg. | TODO |
| 3 | Breed Classification | 3 tyg. | TODO |
| 4 | Keypoint Detection | 4 tyg. | TODO |
| 5 | Emotion Classification | 3 tyg. | TODO |
| 6 | Inference Pipeline | 2 tyg. | TODO |
| 7 | Demo Application | 2 tyg. | TODO |
| 8 | Data Collection | 2 tyg. | TODO |
| 9 | Batch Annotation | 2 tyg. | TODO |
| 10 | Manual Verification | 2 tyg. | TODO |
| 11 | Dataset Finalization | 1 tyg. | TODO |
| 12 | Statistics & Reporting | 1 tyg. | TODO |

### 4.2 Kamienie milowe

| Data | Kamień milowy |
|------|---------------|
| TODO | Działający model detekcji |
| TODO | Kompletny pipeline AI |
| TODO | Aplikacja demo |
| TODO | Gotowy dataset |
| TODO | Prezentacja końcowa |

---

## 5. Narzędzia i technologie

### 5.1 Stack technologiczny

| Kategoria | Technologia |
|-----------|-------------|
| Język | Python 3.11+ |
| Deep Learning | PyTorch, Ultralytics |
| Modele | YOLOv8, ViT/EfficientNet, HRNet |
| Demo | Streamlit |
| Format danych | COCO JSON |
| Wersjonowanie | Git, GitHub |
| CI/CD | GitHub Actions |
| Linter | Ruff |
| Type checking | MyPy |

### 5.2 Repozytorium

- **URL**: https://github.com/eternaki/group-project
- **Struktura gałęzi**: main → develop → sprint-X

---

## 6. Ryzyka projektu

| Ryzyko | Prawdopodobieństwo | Wpływ | Mitygacja |
|--------|-------------------|-------|-----------|
| Brak danych treningowych | Średnie | Wysoki | Użycie publicznie dostępnych datasetów |
| Niska dokładność modeli | Średnie | Wysoki | Transfer learning, fine-tuning |
| Opóźnienia w harmonogramie | Wysokie | Średni | Buffer czasowy, priorytyzacja |
| Problemy z GPU | Niskie | Wysoki | Google Colab jako backup |

---

*Dokument wygenerowany przez AI. Wymaga weryfikacji i uzupełnienia przez zespół.*
