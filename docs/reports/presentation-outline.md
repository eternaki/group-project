# Dog FACS Dataset - Prezentacja

## Szablon Prezentacji dla Komisji

**Czas:** 15-20 minut + Q&A
**Format:** Slajdy + Demo

---

## Slajd 1: Strona Tytułowa

```
DOG FACS DATASET
Automatyczna Anotacja Emocji Psów z Wykorzystaniem AI

Projekt Grupowy
Politechnika Gdańska, WETI
Semestr zimowy 2025/2026

Zespół: [Imiona]
```

---

## Slajd 2: Agenda

1. Wprowadzenie i Motywacja
2. Cele Projektu
3. Architektura Systemu
4. Demo (na żywo lub wideo)
5. Wyniki i Statystyki
6. Wnioski
7. Pytania

---

## Slajd 3: Problem i Motywacja

**Problem:**
- Brak publicznych datasetów z etykietami emocji psów
- Manualna anotacja jest czasochłonna i droga
- Potrzeba standaryzacji formatu danych

**Motywacja:**
- Zastosowania w weterynarii i opiece nad zwierzętami
- Badania naukowe nad zachowaniem zwierząt
- Aplikacje dla właścicieli psów

*[Grafika: Przykłady różnych emocji psów]*

---

## Slajd 4: Cele Projektu

| Cel | Status |
|-----|--------|
| 25,000 anotowanych klatek | ✅ |
| 6 kategorii emocji | ✅ |
| Format COCO | ✅ |
| 25% weryfikacja manualna | ✅ |
| Aplikacja demo | ✅ |

---

## Slajd 5: Architektura Systemu

```
┌─────────────────────────────────────────────────────┐
│                 INFERENCE PIPELINE                   │
│                                                      │
│  YouTube ──▶ Frames ──▶ [YOLOv8] ──▶ [Breed] ──▶   │
│                         [Keypoints] ──▶ [Emotion]   │
│                                                      │
│                         ▼                            │
│                    COCO JSON                         │
└─────────────────────────────────────────────────────┘
```

*[Diagram z ikonami dla każdego modelu]*

---

## Slajd 6: Modele AI

| Model | Zadanie | Architektura |
|-------|---------|--------------|
| Detekcja | Wykrywanie psów | YOLOv8-m |
| Rasa | 120 klas ras | EfficientNet-B4 |
| Keypoints | 46 punktów twarzy | SimpleBaseline |
| Emocje | 6 kategorii | EfficientNet-B0 |

*[Przykładowe wyniki dla każdego modelu]*

---

## Slajd 7: Kategorie Emocji

| Emocja | Cechy | Przykład |
|--------|-------|----------|
| 😊 Happy | Machanie ogonem, "uśmiech" | [foto] |
| 😢 Sad | Opuszczone uszy | [foto] |
| 😠 Angry | Warczenie, zęby | [foto] |
| 😌 Relaxed | Spokój, odpoczynek | [foto] |
| 😨 Fearful | Ogon między nogami | [foto] |
| 😐 Neutral | Brak cech | [foto] |

---

## Slajd 8: DEMO

**Opcja A: Demo na żywo**
- Uruchomienie aplikacji Streamlit
- Upload przykładowego obrazu
- Pokazanie wyników pipeline'u

**Opcja B: Wideo demo (2-3 min)**
- Nagranie działania systemu
- Komentarz do wyników

*[Przygotować backup - zrzuty ekranu]*

---

## Slajd 9: Format Danych COCO

```json
{
  "images": [...],
  "annotations": [
    {
      "id": 1,
      "bbox": [100, 150, 400, 300],
      "breed": {"name": "Labrador", "confidence": 0.92},
      "emotion": {"name": "happy", "confidence": 0.87},
      "keypoints": [x1, y1, v1, ...],
      "num_keypoints": 46
    }
  ]
}
```

---

## Slajd 10: Wyniki - Statystyki Datasetu

| Metryka | Wartość |
|---------|---------|
| Łącznie klatek | X,XXX |
| Łącznie anotacji | X,XXX |
| Unikalne rasy | XX |
| Śr. confidence | X.XX |

*[Wykres rozkładu emocji]*

---

## Slajd 11: Wyniki - Rozkład Emocji

*[Histogram lub pie chart z rozkładem 6 emocji]*

- Najczęstsza: [emocja] (XX%)
- Najrzadsza: [emocja] (XX%)

---

## Slajd 12: Wyniki - Jakość

| Metryka | Cel | Osiągnięty |
|---------|-----|------------|
| BBox IoU | > 85% | XX% |
| Emotion agreement | > 75% | XX% |
| Cohen's Kappa | > 0.6 | X.XX |

---

## Slajd 13: Harmonogram Projektu

```
Sprint 1-3:   Setup + Modele podstawowe
Sprint 4-6:   Keypoints + Emotion + Pipeline
Sprint 7-9:   Demo + Data Collection + Batch
Sprint 10-12: Verification + Finalization + Report
```

*[Timeline graficzny z milestones]*

---

## Slajd 14: Technologie

| Kategoria | Technologia |
|-----------|-------------|
| Język | Python 3.10+ |
| Deep Learning | PyTorch, timm |
| Detekcja | YOLOv8 |
| UI | Streamlit |
| Format | COCO JSON |
| VCS | Git, GitHub |

---

## Slajd 15: Wyzwania i Rozwiązania

| Wyzwanie | Rozwiązanie |
|----------|-------------|
| Jakość wideo | Preprocessing, filtrowanie |
| Subiektywność emocji | Weryfikacja manualna |
| Wydajność | Batch processing, GPU |
| Skalowalność | Modułowa architektura |

---

## Slajd 16: Przyszłe Kierunki

1. **Rozszerzenie DogFACS** - Pełne kodowanie mimiki
2. **Active Learning** - Iteracyjne ulepszanie modeli
3. **Aplikacja mobilna** - Rozpoznawanie w czasie rzeczywistym
4. **Inne gatunki** - Transfer learning do kotów, koni

---

## Slajd 17: Wnioski

✅ **Osiągnięcia:**
- Funkcjonalny pipeline AI
- Dataset w formacie COCO
- Narzędzia do weryfikacji
- Dokumentacja kompletna

📈 **Wartość projektu:**
- Podstawa do dalszych badań
- Potencjalne zastosowania komercyjne
- Doświadczenie w ML pipeline

---

## Slajd 18: Zespół

| Osoba | Rola | Wkład |
|-------|------|-------|
| U1 | Lead / ML | Architektura, pipeline |
| U2 | ML Engineer | Modele, trening |
| U3 | Data Engineer | Dane, batch |
| U4 | QA | Weryfikacja, testy |

---

## Slajd 19: Pytania?

```
DOG FACS DATASET

GitHub: github.com/eternaki/group-project
Demo: streamlit run apps/demo/app.py

Dziękujemy za uwagę!
```

---

## Materiały Pomocnicze

### Przygotowanie do Q&A

**Potencjalne pytania:**

1. *Dlaczego wybrano te konkretne modele?*
   - YOLOv8: SOTA w detekcji, szybkość
   - EfficientNet: Balans dokładność/rozmiar

2. *Jak walidowano etykiety emocji?*
   - Weryfikacja manualna 25%
   - Cohen's Kappa dla zgodności

3. *Jakie są ograniczenia systemu?*
   - Zależność od jakości źródła
   - Subiektywność emocji

4. *Jak można rozszerzyć projekt?*
   - Pełny DogFACS
   - Inne gatunki
   - Aplikacja mobilna

### Checklist przed prezentacją

- [ ] Sprawdzić demo działa
- [ ] Przygotować backup (zrzuty ekranu)
- [ ] Przetestować projektor
- [ ] Mieć offline kopię slajdów
- [ ] Przećwiczyć timing (15-20 min)

---

*Szablon prezentacji - do dostosowania przed prezentacją*
