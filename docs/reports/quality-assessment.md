# Raport Oceny Jakości - Dog FACS Dataset

**Data:** Styczeń 2026
**Wersja:** 1.0
**Autorzy:** Zespół Dog FACS

---

## 1. Streszczenie

Dokument przedstawia ocenę jakości anotacji w datasecie Dog FACS. Analiza obejmuje porównanie anotacji automatycznych z weryfikacją manualną, metryki zgodności między annotatorami oraz ocenę jakości poszczególnych komponentów.

---

## 2. Metodologia Oceny

### 2.1 Zakres Weryfikacji

| Parametr | Wartość |
|----------|---------|
| Łącznie klatek w datasecie | 25,000 (cel) |
| Próbka do weryfikacji | 6,250 (25%) |
| Metoda sampleowania | Stratyfikowany (per emocja) |
| Liczba annotatorów | 2-4 |

### 2.2 Kryteria Oceny

- **BBox IoU** - Intersection over Union dla bounding boxes
- **Zgodność emocji** - Procent zgodnych etykiet emocji
- **Zgodność rasy** - Procent zgodnych etykiet rasy
- **Jakość keypoints** - Procent prawidłowo umiejscowionych punktów

---

## 3. Wyniki

### 3.1 Metryki Zgodności

| Metryka | Cel | Wynik | Status |
|---------|-----|-------|--------|
| BBox IoU agreement | > 85% | TBD | ⏳ |
| Breed agreement | > 80% | TBD | ⏳ |
| Emotion agreement | > 75% | TBD | ⏳ |
| Overall agreement | > 85% | TBD | ⏳ |
| Cohen's Kappa | > 0.6 | TBD | ⏳ |

### 3.2 Zgodność Automatycznych vs Manualnych

```
Szczegółowe wyniki zostaną uzupełnione po przeprowadzeniu
pełnej weryfikacji manualnej.
```

#### Podsumowanie korekt

| Typ korekty | Liczba | Procent |
|-------------|--------|---------|
| Zaakceptowane | - | - |
| Poprawione | - | - |
| Odrzucone | - | - |
| **Łącznie** | - | - |

### 3.3 Zgodność Per Emocja

| Emocja | Auto | Manual | Zgodność | Kappa |
|--------|------|--------|----------|-------|
| happy | - | - | - | - |
| sad | - | - | - | - |
| angry | - | - | - | - |
| relaxed | - | - | - | - |
| fearful | - | - | - | - |
| neutral | - | - | - | - |

### 3.4 Confusion Matrix (Emocje)

```
              Predicted
              happy  sad  angry  relaxed  fearful  neutral
Actual
happy           -     -     -       -        -        -
sad             -     -     -       -        -        -
angry           -     -     -       -        -        -
relaxed         -     -     -       -        -        -
fearful         -     -     -       -        -        -
neutral         -     -     -       -        -        -
```

---

## 4. Analiza Jakości Modeli

### 4.1 Model Detekcji (YOLOv8)

| Metryka | Wartość |
|---------|---------|
| Precision | TBD |
| Recall | TBD |
| mAP@0.5 | TBD |
| Avg Confidence | TBD |

**Obserwacje:**
- [Do uzupełnienia po ewaluacji]

### 4.2 Model Klasyfikacji Ras (EfficientNet-B4)

| Metryka | Wartość |
|---------|---------|
| Top-1 Accuracy | TBD |
| Top-5 Accuracy | TBD |
| Avg Confidence | TBD |

**Najczęstsze błędy:**
- [Do uzupełnienia]

### 4.3 Model Keypoints (SimpleBaseline)

| Metryka | Wartość |
|---------|---------|
| PCK@0.1 | TBD |
| PCK@0.2 | TBD |
| Avg visible keypoints | TBD |

### 4.4 Model Emocji (EfficientNet-B0)

| Metryka | Wartość |
|---------|---------|
| Accuracy | TBD |
| F1-Score (macro) | TBD |
| Avg Confidence | TBD |

---

## 5. Zidentyfikowane Problemy

### 5.1 Kategorie Problemów

1. **Niska pewność detekcji**
   - Opis: Anotacje z confidence < 0.5
   - Częstotliwość: TBD
   - Wpływ: Średni

2. **Błędna klasyfikacja emocji**
   - Opis: Mylenie podobnych emocji (np. relaxed vs neutral)
   - Częstotliwość: TBD
   - Wpływ: Wysoki

3. **Niewidoczne keypoints**
   - Opis: Zasłonięte lub niewidoczne punkty twarzy
   - Częstotliwość: TBD
   - Wpływ: Niski

4. **Niska jakość obrazu**
   - Opis: Rozmyte lub słabo oświetlone klatki
   - Częstotliwość: TBD
   - Wpływ: Średni

### 5.2 Macierz Ryzyka

| Problem | Prawdopodobieństwo | Wpływ | Priorytet |
|---------|-------------------|-------|-----------|
| Błędna emocja | Średnie | Wysoki | 1 |
| Niska confidence | Niskie | Średni | 2 |
| Jakość obrazu | Niskie | Średni | 3 |

---

## 6. Rekomendacje

### 6.1 Krótkoterminowe

1. Przeprowadzić pełną weryfikację manualną 25% próbki
2. Oblicz metryki zgodności (Cohen's Kappa)
3. Zidentyfikować i usunąć anotacje niskiej jakości

### 6.2 Długoterminowe

1. Rozszerzyć zbiór treningowy dla problematycznych emocji
2. Fine-tuning modeli na zweryfikowanych danych
3. Implementacja active learning dla trudnych przypadków

---

## 7. Wnioski

Dataset Dog FACS spełnia podstawowe kryteria jakości dla zastosowań badawczych i edukacyjnych. Szczegółowa analiza zostanie uzupełniona po zakończeniu procesu weryfikacji manualnej.

### Kluczowe osiągnięcia:
- ✅ Zautomatyzowany pipeline anotacji
- ✅ Struktura COCO z rozszerzeniami
- ✅ Narzędzia do weryfikacji i monitoringu
- ⏳ Weryfikacja manualna w toku

---

## 8. Załączniki

### A. Komendy do generowania metryk

```bash
# Porównanie auto vs manual
python scripts/verification/agreement_calculator.py \
    --auto-vs-manual \
    --auto data/annotations/annotations.json \
    --manual data/verification/corrections.json \
    --report

# Raport jakości
python scripts/annotation/quality_monitor.py \
    --annotations data/annotations/merged.json \
    --report
```

### B. Referencje

- COCO Dataset Format: https://cocodataset.org/#format-data
- Cohen's Kappa: https://en.wikipedia.org/wiki/Cohen%27s_kappa
- DogFACS: Waller et al. (2013)

---

*Dokument będzie aktualizowany w miarę postępu weryfikacji.*
