# Sprint 16: AU Neural Network

**Sprint Goal:** Zaimplementować i wytrenować sieć neuronową MLP do automatycznego wykrywania 21 AU DogFACS z keypoints (138 wejść → 21 AU).

**Duration:** Do ustalenia
**Semester:** 2
**Phase:** Ulepszenie modeli — AU NN

---

## Overview

Zamiast ręcznych reguł, sieć neuronowa MLP uczy się mapowania z 46 keypoints (138 koordynatów x,y + delta od klatki neutralnej) na 21 wartości AU. Trenowanie na ręcznie zanotowanym datasecie (Sprint 15).

**Architektura wejście:** 46 keypoints × 3 (x, y, delta) = 138 wartości
**Architektura wyjście:** 21 AU DogFACS (wartości ciągłe 0.0–5.0)

---

## Team Assignments

| Story | Assignee | Priority |
|-------|----------|----------|
| 16.1 Projekt architektury MLP | TBD | High |
| 16.2 Przygotowanie danych treningowych | TBD | High |
| 16.3 Trenowanie i walidacja | TBD | High |
| 16.4 Integracja z pipeline | TBD | Medium |

---

## Stories

| ID | Title | Status |
|----|-------|--------|
| [16.1](stories/16.1-mlp-architecture.md) | Projekt architektury MLP | To Do |
| [16.2](stories/16.2-training-data.md) | Przygotowanie danych treningowych | To Do |
| [16.3](stories/16.3-training-validation.md) | Trenowanie i walidacja modelu | To Do |
| [16.4](stories/16.4-pipeline-integration.md) | Integracja z pipeline inference | To Do |

---

## Success Criteria

- MLP wytrenowany z MAE < 0.5 na zbiorze testowym
- Model zintegrowany z InferencePipeline
- Porównanie z rule-based (Sprint 12) — co jest dokładniejsze?
- Plik modelu zapisany w `models/au_mlp.pt`

---

## Deliverables

- [ ] Implementacja MLP (`packages/models/au_mlp.py`)
- [ ] Skrypt trenowania (`scripts/training/train_au_mlp.py`)
- [ ] Wytrenowany model (`models/au_mlp.pt`)
- [ ] Raport porównania z rule-based
- [ ] Integracja w `packages/pipeline/inference.py`

---

## Dependencies

- Sprint 11 (AU Detection) — format danych AU
- Sprint 15 (Manual Verification) — ręcznie zanotowane dane treningowe
- Sprint 10 (Keypoint Detection) — keypoints jako wejście

---

## Architektura MLP (propozycja)

```
Wejście: 138 → Dense(256, ReLU) → Dropout(0.3)
             → Dense(128, ReLU) → Dropout(0.2)
             → Dense(64, ReLU)
             → Dense(21, Sigmoid×5)  # 0.0–5.0
```

## Pytania do zbadania

- Ile danych treningowych jest potrzebnych (minimum)?
- Regresja vs klasyfikacja binaryjna dla każdego AU?
- Jak traktować brakujące AU w annotacjach?
