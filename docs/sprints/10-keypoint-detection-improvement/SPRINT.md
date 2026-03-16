# Sprint 10: Keypoint Detection — Ulepszenie

**Sprint Goal:** Przeanalizować i ulepszyć model detekcji 46 punktów kluczowych twarzy psa (ResNet34/DogFLW) pod kątem jakości, architektury i adaptacji klatek.

**Duration:** Do ustalenia
**Semester:** 2
**Phase:** Ulepszenie modeli

---

## Overview

Ten sprint skupia się na analizie i ulepszeniu detekcji 46 punktów kluczowych DogFLW. Kluczowe pytania: czy ResNet34 jest wystarczający, jak normalizować punkty, jak obsługiwać różne orientacje głowy psa.

---

## Team Assignments

| Story | Assignee | Priority |
|-------|----------|----------|
| 10.1 Analiza architektury | TBD | High |
| 10.2 Analiza I/O i normalizacja | TBD | High |
| 10.3 Adaptacja klatek | TBD | High |
| 10.4 Metryki i ewaluacja | TBD | Medium |

---

## Stories

| ID | Title | Status |
|----|-------|--------|
| [10.1](stories/10.1-architecture-analysis.md) | Analiza architektury ResNet34/ViTPose | To Do |
| [10.2](stories/10.2-io-normalization.md) | Analiza I/O i normalizacja keypoints | To Do |
| [10.3](stories/10.3-frame-adaptation.md) | Adaptacja klatek (crop twarzy, orientacja) | To Do |
| [10.4](stories/10.4-metrics-evaluation.md) | Metryki i ewaluacja (OKS, PCK) | To Do |

---

## Success Criteria

- Udokumentowane 46 punktów DogFLW ze znaczeniem anatomicznym
- Zdefiniowana normalizacja koordynatów
- Pipeline adaptacji croppów twarzy psa
- Metryki: OKS, PCK@0.1 zmierzone

---

## Deliverables

- [ ] Dokument analizy architektury (ResNet34 vs ViTPose)
- [ ] Specyfikacja normalizacji 46 punktów
- [ ] Skrypt adaptacji croppów twarzy
- [ ] Raport metryk OKS/PCK

---

## Dependencies

- Sprint 8 (Dog Detection) — bbox do croppów
- Sprint 4 (Keypoint Detection) — bazowa implementacja

---

## Pytania do zbadania

- ResNet34 vs ViTPose — które jest lepsze dla psów?
- Jak normalizować punkty (względem bbox, względem głowy)?
- Jak obsługiwać psy z głową odwróconą lub niewidoczną?
- Czy 46 punktów DogFLW pokrywa wszystkie potrzebne AU?
