# Sprint 9: Breed Classification — Ulepszenie

**Sprint Goal:** Przeanalizować i ulepszyć model klasyfikacji ras psów (EfficientNet-B4) pod kątem jakości, architektury i adaptacji klatek.

**Duration:** Do ustalenia
**Semester:** 2
**Phase:** Ulepszenie modeli

---

## Overview

Ten sprint skupia się na analizie i ulepszeniu modelu klasyfikacji ras. Obejmuje ocenę architektury EfficientNet-B4, analizę 120 klas ras, preprocessing croppów psów i pomiar metryk.

---

## Team Assignments

| Story | Assignee | Priority |
|-------|----------|----------|
| 9.1 Analiza architektury | TBD | High |
| 9.2 Analiza I/O i preprocessing | TBD | High |
| 9.3 Adaptacja croppów | TBD | Medium |
| 9.4 Metryki i ewaluacja | TBD | Medium |

---

## Stories

| ID | Title | Status |
|----|-------|--------|
| [9.1](stories/9.1-architecture-analysis.md) | Analiza architektury EfficientNet-B4 | To Do |
| [9.2](stories/9.2-io-analysis.md) | Analiza wejść/wyjść i preprocessing | To Do |
| [9.3](stories/9.3-crop-adaptation.md) | Adaptacja croppów psów | To Do |
| [9.4](stories/9.4-metrics-evaluation.md) | Metryki i ewaluacja modelu | To Do |

---

## Success Criteria

- Udokumentowana architektura i uzasadnienie
- Zdefiniowany format croppów (rozmiar, normalizacja)
- Poprawiona dokładność klasyfikacji ras
- Metryki: top-1 accuracy, top-5 accuracy zmierzone

---

## Deliverables

- [ ] Dokument analizy architektury
- [ ] Specyfikacja I/O modelu
- [ ] Skrypt adaptacji croppów
- [ ] Raport metryk ewaluacji

---

## Dependencies

- Sprint 8 (Dog Detection Ulepszenie) — bbox jako wejście do croppów
- Sprint 3 (Breed Classification) — bazowa implementacja

---

## Pytania do zbadania

- Czy 120 klas ras jest wystarczające dla naszego datasetu?
- Jak najlepiej przycinać i normalizować croppy psów?
- Czy EfficientNet-B4 jest optymalny, czy rozważyć inne architektury?
