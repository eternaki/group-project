# Sprint 8: Dog Detection — Ulepszenie

**Sprint Goal:** Przeanalizować i ulepszyć model detekcji psów (YOLOv8m) pod kątem jakości, architektury i adaptacji klatek.

**Duration:** Do ustalenia
**Semester:** 2
**Phase:** Ulepszenie modeli

---

## Overview

Ten sprint skupia się na dogłębnej analizie i ulepszeniu modelu detekcji psów. Każdy sprint ulepszenia modelu obejmuje: ocenę aktualnej architektury, analizę danych wejściowych/wyjściowych, adaptację klatek wideo do wymagań modelu oraz pomiar metryk.

---

## Team Assignments

| Story | Assignee | Priority |
|-------|----------|----------|
| 8.1 Analiza architektury | TBD | High |
| 8.2 Analiza I/O i preprocessing | TBD | High |
| 8.3 Adaptacja klatek | TBD | Medium |
| 8.4 Metryki i ewaluacja | TBD | Medium |

---

## Stories

| ID | Title | Status |
|----|-------|--------|
| [8.1](stories/8.1-architecture-analysis.md) | Analiza architektury YOLOv8 | To Do |
| [8.2](stories/8.2-io-analysis.md) | Analiza wejść/wyjść i preprocessing | To Do |
| [8.3](stories/8.3-frame-adaptation.md) | Adaptacja klatek wideo | To Do |
| [8.4](stories/8.4-metrics-evaluation.md) | Metryki i ewaluacja modelu | To Do |

---

## Success Criteria

- Udokumentowana architektura i uzasadnienie wyboru
- Zdefiniowany format wejść (rozmiar, normalizacja, format)
- Zdefiniowany format wyjść (bbox format, confidence threshold)
- Pipeline adaptacji klatek wdrożony
- Metryki: mAP@50, precision, recall zmierzone

---

## Deliverables

- [ ] Dokument analizy architektury
- [ ] Specyfikacja I/O modelu
- [ ] Skrypt adaptacji klatek
- [ ] Raport metryk ewaluacji

---

## Dependencies

- Sprint 7 (Annotation Webapp) — ukończony
- Sprint 2 (Dog Detection) — bazowa implementacja

---

## Pytania do zbadania

- Czy YOLOv8m jest optymalny czy rozważyć YOLOv8l/x?
- Jakie preprocessing klatek jest potrzebny (resize, normalizacja, augmentacja)?
- Jak obsługiwać wiele psów na jednej klatce?
- Jaki próg pewności stosować?
