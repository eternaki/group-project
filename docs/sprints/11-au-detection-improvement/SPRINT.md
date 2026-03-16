# Sprint 11: AU Detection — Ulepszenie

**Sprint Goal:** Ulepszyć wyodrębnianie 21 Action Units (DogFACS) z delta keypoints — normalizacja, kalibracja, obsługa klatek neutralnych.

**Duration:** Do ustalenia
**Semester:** 2
**Phase:** Ulepszenie modeli

---

## Overview

DeltaActionUnitsExtractor oblicza 21 AU DogFACS na podstawie różnicy między keypoints klatki docelowej a klatki neutralnej. Ten sprint ulepsza normalizację, kalibrację względem rozmiaru głowy i robustność na brakujące punkty.

---

## Team Assignments

| Story | Assignee | Priority |
|-------|----------|----------|
| 11.1 Analiza obecnej implementacji AU | TBD | High |
| 11.2 Normalizacja i kalibracja | TBD | High |
| 11.3 Obsługa klatki neutralnej | TBD | Medium |
| 11.4 Walidacja AU | TBD | Medium |

---

## Stories

| ID | Title | Status |
|----|-------|--------|
| [11.1](stories/11.1-au-analysis.md) | Analiza obecnej implementacji 21 AU | To Do |
| [11.2](stories/11.2-normalization.md) | Normalizacja i kalibracja względem głowy | To Do |
| [11.3](stories/11.3-neutral-frame.md) | Ulepszenie wyboru klatki neutralnej | To Do |
| [11.4](stories/11.4-au-validation.md) | Walidacja AU względem DogFACS standard | To Do |

---

## Success Criteria

- Wszystkie 21 AU obliczane poprawnie (AU101, AU143, AU145, AU109, AU110, AU12, AU116, AU118, AU25, AU26, AU27, AD19, AD33, AD35, AD37, AD137, EAD101, EAD102, EAD103, EAD104, EAD105)
- Normalizacja uwzględnia rozmiar głowy
- Klatka neutralna wybierana automatycznie lub ręcznie
- Dokumentacja każdego AU

---

## Deliverables

- [ ] Ulepszona implementacja DeltaActionUnitsExtractor
- [ ] Specyfikacja normalizacji
- [ ] Ulepszona logika neutral_frame.py
- [ ] Testy jednostkowe dla wszystkich 21 AU

---

## Dependencies

- Sprint 10 (Keypoint Detection) — keypoints jako wejście
- Sprint 5 (Emotion Classification) — AU jako wejście do emocji

---

## Pytania do zbadania

- Jak normalizować delty względem rozmiaru głowy?
- Jak automatycznie wybrać najlepszą klatkę neutralną?
- Które AU są najtrudniejsze do wyodrębnienia i dlaczego?
