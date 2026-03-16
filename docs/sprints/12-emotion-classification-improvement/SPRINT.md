# Sprint 12: Emotion Classification — Ulepszenie

**Sprint Goal:** Ulepszyć rule-based klasyfikator 9 emocji DogFACS — pokrycie AU, reguły, dokładność i obsługa edge cases.

**Duration:** Do ustalenia
**Semester:** 2
**Phase:** Ulepszenie modeli

---

## Overview

DogFACSRuleEngine klasyfikuje 9 emocji (happy, sad, angry, fearful, relaxed, neutral, surprise, pain, submission) na podstawie 21 AU. Ten sprint ulepsza reguły klasyfikacji, pokrycie przypadków i dodaje walidację względem literatury naukowej (Mota-Rojas et al. 2021).

---

## Team Assignments

| Story | Assignee | Priority |
|-------|----------|----------|
| 12.1 Analiza obecnych reguł | TBD | High |
| 12.2 Pokrycie przypadków brzegowych | TBD | High |
| 12.3 Walidacja naukowa | TBD | Medium |
| 12.4 Testy i metryki | TBD | Medium |

---

## Stories

| ID | Title | Status |
|----|-------|--------|
| [12.1](stories/12.1-rules-analysis.md) | Analiza obecnych reguł klasyfikacji | To Do |
| [12.2](stories/12.2-edge-cases.md) | Pokrycie przypadków brzegowych | To Do |
| [12.3](stories/12.3-scientific-validation.md) | Walidacja naukowa (Mota-Rojas 2021) | To Do |
| [12.4](stories/12.4-metrics.md) | Testy i metryki dokładności | To Do |

---

## Success Criteria

- Wszystkie 9 emocji ma zdefiniowane reguły AU
- Reguły zgodne z Mota-Rojas et al. 2021
- Obsługa przypadków wieloznacznych (np. kilka emocji jednocześnie)
- Dokumentacja każdej reguły z uzasadnieniem naukowym

---

## Deliverables

- [ ] Ulepszona implementacja DogFACSRuleEngine
- [ ] Dokument reguł AU → emocje z cytowaniami
- [ ] Testy jednostkowe dla wszystkich 9 emocji
- [ ] Raport pokrycia przypadków

---

## Dependencies

- Sprint 11 (AU Detection) — AU jako wejście
- Sprint 5 (Emotion Classification) — bazowa implementacja

---

## Pytania do zbadania

- Które emocje są najtrudniejsze do odróżnienia?
- Jak obsługiwać przypadki, gdy kilka emocji pasuje jednocześnie?
- Czy reguły są zgodne z najnowszą literaturą DogFACS?
