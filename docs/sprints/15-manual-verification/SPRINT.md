# Sprint 15: Weryfikacja ręczna (ground truth dla AU)

**Cel sprintu:** Zweryfikować ręcznie automatyczne pre-etykiety AU na czystym zbiorze
klatek, aby powstał ground truth do trenowania sieci AU (Sprint 16).

**Semestr:** 2
**Faza:** Quality Assurance

---

## Kontekst (stan faktyczny)

Batch annotation (Sprint 13/14) przetworzył **1491 wideo** → **2826 klatek peak** →
**549 klatek** w tierach `strict` + `good` (`data/dataset_final/annotations_clean.json`).
To jest realna pula do weryfikacji — wcześniejszy plan zakładał 6250 klatek z
25% próbki znacznie większego zbioru, który nie powstał.

Pre-etykiety **nie są** ground truth: wejście (46 keypoints) jest wiarygodne na
klatkach frontalnych, ale cel (21 AU i emocja) pochodzi z reguł geometrycznych
i wymaga potwierdzenia przez człowieka.

---

## Blokery narzędziowe (do zrobienia przed weryfikacją)

| # | Brak | Skutek |
|---|------|--------|
| 1 | Import gotowego COCO do webapp | Sesja powstaje tylko przez `POST /api/process_video` (upload wideo) — 549 gotowych klatek nie da się otworzyć w edytorze |
| 2 | Eksport zweryfikowanych etykiet do CSV | Sprint 16 (MLP 138 → 21 AU) nie ma czym karmić treningu |
| 3 | Tryb weryfikacji AU (toggle aktywne/nieaktywne) | Edytor AU operuje na wartościach ciągłych; etykiety do treningu są binarne |

---

## Stories

| ID | Tytuł | Status |
|----|-------|--------|
| [15.1](stories/15.1-verification-tool-setup.md) | Narzędzie weryfikacji (import COCO + tryb AU) | To Do |
| [15.2](stories/15.2-sample-selection.md) | Wybór próbki do weryfikacji | To Do |
| [15.3](stories/15.3-verification-execution.md) | Przeprowadzenie weryfikacji | To Do |

---

## Kryteria akceptacji

- Zweryfikowane klatki mają binarne etykiety 21 AU potwierdzone przez człowieka
- Zmierzona zgodność między anotatorami (inter-annotator agreement) na wspólnej próbce
- Eksport zweryfikowanego zbioru do CSV (138 keypoints → 21 AU)
- Udokumentowana instrukcja anotacji (co znaczy „AU aktywne")

---

## Zależności

- Sprint 13/14 (batch annotation) — **zrobione**, zbiór `data/dataset_final/`
- Format AU z wiarygodnością (`ratio` + `is_active` + `confidence`) — **zrobione**

---

## Uwaga o rozkładzie klas

W czystym zbiorze 549 klatek emocje rozkładają się skrajnie nierówno:
relaxed 260, neutral 242, sad 27, fearful 6, angry 5, happy 4, submission 3, surprise 2.
Weryfikacja AU ma sens niezależnie od tego (AU są bardziej podstawowe niż emocje),
ale zbiór **nie nadaje się** do trenowania klasyfikatora emocji — patrz analiza
w `docs/SESSION_HANDOFF.md` i `data/dataset_final/README.md`.
