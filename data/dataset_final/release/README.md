# Dog FACS Dataset — zbiór finalny

Złożono: 2026-09-07 20:31 UTC
Źródło: `data/dataset_final/curated.json` + `data/labels/dataset_final/*.jsonl`

## Zawartość

- Par zweryfikowanych przez człowieka: **529**
- Obrazów (kadry mordy): **825**
- Rozmiar obrazów: **23.9 MB**
- Par spornych (różni anotatorzy, różny werdykt): **11**

```
images/            kadry mordy, JPEG q90, dłuższy bok <= 512 px
annotations.json     COCO: 46 keypoints, 21 AU (reguły + werdykt człowieka + auto)
au_labels.csv         tabela pod trening sieci AU: werdykt CZŁOWIEKA (Sprint 16)
au_auto_labels.csv    werdykt AUTOMATYCZNY po szumowym gejcie dla całego zbioru 9k
```

## Czego tu nie ma

- **Pełnych klatek.** Obrazem jest kadr mordy — zbiór opisuje twarz, a tułów
  kosztowałby trzynaście razy więcej miejsca bez jednej dodatkowej etykiety.
  Powrót do oryginału: `source_video` + `frame_number` wskazują nagranie
  w `data/drive_dogs/`, a `source_bbox` położenie kadru w pełnej klatce.
- **Nagrań źródłowych.** COCO opisuje obrazy, nie wideo.

## Jak czytać etykiety AU

`au_verdicts` to ocena CZŁOWIEKA i tylko ona jest etykietą. Jest trójstanowa:
`active` / `inactive` / `not_observable` — ostatnie znaczy **brak wiedzy**
(np. ucho poza kadrem), a nie brak ruchu. W CSV odpowiada mu pusta komórka;
potraktowanie jej jako zera nauczyłoby sieć wymyślonych negatywów.

`au_analysis` to pomiar reguł geometrycznych — materiał porównawczy, NIE etykieta.
Zmierzony szum tych reguł przewyższa próg aktywacji na większości par trek–AU.

`au_auto_verdict` (w COCO) i `au_auto_labels.csv` (dla całego zbioru 9k) to
etykieta AUTOMATYCZNA po szumowym gejcie: aktywacja orzeczona tylko wtedy, gdy
reguła zapaliła AU I sygnał przewyższył zmierzony szum treku. Surowe `is_active`
jest bezużyteczne jako etykieta — 44% jego zapaleń na materiale 9k tonie w szumie.
Gejt je usuwa (znak `not_observable`), ale nie dorównuje weryfikacji człowieka:
to słaba, choć uczciwa etykieta reguł. Odtworzenie: `apply_noise_gate.py`.

## Rozkłady

Emocje (klatki zweryfikowane): neutral 200, relaxed 183, sad 55, happy 33, surprise 33, fearful 5, angry 5, submission 3, pain 1

AU oznaczone jako aktywne: AD19 36, AU25 35, AU27 24, EAD105 19, AU101 19, AU26 18, AU116 15, AU118 11, AU12 10, AU143 6, AD137 5, AD37 4, EAD104 4, AU145 3, EAD103 3, AU109 2, AU110 1

## Pełny zbiór 9k

Obok złotego podzbioru (`annotations.json` — kadry mordy zweryfikowane
przez człowieka) leży CAŁY materiał:

```
annotations_full.json   COCO 12238 klatek, 19038 anotacji
au_full_labels.csv      9519 pików: werdykt człowieka albo auto (label_source)
LICENSE                 CC BY-NC 4.0
```

- Klatek: **12238**, pików: **9519** (w tym **529** z werdyktem człowieka, reszta etykieta automatyczna).
- **Obrazem są PEŁNE klatki** z `data/dataset_final/work/frames/` (już w repo),
  a `file_name` wskazuje je względem tego katalogu. Punkty są w układzie pełnej
  klatki. Pomiar reguł (`au_analysis`) jest w `work/curated.json`.

### Skąd bierze się etykieta AU i ile jest warta

Kolumna `label_source` mówi, kto orzekł. Zmierzone na parach człowieka,
sprawdzianem krzyżowym z podziałem po nagraniach:

| źródło | pole w COCO | precyzja | pokrycie |
|--------|-------------|----------|----------|
| `human_verified` | `au_verdicts` | etykieta odniesienia | — |
| `auto_model` | `au_model_verdict` | 32.7% | 26.3% |
| `auto_rules` | `au_auto_verdict` | 5.0% | 40.5% |

- **Do treningu bierz `au_verdicts` tam, gdzie jest, a dalej `au_model_verdict`.**
  Reguły (`au_auto_verdict`) zostają w pliku wyłącznie dla porównania.
- **Żadna etykieta automatyczna nie jest prawdą.** Przy precyzji 33% dwie
  aktywacje na trzy są zmyślone.
- **Model odtwarza JEDEN standard oceniania.** Uczony jest na ocenach tych
  anotatorów, którzy w ogóle orzekają aktywacje: udziały w zespole rozjeżdżają
  się od 0.17% do 5.4% komórek, a mieszanie ich obniża precyzję z 32.7% na 26.8%.
- **Sufit jest nisko i to nie wina modelu.** Na parach ocenionych niezależnie
  przez dwie osoby zgoda na aktywacjach AU wynosi 7.4% (kappa 0.132), więc
  samo zjawisko jest słabo powtarzalne. Odtworzenie liczb:
  `python -m scripts.annotation.eval_au_rules` i `train_au_model`.
