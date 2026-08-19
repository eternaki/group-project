# Dog FACS Dataset — zbiór finalny

Złożono: 2026-08-17 10:53 UTC
Źródło: `data/dataset_final/curated.json` + `data/labels/dataset_final/*.jsonl`

## Zawartość

- Par zweryfikowanych przez człowieka: **5**
- Obrazów (kadry mordy): **8**
- Rozmiar obrazów: **0.3 MB**
- Par spornych (różni anotatorzy, różny werdykt): **0**

```
images/          kadry mordy, JPEG q90, dłuższy bok <= 512 px
annotations.json   COCO: 46 keypoints, 21 AU (reguły + werdykt człowieka)
au_labels.csv       tabela pod trening sieci AU (Sprint 16)
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

## Rozkłady

Emocje (klatki zweryfikowane): relaxed 3, neutral 2

AU oznaczone jako aktywne: AU143 2, EAD103 1
