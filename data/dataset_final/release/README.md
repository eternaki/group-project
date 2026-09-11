# Dog FACS Dataset — zbiór finalny

Złożono: 2026-09-07 10:12 UTC
Źródło: `data/dataset_final/curated.json` + `data/labels/dataset_final/*.jsonl`

## Zawartość

- Par zweryfikowanych przez człowieka: **528**
- Obrazów (kadry mordy): **824**
- Rozmiar obrazów: **20.1 MB**
- Par spornych (różni anotatorzy, różny werdykt): **11**

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

Emocje (klatki zweryfikowane): neutral 200, relaxed 182, sad 55, happy 33, surprise 33, fearful 5, angry 5, submission 3, pain 1

AU oznaczone jako aktywne: AD19 36, AU25 35, AU27 24, EAD105 19, AU101 19, AU26 18, AU116 15, AU118 11, AU12 10, AU143 6, AD137 5, AD37 4, EAD104 4, AU145 3, EAD103 3, AU109 2, AU110 1
