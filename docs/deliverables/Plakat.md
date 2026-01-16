# Plakat Projektu

> Skopiuj zawartość do szablonu: `template/PG_WETI_Plakat.doc`

---

## Nagłówek

**POLITECHNIKA GDAŃSKA**
Wydział Elektroniki, Telekomunikacji i Informatyki

**Katedra**: TODO

---

## Tytuł

# Dog FACS Dataset
### Pipeline do automatycznej anotacji emocji psów z wykorzystaniem AI

---

## Zespół

| | |
|---|---|
| TODO | TODO |
| TODO | TODO |

**Opiekun**: TODO

---

## Cel projektu

Stworzenie datasetu w formacie COCO zawierającego anotacje emocji psów wykrywanych automatycznie przez modele głębokiego uczenia.

---

## Architektura systemu

```
[Obraz/Wideo]
    ↓
[YOLOv8 - Detekcja psów]
    ↓
[Crop]
    ↓
┌──────────────────┬─────────────────┐
↓                  ↓                 ↓
[ViT - Rasa]  [HRNet - Keypoints]   │
                   ↓                 │
              [MLP - Emocje]         │
                   ↓                 │
└──────────────────┴─────────────────┘
    ↓
[COCO JSON]
```

---

## Kluczowe funkcje

- **Detekcja psów** - YOLOv8 (mAP > 0.85)
- **Klasyfikacja ras** - ViT (Top-5 > 90%)
- **Keypoints twarzy** - HRNet (20+ punktów)
- **Emocje DogFACS** - klasyfikator MLP

---

## Technologie

| Python | PyTorch | Streamlit |
|--------|---------|-----------|
| YOLOv8 | ViT | HRNet |

---

## Wyniki

| Metryka | Wartość |
|---------|---------|
| Obrazów w datasecie | TODO |
| Anotacji psów | TODO |
| Dokładność detekcji | TODO |
| Czas inference | TODO ms |

---

## Demo

[QR kod lub URL do demo]

---

## Kontakt

TODO@student.pg.edu.pl

---

*Projekt grupowy, semestr zimowy 2025/2026*
