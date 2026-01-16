# Prezentacja Końcowa

> Struktura slajdów do PowerPoint

---

## Slajd 1: Strona tytułowa

**POLITECHNIKA GDAŃSKA**
Wydział Elektroniki, Telekomunikacji i Informatyki

# Dog FACS Dataset
### Pipeline do automatycznej anotacji emocji psów

Zespół: TODO, TODO, TODO, TODO
Opiekun: TODO

Semestr zimowy 2025/2026

---

## Slajd 2: Agenda

1. Cel projektu
2. Architektura systemu
3. Modele AI
4. Demo aplikacji
5. Wyniki i metryki
6. Wnioski

---

## Slajd 3: Cel projektu

### Problem
- Brak dużych datasetów z anotacjami emocji psów
- Manualna anotacja jest czasochłonna i kosztowna

### Rozwiązanie
- Automatyczny pipeline AI do anotacji
- Format COCO dla kompatybilności
- Weryfikacja manualna dla jakości

### Rezultat
- Dataset z TODO obrazami
- 4 wytrenowane modele AI
- Aplikacja demo

---

## Slajd 4: Architektura systemu

```
[Diagram pipeline]

Obraz/Wideo → YOLOv8 → Crop → ViT (rasa)
                              → HRNet (keypoints) → MLP (emocje)

                              → COCO JSON
```

---

## Slajd 5: Model detekcji (YOLOv8)

- **Architektura**: YOLOv8-m
- **Trening**: Fine-tuning na TODO obrazach
- **Metryka**: mAP@0.5 = TODO

[Przykład detekcji]

---

## Slajd 6: Klasyfikacja ras (ViT)

- **Architektura**: Vision Transformer
- **Klasy**: TODO ras psów
- **Metryka**: Top-5 Accuracy = TODO

[Przykład klasyfikacji]

---

## Slajd 7: Keypoints (HRNet)

- **Architektura**: HRNet-W48
- **Punkty**: 20+ keypoints twarzy
- **Metryka**: PCK@0.2 = TODO

[Wizualizacja keypoints]

---

## Slajd 8: Emocje (DogFACS)

- **System**: Dog Facial Action Coding System
- **Klasy**: TODO emocji
- **Metryka**: F1-score = TODO

[Przykłady emocji]

---

## Slajd 9: Demo

### LIVE DEMO

[Pokazać aplikację Streamlit]

1. Upload obrazu
2. Przetwarzanie
3. Wyniki
4. Eksport COCO

---

## Slajd 10: Wyniki datasetu

| Statystyka | Wartość |
|------------|---------|
| Obrazów | TODO |
| Anotacji | TODO |
| Ras | TODO |
| Keypoints/pies | TODO |

[Wykresy statystyk]

---

## Slajd 11: Wnioski

### Osiągnięcia
- ✅ Działający pipeline
- ✅ 4 wytrenowane modele
- ✅ Aplikacja demo
- ✅ Dataset COCO

### Wyzwania
- TODO

### Przyszłość
- TODO

---

## Slajd 12: Pytania

# Dziękujemy za uwagę!

**Pytania?**

---

Repozytorium: https://github.com/eternaki/group-project

Kontakt: TODO@student.pg.edu.pl
