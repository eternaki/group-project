# Dog FACS Demo Application

Aplikacja demonstracyjna do analizy emocji psów.

## Wymagania

```bash
pip install streamlit
pip install -e .  # Instalacja pakietów projektu
```

## Uruchomienie

```bash
streamlit run apps/demo/app.py
```

Aplikacja otworzy się w przeglądarce pod adresem `http://localhost:8501`.

## Funkcjonalność

### Upload obrazów
- Obsługiwane formaty: JPG, JPEG, PNG
- Maksymalny rozmiar: 10MB
- Walidacja formatu i rozmiaru

### Analiza
Pipeline przetwarza obraz przez 4 modele:
1. **YOLOv8m** - Detekcja psów (bounding boxy)
2. **EfficientNet-B4** - Klasyfikacja rasy (120 ras)
3. **SimpleBaseline** - Detekcja keypoints (46 punktów)
4. **EfficientNet-B0** - Klasyfikacja emocji (4 klasy)

### Wizualizacja
- Bounding boxy z kolorowym kodowaniem
- Keypoints na twarzy psa
- Etykiety z rasą i emocją
- Porównanie oryginalnego i anotowanego obrazu

### Eksport
- **COCO JSON** - Pełne anotacje w formacie COCO
- **Obraz JPG** - Anotowany obraz z wizualizacją

## Ustawienia

W panelu bocznym dostępne są:
- **Próg pewności** - Minimalny confidence dla detekcji (0.1-0.9)
- **Wizualizacja** - Włącz/wyłącz bounding boxy, keypoints, etykiety

## Klasy emocji

| ID | Nazwa | Emoji |
|----|-------|-------|
| 0 | sad | 😢 |
| 1 | angry | 😠 |
| 2 | relaxed | 😌 |
| 3 | happy | 😊 |

## Struktura plików

```
apps/demo/
├── app.py          # Główna aplikacja Streamlit
├── __init__.py     # Moduł pakietu
└── README.md       # Ten plik
```

## Wymagane modele

Upewnij się, że w katalogu `models/` znajdują się:
- `yolov8m.pt` - Wagi YOLOv8
- `breed.pt` - Wagi klasyfikatora ras
- `keypoints_best.pt` - Wagi detektora keypoints
- `emotion.pt` - Wagi klasyfikatora emocji

## Screenshoty

Po uruchomieniu aplikacji:

1. Wgraj obraz z psem
2. Kliknij "Analizuj"
3. Zobacz wyniki i pobierz anotacje
