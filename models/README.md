# Model Weights

Wagi modeli AI dla Dog FACS Demo.

## Modele

| Plik | Rozmiar | Opis |
|------|---------|------|
| `yolov8m.pt` | 52 MB | Model detekcji psów (YOLOv8m) |
| `breed.pt` | 71 MB | Model klasyfikacji ras (EfficientNet-B4, 120 ras) |
| `keypoints_best.pt` | 136 MB | Model punktów kluczowych (SimpleBaseline/ResNet50) |
| `emotion.pt` | 16 MB | Model klasyfikacji emocji (EfficientNet-B0, 4 klasy) |

## Git LFS

Pliki `.pt` są przechowywane za pomocą Git Large File Storage (LFS).

### Instalacja Git LFS

```bash
# macOS
brew install git-lfs

# Ubuntu/Debian
sudo apt install git-lfs

# Windows - pobierz z https://git-lfs.github.com
```

### Po klonowaniu repozytorium

Jeśli sklonowałeś repozytorium przed instalacją Git LFS:

```bash
git lfs install
git lfs pull
```

### Weryfikacja

```bash
# Sprawdź czy pliki są poprawnie pobrane (nie powinny być małymi plikami tekstowymi)
ls -la models/*.pt
```

Prawidłowe rozmiary:
- `yolov8m.pt` - ~52 MB
- `breed.pt` - ~71 MB
- `keypoints_best.pt` - ~136 MB
- `emotion.pt` - ~16 MB

Jeśli widzisz małe pliki (~130 bajtów), uruchom `git lfs pull`.
