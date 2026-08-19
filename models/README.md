# Model Weights

Wagi modeli AI dla pipeline Dog FACS. Stan na 2026-08-03.

## Aktywne modele

| Plik | Rozmiar | Architektura | Wynik |
|------|---------|--------------|-------|
| `yolov8m.pt` | 52 MB | YOLOv8m | Detekcja psów (bbox) |
| `breed.pt` | 71 MB | EfficientNet-B4 @380 | Klasyfikacja rasy, 120 klas, Top-1 **91.5%** |
| `dogface_yolo.pt` | 18 MB | YOLOv8n | Detektor **mordy** (kadrowanie przed keypoints), mAP50 **0.99** |
| `keypoints_dogflw.pt` | 271 MB | HRNet-W48 (wejście 320 → heatmapa 80) | 46 keypoints DogFLW, NME_iod **0.091**, PCK **0.748** |

Emocje **nie mają** modelu wagowego — są liczone regułami DogFACS z 21 AU
(`packages/models/emotion.py`), a AU geometrycznie z keypoints
(`packages/models/delta_action_units.py`).

## Wagi zapasowe (historia treningu keypoints)

`keypoints_hrnet256_nme139.pt`, `keypoints_hrnet320_nme126.pt`,
`keypoints_hrnet320_s15_nme118.pt`, `keypoints_hrnet48_ear_nme091.pt`
(= aktualny `keypoints_dogflw.pt`), `keypoints_dogflw_resnet_OLD.pt` (ResNet34, przed
naprawą błędu flip), `keypoints_best.pt` (SimpleBaseline, archiwum).

Sufiks `nmeXXX` = NME_iod ×1000. Postęp: 0.139 → 0.126 → 0.118 → **0.091**
(odniesienie z pracy ELD: 0.0652).

## Architektura pipeline

```
Obraz → BBox (YOLOv8m) → Crop psa
                          → Rasa (EfficientNet-B4)
                          → Detektor mordy (YOLOv8n) → Keypoints (HRNet-W48, 46 pkt)
                              → Delta AU vs klatka neutralna (21 AU DogFACS)
                              → Emocja (reguły DogFACS, 9 klas)
```

## Emocje (9 klas, reguły)

happy, sad, angry, fearful, relaxed, neutral, surprise, pain, submission
(Mota-Rojas et al. 2021). Definicje: `packages/data/schemas.py`.

## Git LFS

Pliki `.pt` są przechowywane przez Git Large File Storage.

```bash
git lfs install
git lfs pull
ls -la models/*.pt   # weryfikacja
```

## Historia

Wcześniejsza architektura emocji (CNN EfficientNet-B0 na pikselach, 4–6 klas oraz
MLP `KeypointsEmotionMLP` trenowany na keypoints) została **usunięta** — emocje są
pochodną AU, nie pikseli. Planowana sieć (Sprint 16) to MLP **138 keypoints → 21 AU**,
trenowany na zweryfikowanych ręcznie danych (Sprint 15).
