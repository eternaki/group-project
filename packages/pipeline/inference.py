"""
Zunifikowany pipeline inference dla projektu Dog FACS.

Pipeline przetwarza obrazy przez wszystkie 4 modele:
1. BBox detection (YOLOv8) - wykrywa psy na obrazie
2. Breed classification (EfficientNet-B4) - klasyfikuje rasę
3. Keypoints detection (SimpleBaseline) - wykrywa punkty kluczowe
4. Emotion classification (EfficientNet-B0) - klasyfikuje emocję

Przetwarzanie wideo (`process_video_for_dataset`) idzie po TREKACH psów, nie po
klatkach: każdy pies dostaje własną klatkę neutralną, własne wygładzanie punktów
i własne peaki. Wcześniej brana była detekcja o najwyższej pewności, więc na wideo
z wieloma psami baza AU mogła należeć do innego psa niż klatka docelowa.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from packages.data.schemas import NUM_KEYPOINTS, Keypoint
from packages.models import (
    BBoxConfig,
    BBoxModel,
    BreedConfig,
    BreedModel,
    BreedPrediction,
    EmotionPrediction,
    KeypointsConfig,
    KeypointsModel,
    KeypointsPrediction,
)
from packages.models.bbox import Detection
from packages.models.delta_action_units import DeltaActionUnitsExtractor
from packages.models.head_pose import (
    DEFAULT_MAX_ROLL,
    DEFAULT_MAX_YAW_ASYMMETRY,
    estimate_head_pose,
)
from packages.pipeline.dog_tracker import DogTracker
from packages.pipeline.landmark_smoothing import KeypointSmoother
from packages.pipeline.neutral_frame import NeutralFrameDetector, collect_neutral_baseline
from packages.pipeline.peak_selector import PeakFrameSelector
from packages.pipeline.track_processing import (
    DEFAULT_TRACK_QUALITY,
    NEUTRAL_SOURCE_AUTO,
    NEUTRAL_SOURCE_MANUAL,
    TrackFrame,
    TrackQuality,
    TrackResult,
    build_track_result,
    rejected_track,
)

logger = logging.getLogger(__name__)

# Awarie pojedynczego treku, które wolno zamienić na odrzucenie z powodem.
# Celowo NIE ma tu `Exception`: błąd typu czy braku atrybutu to pomyłka w kodzie
# i ma być widoczny od razu, a nie ukryty jako „trek odrzucony".
_TRACK_FAILURES = (ValueError, IndexError, KeyError, ArithmeticError)

# Domyślne próbkowanie klatek wideo [kl./s] — takie, przy jakim zbierany jest zbiór
DEFAULT_SAMPLING_FPS: float = 5.0
# Minimalny odstęp między peakami [s] — tak jak w opublikowanej metodzie (spec, 4.2)
DEFAULT_MIN_PEAK_SEPARATION_S: float = 1.0
# Minimalna średnia pewność keypoints klatki
DEFAULT_MIN_KEYPOINT_CONF: float = 0.5
# Minimalna ostrość mordy (wariancja Laplaciana) — filtr rozmycia ruchowego
DEFAULT_MIN_SHARPNESS: float = 60.0
# Promień okna wokół klatki neutralnej, z którego liczona jest baza median.
# UWAGA: to promień w POZYCJACH treku, nie w klatkach wideo — na treku dziurawym
# (pies wychodzi z kadru) okno ±2 pozycje może objąć klatki oddalone o sekundy.
NEUTRAL_BASELINE_WINDOW: int = 2

# Podział paska postępu między etapy (procenty)
_PROGRESS_TRACKING_END: int = 40
_PROGRESS_TRACKS_END: int = 99


@dataclass(frozen=True)
class _FrameMeasurement:
    """
    Surowy pomiar keypoints jednej klatki treku (przed wygładzeniem).

    Attributes:
        frame_idx: Numer klatki wideo
        keypoints: 138 wartości [x0, y0, v0, ...] w układzie obrazu
        face_box: Boks mordy, w którym zmierzono keypoints
        body_box: Boks psa z detektora, przycięty do kadru
    """

    frame_idx: int
    keypoints: np.ndarray
    face_box: tuple[float, float, float, float]
    body_box: tuple[int, int, int, int]


@dataclass(frozen=True)
class VideoDatasetConfig:
    """
    Progi przetwarzania wideo na treki.

    Trzymane w jednym obiekcie, bo lista parametrów `process_video_for_dataset`
    urosła ponad czytelność.

    `min_keypoint_conf` i `track_quality.min_conf` to CELOWO dwie gałki, mimo że
    obie dotyczą pewności keypoints. Pierwsza odsiewa pojedyncze klatki (wybór
    peaków i klatki neutralnej), druga decyduje o losie całego treku. Gdy jedna
    sterowała obiema, luźny filtr klatek (0.3 w batchu) obniżał bramkę godności
    treku poniżej modułowego minimum 0.4 — czyli poluzowanie doboru klatek po
    cichu wpuszczało do zbioru treki uznane przez moduł za niewiarygodne.

    Attributes:
        num_peaks: Liczba klatek szczytowych do wybrania na trek
        fps: Próbkowanie klatek wejściowych [kl./s]; wchodzi w znaczniki czasu
            filtra wygładzającego i w przeliczenie odstępu peaków na klatki
        neutral_frame_idx: Ręcznie wskazany numer klatki wideo jako neutralna;
            trek, który tej klatki nie zawiera, dostaje klatkę neutralną z
            auto-detekcji (odnotowane w `TrackResult.neutral_source`)
        min_peak_separation_s: Minimalny odstęp między peakami [s] — twardy
        min_keypoint_conf: Minimalna średnia pewność keypoints POJEDYNCZEJ klatki
        max_yaw_asymmetry: Maks. asymetria kącik oka <-> nos (obrót głowy)
        max_roll: Maks. przechylenie głowy [stopnie]
        min_sharpness: Minimalna ostrość mordy; 0 wyłącza filtr rozmycia
        track_quality: Progi godności CAŁEGO treku (liczba klatek, rozmiar mordy,
            mediana pewności)
    """

    num_peaks: int = 10
    fps: float = DEFAULT_SAMPLING_FPS
    neutral_frame_idx: Optional[int] = None
    min_peak_separation_s: float = DEFAULT_MIN_PEAK_SEPARATION_S
    min_keypoint_conf: float = DEFAULT_MIN_KEYPOINT_CONF
    max_yaw_asymmetry: float = DEFAULT_MAX_YAW_ASYMMETRY
    max_roll: float = DEFAULT_MAX_ROLL
    min_sharpness: float = DEFAULT_MIN_SHARPNESS
    track_quality: TrackQuality = DEFAULT_TRACK_QUALITY

    def __post_init__(self) -> None:
        """
        Sprawdza progi.

        Raises:
            ValueError: Gdy któryś próg jest poza sensownym zakresem
        """
        if self.fps <= 0.0:
            raise ValueError(f"fps musi być dodatnie, otrzymano {self.fps}")
        if self.num_peaks < 1:
            raise ValueError(f"num_peaks musi być dodatnie, otrzymano {self.num_peaks}")
        if not 0.0 < self.min_keypoint_conf <= 1.0:
            raise ValueError(
                f"min_keypoint_conf musi być w (0, 1], otrzymano {self.min_keypoint_conf}"
            )
        if self.min_peak_separation_s <= 0.0:
            raise ValueError(
                f"min_peak_separation_s musi być dodatnie, otrzymano {self.min_peak_separation_s}"
            )
        if self.max_yaw_asymmetry <= 0.0 or self.max_roll <= 0.0:
            raise ValueError(
                "Progi pozy głowy muszą być dodatnie, otrzymano "
                f"max_yaw_asymmetry={self.max_yaw_asymmetry}, max_roll={self.max_roll}"
            )
        if self.min_sharpness < 0.0:
            raise ValueError(
                f"min_sharpness nie może być ujemne, otrzymano {self.min_sharpness}"
            )

    @property
    def peak_separation_frames(self) -> int:
        """
        Odstęp peaków w pozycjach treku przy zadanym próbkowaniu.

        Odstęp liczony jest w pozycjach w obrębie treku, a trek bywa dziurawy
        (pies wychodzi z kadru), więc realna przerwa czasowa jest nie mniejsza
        niż zamówiona — nigdy krótsza. Selektor traktuje ten odstęp jako TWARDY:
        wideo bez dość odległych szczytów daje ich po prostu mniej.
        """
        return max(1, round(self.min_peak_separation_s * self.fps))


@dataclass
class PipelineConfig:
    """
    Konfiguracja pipeline inference.

    Attributes:
        bbox_weights: Ścieżka do wag modelu detekcji
        breed_weights: Ścieżka do wag modelu klasyfikacji ras
        keypoints_weights: Ścieżka do wag modelu keypoints
        emotion_weights: Ścieżka do wag modelu emocji
        breeds_json: Ścieżka do pliku z mapowaniem ras
        device: Urządzenie ('cuda', 'cpu')
        confidence_threshold: Próg pewności detekcji
        max_dogs: Maksymalna liczba psów do przetworzenia
    """

    bbox_weights: Path = field(default_factory=lambda: Path("models/yolov8m.pt"))
    breed_weights: Path = field(default_factory=lambda: Path("models/breed.pt"))
    keypoints_weights: Path = field(
        default_factory=lambda: Path("models/keypoints_dogflw.pt")
    )
    emotion_weights: Path = field(default_factory=lambda: Path("models/emotion.pt"))
    breeds_json: Path = field(
        default_factory=lambda: Path("packages/models/breeds.json")
    )
    device: str = "cuda"
    confidence_threshold: float = 0.3
    max_dogs: int = 10
    use_rule_based_emotion: bool = True  # Rule-based emotion (nie wymaga treningu)
    keypoints_two_pass: bool = True  # Dwuprzebiegowa detekcja keypoints (zoom na mordę)
    # Multi-crop TTA: skale cropu mordy w przebiegu 2 (uśrednienie predykcji).
    # Pojedyncza skala (1.6,) = bez TTA; kilka skal = stabilniejsza lokalizacja.
    keypoints_tta_expands: tuple[float, ...] = (1.5, 1.7)
    # Detektor MORDY (YOLOv8) — kadruje twarz pod keypoints zamiast zawodnego
    # two-pass z bboxa ciała (fix: pies na cały kadr). Fallback gdy brak wag.
    use_face_detector: bool = True
    face_detector_weights: Path = field(
        default_factory=lambda: Path("models/dogface_yolo.pt")
    )

    def __post_init__(self) -> None:
        """Konwertuje ścieżki do Path."""
        if isinstance(self.bbox_weights, str):
            self.bbox_weights = Path(self.bbox_weights)
        if isinstance(self.breed_weights, str):
            self.breed_weights = Path(self.breed_weights)
        if isinstance(self.keypoints_weights, str):
            self.keypoints_weights = Path(self.keypoints_weights)
        if isinstance(self.emotion_weights, str):
            self.emotion_weights = Path(self.emotion_weights)
        if isinstance(self.breeds_json, str):
            self.breeds_json = Path(self.breeds_json)


@dataclass
class DogAnnotation:
    """
    Pełna anotacja dla jednego psa.

    Attributes:
        dog_id: ID psa na obrazie (0, 1, 2, ...)
        bbox: Bounding box jako (x, y, w, h)
        bbox_confidence: Pewność detekcji bbox
        breed: Predykcja rasy (opcjonalnie)
        keypoints: Predykcja keypoints (opcjonalnie)
        emotion: Predykcja emocji (opcjonalnie)
    """

    dog_id: int
    bbox: tuple[int, int, int, int]
    bbox_confidence: float
    breed: Optional[BreedPrediction] = None
    keypoints: Optional[KeypointsPrediction] = None
    emotion: Optional[EmotionPrediction] = None

    def to_dict(self) -> dict:
        """Konwertuje anotację do słownika."""
        result = {
            "dog_id": self.dog_id,
            "bbox": list(self.bbox),
            "bbox_confidence": self.bbox_confidence,
        }

        if self.breed:
            result["breed"] = self.breed.class_name
            result["breed_id"] = self.breed.class_id
            result["breed_confidence"] = self.breed.confidence

        if self.keypoints:
            result["keypoints"] = [
                {"x": kp.x, "y": kp.y, "visibility": kp.visibility}
                for kp in self.keypoints.keypoints
            ]
            result["keypoints_confidence"] = self.keypoints.confidence
            result["num_keypoints"] = self.keypoints.num_detected

        if self.emotion:
            result["emotion"] = self.emotion.emotion
            result["emotion_id"] = self.emotion.emotion_id
            result["emotion_confidence"] = self.emotion.confidence

        return result

    def to_coco(self, image_id: int, annotation_id: int) -> dict:
        """
        Konwertuje do formatu COCO annotation.

        Args:
            image_id: ID obrazu w datasecie
            annotation_id: Unikalny ID anotacji

        Returns:
            Słownik w formacie COCO
        """
        x, y, w, h = self.bbox
        coco_ann = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,  # dog
            "bbox": [x, y, w, h],
            "area": w * h,
            "iscrowd": 0,
            "confidence": {
                "bbox": self.bbox_confidence,
            },
        }

        if self.breed:
            coco_ann["breed_id"] = self.breed.class_id
            coco_ann["breed"] = self.breed.class_name
            coco_ann["confidence"]["breed"] = self.breed.confidence

        if self.keypoints:
            coco_ann["keypoints"] = self.keypoints.to_coco_format()
            coco_ann["num_keypoints"] = self.keypoints.num_detected
            coco_ann["confidence"]["keypoints"] = self.keypoints.confidence

        if self.emotion:
            coco_ann["emotion_id"] = self.emotion.emotion_id
            coco_ann["emotion"] = self.emotion.emotion
            coco_ann["confidence"]["emotion"] = self.emotion.confidence

        return coco_ann


@dataclass
class FrameResult:
    """
    Wynik przetwarzania jednej klatki/obrazu.

    Attributes:
        frame_id: ID klatki (lub ścieżka do obrazu)
        width: Szerokość obrazu
        height: Wysokość obrazu
        annotations: Lista anotacji psów
    """

    frame_id: str | int
    width: int
    height: int
    annotations: list[DogAnnotation] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Konwertuje wynik do słownika."""
        return {
            "frame_id": self.frame_id,
            "width": self.width,
            "height": self.height,
            "num_dogs": len(self.annotations),
            "annotations": [ann.to_dict() for ann in self.annotations],
        }


class InferencePipeline:
    """
    Zunifikowany pipeline inference dla anotacji psów.

    Pipeline przetwarza obrazy przez 4 modele:
    1. Detekcja psów (BBox)
    2. Klasyfikacja rasy
    3. Detekcja keypoints
    4. Klasyfikacja emocji

    Użycie:
        config = PipelineConfig()
        pipeline = InferencePipeline(config)
        pipeline.load()

        result = pipeline.process_frame(image)
        for ann in result.annotations:
            print(f"Dog {ann.dog_id}: {ann.breed.class_name}, {ann.emotion.emotion}")
    """

    def __init__(self, config: PipelineConfig) -> None:
        """
        Inicjalizuje pipeline.

        Args:
            config: Konfiguracja pipeline
        """
        self.config = config
        self._models_loaded = False

        # Modele
        self.bbox_model: Optional[BBoxModel] = None
        self.breed_model: Optional[BreedModel] = None
        self.keypoints_model: Optional[KeypointsModel] = None
        # Detektor mordy — ustawiany w load(); tu, żeby odczyt przed load()
        # nie kończył się AttributeError
        self.face_model: Optional[object] = None
        # Emotion: rule-based only (no ML model needed)

    @property
    def is_loaded(self) -> bool:
        """Sprawdza czy modele zostały załadowane."""
        return self._models_loaded

    def load(self) -> None:
        """
        Ładuje wszystkie modele.

        Raises:
            FileNotFoundError: Gdy brakuje plików z wagami
        """
        print("=" * 60)
        print("ŁADOWANIE PIPELINE")
        print("=" * 60)

        # 1. BBox model
        print("\n[1/4] Ładowanie modelu detekcji...")
        bbox_config = BBoxConfig(
            weights_path=self.config.bbox_weights,
            device=self.config.device,
            confidence_threshold=self.config.confidence_threshold,
            max_detections=self.config.max_dogs,
        )
        self.bbox_model = BBoxModel(bbox_config)
        self.bbox_model.load()

        # 2. Breed model
        print("\n[2/4] Ładowanie modelu klasyfikacji ras...")
        if self.config.breed_weights.exists():
            breed_config = BreedConfig(
                weights_path=self.config.breed_weights,
                labels_path=self.config.breeds_json,
                device=self.config.device,
            )
            self.breed_model = BreedModel(breed_config)
            self.breed_model.load()
        else:
            print(f"  ! Brak wag: {self.config.breed_weights}")

        # 3. Keypoints model
        print("\n[3/4] Ładowanie modelu keypoints...")
        if self.config.keypoints_weights.exists():
            keypoints_config = KeypointsConfig(
                weights_path=self.config.keypoints_weights,
                device=self.config.device,
            )
            self.keypoints_model = KeypointsModel(keypoints_config)
            self.keypoints_model.load()
        else:
            print(f"  ! Brak wag: {self.config.keypoints_weights}")

        # 3b. Detektor mordy (opcjonalny — kadruje twarz pod keypoints)
        self.face_model = None
        if self.config.use_face_detector and self.config.face_detector_weights.exists():
            try:
                from ultralytics import YOLO

                self.face_model = YOLO(str(self.config.face_detector_weights))
                print(f"  Detektor mordy: {self.config.face_detector_weights}")
            except Exception as e:
                print(f"  ! Detektor mordy niedostępny: {e}")

        # 4. Emotion classification (Rule-based only, NO ML)
        print("\n[4/4] Konfiguracja klasyfikacji emocji...")
        # Bez znaków spoza cp1250: przy przekierowanym stdout konsola Windows
        # wywracała przetwarzanie na UnicodeEncodeError
        print("  Używam RULE-BASED classification (DogFACS)")
        print("  Nie wymaga wytrenowanego modelu!")

        self._models_loaded = True
        print("\n" + "=" * 60)
        print("PIPELINE ZAŁADOWANY")
        print("=" * 60)

    @staticmethod
    def _square_crop(
        image: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        expand: float = 1.1,
    ) -> tuple[np.ndarray, int, int]:
        """
        Wycina kwadratowy region wokół bboxa (z zachowaniem proporcji).

        Model keypoints (DogFLW) skaluje wejście do 256x256. Podanie
        niekwadratowego cropa powoduje zniekształcenie proporcji twarzy
        (a przez to błędną geometrię i delta AU). Kwadratowy crop to eliminuje.

        Args:
            image: Pełny obraz
            x, y, w, h: Bounding box psa
            expand: Współczynnik powiększenia boku kwadratu

        Returns:
            (kwadratowy_crop, x0, y0) — crop oraz offset jego lewego górnego rogu
        """
        img_h, img_w = image.shape[:2]
        cx, cy = x + w / 2, y + h / 2
        side = max(w, h) * expand
        x0 = int(max(0, cx - side / 2))
        y0 = int(max(0, cy - side / 2))
        x1 = int(min(img_w, cx + side / 2))
        y1 = int(min(img_h, cy + side / 2))
        return image[y0:y1, x0:x1], x0, y0

    @staticmethod
    def _keypoints_face_region(
        keypoints: list,
        conf_min: float = 0.2,
    ) -> tuple[float, float, float, float]:
        """
        Wyznacza bounding box regionu mordy na podstawie pewnych keypoints.

        Args:
            keypoints: Lista keypoints (układ wejścia 1. przebiegu)
            conf_min: Minimalna pewność punktu, by uwzględnić go w regionie

        Returns:
            (x, y, w, h) regionu mordy w układzie wejścia keypoints
        """
        pts = [(kp.x, kp.y) for kp in keypoints if kp.visibility >= conf_min]
        if len(pts) < 5:
            pts = [(kp.x, kp.y) for kp in keypoints]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)

    def _predict_keypoints_two_pass(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> KeypointsPrediction:
        """
        Dwuprzebiegowa detekcja keypoints ze współrzędnymi w układzie pełnego obrazu.

        Model DogFLW oczekuje ciasnego cropa mordy. Bbox psa z detektora obejmuje
        całe ciało, przez co morda zajmuje mały, przesunięty fragment kadru i punkty
        rozjeżdżają się. Dlatego:
        1. przebieg zgrubny na kwadratowym cropie bboxa psa,
        2. wyznaczenie regionu mordy z pewnych keypoints,
        3. przebieg dokładny na zawężonym kwadratowym cropie mordy.

        Args:
            image: Pełny obraz
            x, y, w, h: Bounding box psa

        Returns:
            KeypointsPrediction we współrzędnych pełnego obrazu
        """
        # Ścieżka A: detektor mordy (jeśli dostępny) — kadruje twarz wprost,
        # niezawodnie nawet gdy pies zajmuje cały kadr.
        if self.face_model is not None:
            face = self._detect_face(image, x, y, w, h)
            if face is not None:
                pred = self._keypoints_on_region(image, *face)
                if pred is not None:
                    return pred

        return self._keypoints_from_body_bbox(image, x, y, w, h)

    def _keypoints_from_body_bbox(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> KeypointsPrediction:
        """
        Ścieżka zapasowa: dwuprzebiegowy zoom na mordę z bboxa ciała.

        Wydzielona z `_predict_keypoints_two_pass`, żeby wywołujący, który już
        wie, że detektor mordy nic nie znalazł, nie uruchamiał go po raz drugi.

        Args:
            image: Pełny obraz
            x, y, w, h: Bounding box psa

        Returns:
            KeypointsPrediction we współrzędnych pełnego obrazu
        """
        # Przebieg 1: kwadratowy crop całego bboxa psa
        crop1, ox1, oy1 = self._square_crop(image, x, y, w, h, expand=1.1)
        pred1 = self.keypoints_model.predict(crop1)

        if not self.config.keypoints_two_pass:
            for kp in pred1.keypoints:
                kp.x += ox1
                kp.y += oy1
            return pred1

        # Region mordy z pewnych keypoints → układ pełnego obrazu
        fx, fy, fw, fh = self._keypoints_face_region(pred1.keypoints)
        fx += ox1
        fy += oy1

        # Przebieg 2: multi-crop TTA — kilka skal cropu mordy, uśrednienie
        # predykcji ważone widocznością (stabilniejsza lokalizacja).
        preds = []
        for ex in self.config.keypoints_tta_expands:
            crop2, ox2, oy2 = self._square_crop(
                image, int(fx), int(fy), int(fw), int(fh), expand=ex
            )
            if crop2.size == 0 or min(crop2.shape[:2]) <= 10:
                continue
            pr = self.keypoints_model.predict(crop2)
            for kp in pr.keypoints:
                kp.x += ox2
                kp.y += oy2
            preds.append(pr)

        if not preds:
            # Fallback: wynik 1. przebiegu
            for kp in pred1.keypoints:
                kp.x += ox1
                kp.y += oy1
            return pred1
        if len(preds) == 1:
            return preds[0]
        return self._merge_keypoint_predictions(preds)

    @staticmethod
    def _merge_keypoint_predictions(
        preds: list[KeypointsPrediction],
    ) -> KeypointsPrediction:
        """Uśrednia predykcje keypoints per-punkt, ważąc widocznością (multi-crop TTA)."""
        merged = []
        for i in range(NUM_KEYPOINTS):
            pts = [p.keypoints[i] for p in preds]
            wsum = sum(max(k.visibility, 0.0) for k in pts)
            if wsum > 1e-6:
                x = sum(max(k.visibility, 0.0) * k.x for k in pts) / wsum
                y = sum(max(k.visibility, 0.0) * k.y for k in pts) / wsum
            else:
                x = sum(k.x for k in pts) / len(pts)
                y = sum(k.y for k in pts) / len(pts)
            merged.append(
                Keypoint(x=x, y=y, visibility=max(k.visibility for k in pts))
            )
        conf = float(np.mean([k.visibility for k in merged]))
        visible = sum(1 for k in merged if k.visibility > 0.15)
        return KeypointsPrediction(
            keypoints=merged, confidence=conf, num_detected=visible
        )

    def _detect_face(
        self, image: np.ndarray, x: int, y: int, w: int, h: int
    ) -> Optional[tuple[float, float, float, float]]:
        """
        Wykrywa mordę w regionie bboxa ciała (z zapasem). Zwraca (fx, fy, fw, fh)
        w układzie pełnego obrazu lub None.
        """
        pad = int(0.10 * max(w, h))
        ih, iw = image.shape[:2]
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(iw, x + w + pad), min(ih, y + h + pad)
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        try:
            res = self.face_model.predict(
                crop, verbose=False, conf=0.25, device=self.config.device
            )
        except Exception:
            return None
        boxes = res[0].boxes if res else None
        if boxes is None or len(boxes) == 0:
            return None
        i = int(boxes.conf.argmax())
        bx0, by0, bx1, by1 = (float(v) for v in boxes.xyxy[i].tolist())
        return (bx0 + x0, by0 + y0, bx1 - bx0, by1 - by0)

    def _keypoints_on_region(
        self, image: np.ndarray, fx: float, fy: float, fw: float, fh: float
    ) -> Optional[KeypointsPrediction]:
        """Keypoints na regionie mordy z multi-crop TTA (współrz. pełnego obrazu)."""
        preds = []
        for ex in self.config.keypoints_tta_expands:
            crop, ox, oy = self._square_crop(
                image, int(fx), int(fy), int(fw), int(fh), expand=ex
            )
            if crop.size == 0 or min(crop.shape[:2]) <= 10:
                continue
            pr = self.keypoints_model.predict(crop)
            for kp in pr.keypoints:
                kp.x += ox
                kp.y += oy
            preds.append(pr)
        if not preds:
            return None
        return preds[0] if len(preds) == 1 else self._merge_keypoint_predictions(preds)

    def process_frame(
        self,
        image: np.ndarray,
        frame_id: str | int = 0,
    ) -> FrameResult:
        """
        Przetwarza pojedynczą klatkę przez wszystkie modele.

        Args:
            image: Obraz jako numpy array (BGR lub RGB)
            frame_id: ID klatki

        Returns:
            FrameResult z anotacjami wszystkich psów

        Raises:
            RuntimeError: Gdy modele nie zostały załadowane
        """
        if not self._models_loaded:
            raise RuntimeError("Pipeline nie załadowany. Wywołaj load() najpierw.")

        height, width = image.shape[:2]
        result = FrameResult(
            frame_id=frame_id,
            width=width,
            height=height,
        )

        # 1. Detekcja psów
        detections = self.bbox_model.predict(image)

        if not detections:
            return result

        # 2. Przetwarzanie każdego psa
        for dog_id, detection in enumerate(detections[: self.config.max_dogs]):
            x, y, w, h = detection.bbox

            # Upewnij się, że bbox mieści się w obrazie
            x = max(0, x)
            y = max(0, y)
            w = min(w, width - x)
            h = min(h, height - y)

            # Wytnij psa
            cropped = image[y : y + h, x : x + w]

            if cropped.size == 0:
                continue

            annotation = DogAnnotation(
                dog_id=dog_id,
                bbox=(x, y, w, h),
                bbox_confidence=detection.confidence,
            )

            # Klasyfikacja rasy
            if self.breed_model is not None:
                try:
                    breed_pred = self.breed_model.predict(cropped)
                    annotation.breed = breed_pred
                except Exception as e:
                    print(f"  ! Błąd klasyfikacji rasy: {e}")

            # Detekcja keypoints — dwuprzebiegowa (zoom na mordę),
            # ze współrzędnymi w układzie pełnego obrazu.
            if self.keypoints_model is not None:
                try:
                    annotation.keypoints = self._predict_keypoints_two_pass(
                        image, x, y, w, h
                    )
                except Exception as e:
                    print(f"  ! Błąd detekcji keypoints: {e}")

            # Klasyfikacja emocji wymaga neutral frame — pomijamy w trybie single-frame.
            # Emocja jest klasyfikowana w process_video_for_dataset() po auto-detekcji
            # klatki neutralnej i obliczeniu delta AU względem niej.

            result.annotations.append(annotation)

        return result

    def process_video_for_dataset(
        self,
        frames_list: list[np.ndarray],
        config: Optional[VideoDatasetConfig] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> dict:
        """
        Przetwarza sekwencję wideo na treki psów gotowe do zapisu w zbiorze.

        Dla każdego psa osobno: klatki treku (odsiane progiem pewności) →
        wygładzone keypoints → własna klatka neutralna → delta AU względem TEJ
        klatki → peaki. Trek, który nie przechodzi progu godności ALBO którego
        przetwarzanie się wywróciło, trafia do `rejected_tracks` z zapisanym
        powodem — wideo bez godnych treków nie kończy się wyjątkiem, bo cichy
        brak wyniku i błąd to dwie różne informacje dla audytu.

        Args:
            frames_list: Kolejne klatki wideo (próbkowane z `config.fps`)
            config: Progi przetwarzania (domyślnie `VideoDatasetConfig()`)
            progress_callback: Wywoływane jako `(procent, etap)`

        Returns:
            Słownik z wynikami:
            {
                "tracks": list[TrackResult],           # treki przyjęte
                "rejected_tracks": list[TrackResult],  # z wypełnionym rejected_reason
                "total_frames": int,
            }

        Raises:
            RuntimeError: Gdy modele nie zostały załadowane
        """
        if not self._models_loaded:
            raise RuntimeError("Pipeline nie załadowany. Wywołaj load() najpierw.")

        config = config or VideoDatasetConfig()
        report = _progress_reporter(progress_callback)

        frames_by_track = self._assign_tracks(frames_list, report)
        accepted: list[TrackResult] = []
        rejected: list[TrackResult] = []

        for done, (track_id, entries) in enumerate(sorted(frames_by_track.items()), 1):
            track = self._process_track(frames_list, track_id, entries, config)
            (rejected if track.rejected_reason else accepted).append(track)
            report(
                _progress_percent(
                    done, len(frames_by_track), _PROGRESS_TRACKING_END, _PROGRESS_TRACKS_END
                ),
                "Processing dog tracks...",
            )

        _log_track_summary(accepted, rejected, len(frames_list))
        return {
            "tracks": accepted,
            "rejected_tracks": rejected,
            "total_frames": len(frames_list),
        }

    def _assign_tracks(
        self,
        frames_list: list[np.ndarray],
        report: Callable[[int, str], None],
    ) -> dict[int, list[tuple[int, Detection]]]:
        """
        Przypisuje detekcje psów do treków przez całe wideo.

        Jeden `DogTracker` na wideo — to on trzyma tożsamość psa między klatkami.

        Args:
            frames_list: Kolejne klatki wideo
            report: Raportowanie postępu

        Returns:
            Słownik track_id → lista (numer klatki wideo, detekcja)
        """
        tracker = DogTracker()
        frames_by_track: dict[int, list[tuple[int, Detection]]] = {}

        for frame_idx, frame in enumerate(frames_list):
            detections = self.bbox_model.predict(frame)
            track_ids = tracker.update(frame, detections)
            for track_id, detection in zip(track_ids, detections):
                frames_by_track.setdefault(track_id, []).append((frame_idx, detection))
            report(
                _progress_percent(frame_idx + 1, len(frames_list), 0, _PROGRESS_TRACKING_END),
                "Tracking dogs...",
            )

        return frames_by_track

    def _process_track(
        self,
        frames_list: list[np.ndarray],
        track_id: int,
        entries: list[tuple[int, Detection]],
        config: VideoDatasetConfig,
    ) -> TrackResult:
        """
        Przetwarza jeden trek, zamieniając jego awarię w odrzucenie z powodem.

        Bez tej izolacji jeden zepsuty trek (np. zdegenerowana geometria w
        ekstraktorze AU) wywracał przetwarzanie CAŁEGO wideo, a `batch_annotate`
        pomijał wtedy nagranie w całości — razem ze wszystkimi pozostałymi psami.

        Args:
            frames_list: Kolejne klatki wideo
            track_id: Identyfikator treku z `DogTracker`
            entries: Klatki tego treku jako (numer klatki wideo, detekcja)
            config: Progi przetwarzania

        Returns:
            Wynik treku — przyjęty albo odrzucony z wypełnionym `rejected_reason`
        """
        track_frames = self._build_track_frames(frames_list, entries, config)

        try:
            return self._finish_track(frames_list, track_id, track_frames, config)
        except _TRACK_FAILURES as error:
            return rejected_track(
                track_id,
                track_frames,
                f"błąd przetwarzania treku: {type(error).__name__}: {error}",
            )

    def _finish_track(
        self,
        frames_list: list[np.ndarray],
        track_id: int,
        track_frames: list[TrackFrame],
        config: VideoDatasetConfig,
    ) -> TrackResult:
        """
        Domyka trek: klatka neutralna → delta AU → peaki → wynik.

        Args:
            frames_list: Kolejne klatki wideo
            track_id: Identyfikator treku z `DogTracker`
            track_frames: Klatki treku z wygładzonymi keypoints
            config: Progi przetwarzania

        Returns:
            Wynik treku — przyjęty albo odrzucony z wypełnionym `rejected_reason`

        Raises:
            ValueError: Gdy dane treku są niepoliczalne (np. zdegenerowana geometria)
            IndexError: Gdy pozycja wykracza poza trek
            KeyError: Gdy brakuje oczekiwanego AU
            ArithmeticError: Gdy pomiar prowadzi do dzielenia przez zero
        """
        if not track_frames:
            # Bramka godności zna właściwy komunikat („za mało klatek z mordą"),
            # a detektor klatki neutralnej rzuciłby tylko „sekwencja jest pusta".
            return build_track_result(track_id, track_frames, 0, [], quality=config.track_quality)

        try:
            neutral_position, neutral_source = self._neutral_position(
                frames_list, track_frames, config
            )
        except ValueError as error:
            return rejected_track(track_id, track_frames, f"brak klatki neutralnej: {error}")

        baseline = collect_neutral_baseline(
            [frame.keypoints for frame in track_frames],
            neutral_position,
            window=NEUTRAL_BASELINE_WINDOW,
        )
        extractor = DeltaActionUnitsExtractor(baseline)
        for track_frame in track_frames:
            track_frame.delta_aus = extractor.extract(track_frame.keypoints)

        # Próg godności i zamiana pozycji w treku na numery klatek wideo są
        # w `build_track_result` — tylko tam liczą się też szum AU i liczba próbek.
        return build_track_result(
            track_id,
            track_frames,
            neutral_position,
            self._select_peaks(frames_list, track_frames, neutral_position, config),
            quality=config.track_quality,
            neutral_source=neutral_source,
        )

    def _build_track_frames(
        self,
        frames_list: list[np.ndarray],
        entries: list[tuple[int, Detection]],
        config: VideoDatasetConfig,
    ) -> list[TrackFrame]:
        """
        Mierzy i wygładza keypoints wszystkich klatek jednego treku.

        Dwa przebiegi, bo wygładzanie potrzebuje skali znanej dla CAŁEGO treku
        (patrz `_smooth_track`).

        Args:
            frames_list: Kolejne klatki wideo
            entries: Klatki tego treku jako (numer klatki wideo, detekcja)
            config: Progi przetwarzania

        Returns:
            Klatki treku z pewnym pomiarem mordy (pozostałe są odsiane)
        """
        measurements, low_confidence = self._measure_track(frames_list, entries, config)

        # Diagnostyka lejka: bez niej „trek ma 3 klatki" nie mówi, czy pies wyszedł
        # z kadru, czy pomiary odsiał próg pewności.
        logger.info(
            "Filtr jakości: %d/%d klatek z pewnymi keypoints (prog conf %.2f); "
            "bez pomiaru mordy: %d, ponizej progu: %d",
            len(measurements),
            len(entries),
            config.min_keypoint_conf,
            len(entries) - len(measurements) - low_confidence,
            low_confidence,
        )
        return self._smooth_track(measurements, config)

    def _measure_track(
        self,
        frames_list: list[np.ndarray],
        entries: list[tuple[int, Detection]],
        config: VideoDatasetConfig,
    ) -> tuple[list["_FrameMeasurement"], int]:
        """
        Przebieg 1: surowe keypoints klatek treku wraz z odsiewem niepewnych.

        Klatka poniżej progu pewności jest odrzucana TU, przed wygładzaniem i przed
        liczeniem AU. Wcześniej trafiała do treku i mimo że nigdy nie weszłaby do
        zbioru, współtworzyła medianową bazę AU, zawyżała zmierzony szum treku
        (czyli obniżała wagę wiarygodności pozostałych klatek) i ciągnęła w dół
        medianę pewności w progu godności.

        Args:
            frames_list: Kolejne klatki wideo
            entries: Klatki tego treku jako (numer klatki wideo, detekcja)
            config: Progi przetwarzania

        Returns:
            (pomiary klatek, liczba klatek odrzuconych przez próg pewności)
        """
        measurements: list[_FrameMeasurement] = []
        low_confidence = 0

        for frame_idx, detection in entries:
            frame = frames_list[frame_idx]
            body_box = _clamp_bbox(detection.bbox, frame.shape[0], frame.shape[1])
            if body_box[2] <= 0 or body_box[3] <= 0:
                continue

            measured = self._face_keypoints(frame, body_box)
            if measured is None:
                continue

            prediction, face_box = measured
            keypoints = np.asarray(prediction.to_coco_format(), dtype=float)
            if _mean_visibility(keypoints) < config.min_keypoint_conf:
                low_confidence += 1
                continue

            measurements.append(
                _FrameMeasurement(
                    frame_idx=frame_idx,
                    keypoints=keypoints,
                    face_box=face_box,
                    body_box=body_box,
                )
            )

        return measurements, low_confidence

    def _smooth_track(
        self,
        measurements: list["_FrameMeasurement"],
        config: VideoDatasetConfig,
    ) -> list[TrackFrame]:
        """
        Przebieg 2: wygładzanie keypoints treku przy STAŁEJ skali odniesienia.

        Filtr wygładzający ma stan, więc jedna instancja przypada na JEDEN trek —
        współdzielenie mieszałoby ruch dwóch psów. Znacznik czasu liczony jest
        z numeru klatki i próbkowania, więc luki w treku nie udają ciągłości.

        Skala normalizacji to MEDIANA rozmiarów boksu mordy w treku, a nie boks
        z bieżącej klatki. Boks per klatka miał dwa źródła (detektor mordy albo
        otoczka keypoints, systematycznie ciaśniejsza) i sam drgał — przełączenie
        źródła dawało skokową zmianę układu współrzędnych bez ruchu psa, którą
        filtr czytał jako dużą prędkość i przestawał tłumić dokładnie tam, gdzie
        pomiar był najtrudniejszy. Przesunięcie zostaje per klatka (od środka
        boksu mordy), bo ono ma podążać za psem.

        Args:
            measurements: Pomiary klatek treku z przebiegu 1
            config: Progi przetwarzania

        Returns:
            Klatki treku z wygładzonymi keypoints
        """
        if not measurements:
            return []

        smoother = KeypointSmoother()
        scale = _median_face_scale(measurements)
        track_frames: list[TrackFrame] = []

        for measurement in measurements:
            smoothed = smoother.smooth(
                measurement.keypoints,
                _normalisation_box(measurement.face_box, scale),
                timestamp=measurement.frame_idx / config.fps,
            )
            track_frames.append(
                TrackFrame(
                    frame_idx=measurement.frame_idx,
                    keypoints=smoothed,
                    face_box=measurement.face_box,
                    body_box=measurement.body_box,
                    head_pose=estimate_head_pose(
                        smoothed,
                        max_yaw_asymmetry=config.max_yaw_asymmetry,
                        max_roll=config.max_roll,
                    ),
                )
            )

        return track_frames

    def _face_keypoints(
        self,
        frame: np.ndarray,
        body_box: tuple[int, int, int, int],
    ) -> Optional[tuple[KeypointsPrediction, tuple[float, float, float, float]]]:
        """
        Wykrywa keypoints mordy psa wraz z boksem, w którym zostały zmierzone.

        Boks mordy jest potrzebny dalej dwa razy: do normalizacji przy wygładzaniu
        i jako miara rozdzielczości pomiaru w progu godności treku.

        Args:
            frame: Pełna klatka wideo
            body_box: Bounding box psa (x, y, w, h) przycięty do kadru

        Returns:
            (predykcja keypoints, boks mordy) albo None, gdy pomiar się nie udał
        """
        if self.keypoints_model is None:
            return None

        x, y, w, h = body_box
        try:
            face_box = (
                self._detect_face(frame, x, y, w, h) if self.face_model is not None else None
            )
            if face_box is not None:
                prediction = self._keypoints_on_region(frame, *face_box)
                if prediction is not None:
                    return prediction, face_box
            # Detektor mordy nic nie znalazł — kadrujemy z bboxa ciała, a boks
            # mordy odtwarzamy z samych keypoints.
            prediction = self._keypoints_from_body_bbox(frame, x, y, w, h)
        except Exception as error:
            # Jedna zła klatka nie może przerwać przetwarzania całego wideo
            logger.warning("Blad keypoints w klatce: %s", error)
            return None

        return prediction, self._keypoints_face_region(prediction.keypoints)

    def _neutral_position(
        self,
        frames_list: list[np.ndarray],
        track_frames: list[TrackFrame],
        config: VideoDatasetConfig,
    ) -> tuple[int, str]:
        """
        Wybiera klatkę neutralną treku i zwraca jej POZYCJĘ w `track_frames`.

        Ręcznie wskazana klatka obowiązuje tylko dla treków, które ją zawierają —
        pies, którego wtedy nie było w kadrze, dostaje klatkę z auto-detekcji
        zamiast cichego podstawienia obcego kadru. Które źródło zadziałało, mówi
        druga wartość — inaczej przy dwóch psach nie da się tego odczytać z wyniku.

        Args:
            frames_list: Kolejne klatki wideo
            track_frames: Klatki treku
            config: Progi przetwarzania

        Returns:
            (pozycja klatki neutralnej w `track_frames`, źródło wyboru)

        Raises:
            ValueError: Gdy nie da się wskazać klatki neutralnej
        """
        if config.neutral_frame_idx is not None:
            for position, track_frame in enumerate(track_frames):
                if track_frame.frame_idx == config.neutral_frame_idx:
                    return position, NEUTRAL_SOURCE_MANUAL

        detector = NeutralFrameDetector(
            min_keypoint_conf=config.min_keypoint_conf,
            max_yaw_asymmetry=config.max_yaw_asymmetry,
            max_roll=config.max_roll,
        )
        position = detector.detect_auto(
            frames=[frames_list[frame.frame_idx] for frame in track_frames],
            keypoints_list=[frame.keypoints for frame in track_frames],
            head_poses=[frame.head_pose for frame in track_frames],
        )
        return position, NEUTRAL_SOURCE_AUTO

    def _select_peaks(
        self,
        frames_list: list[np.ndarray],
        track_frames: list[TrackFrame],
        neutral_position: int,
        config: VideoDatasetConfig,
    ) -> list[int]:
        """
        Wybiera POZYCJE klatek szczytowych w obrębie treku.

        Args:
            frames_list: Kolejne klatki wideo
            track_frames: Klatki treku z policzonymi delta AU
            neutral_position: Pozycja klatki neutralnej w `track_frames`
            config: Progi przetwarzania

        Returns:
            Pozycje peaków w `track_frames`, od najsilniejszego
        """
        selector = PeakFrameSelector(
            min_separation_frames=config.peak_separation_frames,
            # Ścisłej frontalności nie wymagamy — filtr progowy niżej wystarcza,
            # a wymóg `is_frontal` odrzucał praktycznie cały materiał.
            frontal_only=False,
            min_keypoint_conf=config.min_keypoint_conf,
            max_yaw_asymmetry=config.max_yaw_asymmetry,
            max_roll=config.max_roll,
            min_sharpness=config.min_sharpness,
        )
        return selector.select(
            frames=[frames_list[frame.frame_idx] for frame in track_frames],
            keypoints_list=[frame.keypoints for frame in track_frames],
            neutral_idx=neutral_position,
            delta_aus_list=[frame.delta_aus for frame in track_frames],
            head_poses=[frame.head_pose for frame in track_frames],
            num_peaks=config.num_peaks,
        )

    def visualize(
        self,
        image: np.ndarray,
        result: FrameResult,
        draw_bbox: bool = True,
        draw_keypoints: bool = True,
        draw_labels: bool = True,
    ) -> np.ndarray:
        """
        Wizualizuje wyniki na obrazie.

        Args:
            image: Oryginalny obraz
            result: Wynik przetwarzania
            draw_bbox: Czy rysować bounding boxy
            draw_keypoints: Czy rysować keypoints
            draw_labels: Czy rysować etykiety

        Returns:
            Obraz z narysowanymi anotacjami
        """
        from PIL import Image, ImageDraw

        # Kolory dla różnych psów
        colors = [
            (255, 0, 0),    # czerwony
            (0, 255, 0),    # zielony
            (0, 0, 255),    # niebieski
            (255, 255, 0),  # żółty
            (255, 0, 255),  # magenta
            (0, 255, 255),  # cyan
            (255, 128, 0),  # pomarańczowy
            (128, 0, 255),  # fioletowy
            (0, 255, 128),  # turkusowy
            (255, 128, 128),  # różowy
        ]

        # Konwertuj do PIL
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        for ann in result.annotations:
            color = colors[ann.dog_id % len(colors)]
            x, y, w, h = ann.bbox

            # Rysuj bbox
            if draw_bbox:
                draw.rectangle(
                    [(x, y), (x + w, y + h)],
                    outline=color,
                    width=3,
                )

            # Rysuj keypoints
            if draw_keypoints and ann.keypoints:
                for kp in ann.keypoints.keypoints:
                    if kp.visibility > 0.3:
                        # Przelicz współrzędne z cropa na pełny obraz
                        kp_x = x + kp.x
                        kp_y = y + kp.y
                        draw.ellipse(
                            [(kp_x - 2, kp_y - 2), (kp_x + 2, kp_y + 2)],
                            fill=color,
                            outline=(255, 255, 255),
                        )

            # Rysuj etykietę
            if draw_labels:
                label_parts = [f"Dog {ann.dog_id}"]

                if ann.breed:
                    label_parts.append(ann.breed.class_name)

                if ann.emotion:
                    label_parts.append(ann.emotion.emotion.upper())

                label = " | ".join(label_parts)

                # Tło etykiety
                text_bbox = draw.textbbox((x, y - 20), label)
                draw.rectangle(
                    [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2],
                    fill=color,
                )
                draw.text((x, y - 20), label, fill=(255, 255, 255))

        return np.array(pil_image)


# =============================================================================
# Funkcje pomocnicze (prywatne)
# =============================================================================

def _progress_reporter(
    callback: Optional[Callable[[int, str], None]],
) -> Callable[[int, str], None]:
    """
    Zwraca funkcję raportującą postęp, obojętną na brak callbacku.

    Args:
        callback: Funkcja `(procent, etap)` albo None

    Returns:
        Funkcja `(procent, etap)` bezpieczna do wywołania zawsze
    """

    def report(percent: int, stage: str) -> None:
        if callback is not None:
            callback(percent, stage)

    return report


def _progress_percent(done: int, total: int, start: int, end: int) -> int:
    """
    Przelicza postęp etapu na procent całości.

    Args:
        done: Ile pozycji etapu zrobiono
        total: Ile pozycji ma etap (0 → od razu koniec etapu)
        start: Procent na początku etapu
        end: Procent na końcu etapu

    Returns:
        Procent całości
    """
    if total <= 0:
        return end
    return start + int(done / total * (end - start))


def _clamp_bbox(
    bbox: tuple[int, int, int, int],
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    """
    Przycina bounding box do granic obrazu.

    Args:
        bbox: Bounding box (x, y, w, h)
        height: Wysokość obrazu
        width: Szerokość obrazu

    Returns:
        Bounding box wewnątrz obrazu (bok może wyjść niedodatni, gdy boks jest poza kadrem)
    """
    x, y, w, h = bbox
    x = max(0, min(int(x), width))
    y = max(0, min(int(y), height))
    return x, y, min(int(w), width - x), min(int(h), height - y)


def _log_track_summary(
    accepted: list[TrackResult],
    rejected: list[TrackResult],
    total_frames: int,
) -> None:
    """
    Loguje podsumowanie treków wraz z powodami odrzuceń.

    Powody odrzuceń są jedynym śladem, gdzie lejek danych traci materiał —
    bez nich strata jest nie do odróżnienia od braku psów na wideo.

    Komunikaty idą przez `logging`, nie przez `print`: przy przekierowanym
    stdout konsola Windows (cp1250) wywracała przetwarzanie na `UnicodeEncodeError`
    przy pierwszym znaku spoza strony kodowej.

    Args:
        accepted: Treki przyjęte
        rejected: Treki odrzucone
        total_frames: Liczba klatek wideo
    """
    logger.info(
        "Treki: %d przyjete, %d odrzucone (%d klatek wideo)",
        len(accepted),
        len(rejected),
        total_frames,
    )
    for track in rejected:
        logger.info("Trek %d odrzucony: %s", track.track_id, track.rejected_reason)
    for track in accepted:
        logger.info(
            "Trek %d przyjety: %d klatek, neutralna %d (%s), peaki %s",
            track.track_id,
            len(track.frames),
            track.neutral_frame_idx,
            track.neutral_source,
            track.peak_indices,
        )


def _mean_visibility(keypoints: np.ndarray) -> float:
    """
    Średnia pewność wszystkich keypoints klatki.

    Args:
        keypoints: 138 wartości [x0, y0, v0, ...]

    Returns:
        Średnia widoczność w zakresie [0, 1]
    """
    return float(np.mean(np.asarray(keypoints).reshape(NUM_KEYPOINTS, 3)[:, 2]))


def _median_face_scale(measurements: list[_FrameMeasurement]) -> tuple[float, float]:
    """
    Mediana rozmiarów boksu mordy w treku — stała skala odniesienia.

    Args:
        measurements: Pomiary klatek treku (niepuste)

    Returns:
        (szerokość, wysokość) skali normalizacji
    """
    widths = [measurement.face_box[2] for measurement in measurements]
    heights = [measurement.face_box[3] for measurement in measurements]
    return float(np.median(widths)), float(np.median(heights))


def _normalisation_box(
    face_box: tuple[float, float, float, float],
    scale: tuple[float, float],
) -> tuple[float, float, float, float]:
    """
    Buduje boks normalizacji: środek z bieżącej klatki, rozmiar stały dla treku.

    Środek podąża za psem (to ruch, który filtr ma śledzić), a rozmiar jest stały,
    bo drgania i przeskoki skali boksu wracały do wygładzonych współrzędnych.

    Args:
        face_box: Boks mordy bieżącej klatki (x, y, w, h)
        scale: Stała skala treku (szerokość, wysokość)

    Returns:
        Boks (x, y, w, h) do normalizacji współrzędnych
    """
    x, y, width, height = face_box
    scale_width, scale_height = scale
    return (
        x + width / 2 - scale_width / 2,
        y + height / 2 - scale_height / 2,
        scale_width,
        scale_height,
    )
