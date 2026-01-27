"""
Zunifikowany pipeline inference dla projektu Dog FACS.

Pipeline przetwarza obrazy przez wszystkie 4 modele:
1. BBox detection (YOLOv8) - wykrywa psy na obrazie
2. Breed classification (EfficientNet-B4) - klasyfikuje rasę
3. Keypoints detection (SimpleBaseline) - wykrywa punkty kluczowe
4. Emotion classification (EfficientNet-B0) - klasyfikuje emocję
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from packages.models import (
    BBoxConfig,
    BBoxModel,
    Detection,
    BreedConfig,
    BreedModel,
    BreedPrediction,
    KeypointsConfig,
    KeypointsModel,
    KeypointsPrediction,
    EmotionPrediction,
    classify_emotion_from_keypoints,
)


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

        # 4. Emotion classification (Rule-based only, NO ML)
        print("\n[4/4] Konfiguracja klasyfikacji emocji...")
        print("  → Używam RULE-BASED classification (DogFACS)")
        print("  → Nie wymaga wytrenowanego modelu!")

        self._models_loaded = True
        print("\n" + "=" * 60)
        print("PIPELINE ZAŁADOWANY")
        print("=" * 60)

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
        detections = self.bbox_model.filter_dogs_only(image)

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

            # Detekcja keypoints
            if self.keypoints_model is not None:
                try:
                    kp_pred = self.keypoints_model.predict(cropped)
                    annotation.keypoints = kp_pred
                except Exception as e:
                    print(f"  ! Błąd detekcji keypoints: {e}")

            # Klasyfikacja emocji (rule-based: keypoints → AU → emotion)
            if annotation.keypoints is not None:
                try:
                    keypoints_flat = annotation.keypoints.to_coco_format()
                    import numpy as np
                    emotion_pred = classify_emotion_from_keypoints(
                        np.array(keypoints_flat, dtype=np.float32)
                    )
                    annotation.emotion = emotion_pred
                except Exception as e:
                    print(f"  ! Błąd klasyfikacji emocji: {e}")

            result.annotations.append(annotation)

        return result

    def process_video_for_dataset(
        self,
        frames_list: list[np.ndarray],
        num_peaks: int = 10,
        neutral_idx: Optional[int] = None,
        min_separation_frames: int = 30,
    ) -> dict:
        """
        Przetwarza sekwencję wideo do generowania datasetu.

        Pipeline dla dataset generation:
        1. Wykryj keypoints dla wszystkich klatek
        2. Auto-detekcja neutral frame (lub użyj podanego)
        3. Oblicz delta AU dla wszystkich klatek
        4. Wybierz peak frames (wysoka TFM + separacja czasowa)
        5. Klasyfikuj emocje dla peak frames

        Args:
            frames_list: Lista klatek wideo jako numpy arrays
            num_peaks: Liczba peak frames do wybrania
            neutral_idx: Opcjonalny indeks neutral frame (auto-detect jeśli None)
            min_separation_frames: Minimalna separacja czasowa między peaks

        Returns:
            Słownik z wynikami:
            {
                "neutral_frame_idx": int,
                "neutral_keypoints": np.ndarray,
                "peak_frames": [
                    {
                        "frame_idx": int,
                        "frame": np.ndarray,
                        "keypoints": np.ndarray,
                        "delta_aus": dict[str, DeltaActionUnit],
                        "emotion": EmotionPrediction,
                        "tfm_score": float,
                    },
                    ...
                ],
                "all_frames_data": [
                    {
                        "frame_idx": int,
                        "keypoints": Optional[np.ndarray],
                        "head_pose": Optional[HeadPose],
                        "delta_aus": Optional[dict],
                    },
                    ...
                ],
            }

        Raises:
            RuntimeError: Gdy modele nie zostały załadowane
            ValueError: Gdy brak keypoints do przetworzenia
        """
        if not self._models_loaded:
            raise RuntimeError("Pipeline nie załadowany. Wywołaj load() najpierw.")

        from packages.pipeline.neutral_frame import (
            NeutralFrameDetector,
            estimate_head_pose,
        )
        from packages.models.delta_action_units import DeltaActionUnitsExtractor
        from packages.pipeline.peak_selector import PeakFrameSelector, compute_tfm
        from packages.models.emotion import classify_emotion_from_delta_aus

        print("\n" + "=" * 60)
        print("DATASET GENERATION PIPELINE")
        print("=" * 60)

        # Step 1: Wykryj keypoints dla wszystkich klatek
        print(f"\n[1/5] Wykrywanie keypoints dla {len(frames_list)} klatek...")
        keypoints_list = []
        valid_frame_indices = []

        for i, frame in enumerate(frames_list):
            # Wykryj psa
            detections = self.bbox_model.filter_dogs_only(frame)

            if not detections:
                keypoints_list.append(None)
                continue

            # Weź pierwszego psa (największy bbox)
            detection = detections[0]
            x, y, w, h = detection.bbox

            # Crop
            height, width = frame.shape[:2]
            x = max(0, x)
            y = max(0, y)
            w = min(w, width - x)
            h = min(h, height - y)
            cropped = frame[y : y + h, x : x + w]

            if cropped.size == 0:
                keypoints_list.append(None)
                continue

            # Wykryj keypoints
            try:
                kp_pred = self.keypoints_model.predict(cropped)
                keypoints_flat = np.array(kp_pred.to_coco_format(), dtype=np.float32)

                # Transformuj współrzędne z cropa na pełny obraz
                # Format: [x0, y0, v0, x1, y1, v1, ...]
                # Dodaj offset bbox do wszystkich x i y
                keypoints_flat[0::3] += x  # Wszystkie x-y: indeksy 0, 3, 6, ...
                keypoints_flat[1::3] += y  # Wszystkie y-ki: indeksy 1, 4, 7, ...

                keypoints_list.append(keypoints_flat)
                valid_frame_indices.append(i)
            except Exception as e:
                print(f"  ! Błąd keypoints dla klatki {i}: {e}")
                keypoints_list.append(None)

        # Filtruj tylko klatki z keypoints
        valid_keypoints = [kp for kp in keypoints_list if kp is not None]

        if len(valid_keypoints) < 2:
            raise ValueError(
                f"Za mało klatek z keypoints: {len(valid_keypoints)}. Potrzeba min 2."
            )

        print(f"  → Wykryto keypoints w {len(valid_keypoints)}/{len(frames_list)} klatkach")

        # Step 2: Estymacja head pose dla wszystkich klatek
        print("\n[2/5] Estymacja head pose...")
        head_poses = []
        for kp in keypoints_list:
            if kp is not None:
                head_pose = estimate_head_pose(kp)
                head_poses.append(head_pose)
            else:
                head_poses.append(None)

        valid_head_poses = [hp for hp in head_poses if hp is not None]
        frontal_count = sum(1 for hp in valid_head_poses if hp.is_frontal)
        print(f"  → Frontal poses: {frontal_count}/{len(valid_head_poses)}")

        # Step 3: Auto-detekcja neutral frame (jeśli nie podano)
        print("\n[3/5] Detekcja neutral frame...")
        if neutral_idx is None:
            detector = NeutralFrameDetector()
            neutral_idx = detector.detect_auto(
                frames=[frames_list[i] for i in valid_frame_indices],
                keypoints_list=valid_keypoints,
                head_poses=valid_head_poses,
                debug=False,  # Ustaw True dla szczegółowych logów
            )
            # Zmapuj z powrotem na oryginalne indeksy
            neutral_idx = valid_frame_indices[neutral_idx]
            print(f"  → Auto-detected neutral frame: {neutral_idx}")
        else:
            print(f"  → Using manual neutral frame: {neutral_idx}")

        neutral_keypoints = keypoints_list[neutral_idx]

        if neutral_keypoints is None:
            raise ValueError(f"Neutral frame {neutral_idx} nie ma keypoints!")

        # Step 4: Oblicz delta AU dla wszystkich klatek
        print("\n[4/5] Obliczanie delta Action Units...")
        delta_extractor = DeltaActionUnitsExtractor(neutral_keypoints)

        delta_aus_list = []
        for kp in keypoints_list:
            if kp is not None:
                delta_aus = delta_extractor.extract(kp)
                delta_aus_list.append(delta_aus)
            else:
                delta_aus_list.append(None)

        valid_delta_aus = [d for d in delta_aus_list if d is not None]
        print(f"  → Obliczono delta AU dla {len(valid_delta_aus)} klatek")

        # Step 5: Wybierz peak frames
        print(f"\n[5/5] Wybór {num_peaks} peak frames (TFM-based)...")
        selector = PeakFrameSelector(
            min_separation_frames=min_separation_frames,
            frontal_only=False,  # Nie wymagamy ścisłego frontal (zbyt restrykcyjne)
            min_keypoint_conf=0.5,  # Zmniejszony próg
            max_head_angle=40.0,  # Maksymalny kąt yaw/pitch
        )

        peak_indices = selector.select(
            frames=frames_list,
            keypoints_list=keypoints_list,
            neutral_idx=neutral_idx,
            delta_aus_list=delta_aus_list,
            head_poses=head_poses,
            num_peaks=num_peaks,
        )

        print(f"  → Wybrano {len(peak_indices)} peak frames")

        # Step 6: Klasyfikuj emocje dla peak frames
        print("\n[6/6] Klasyfikacja emocji dla peak frames...")
        peak_frames_data = []

        for peak_idx in peak_indices:
            delta_aus = delta_aus_list[peak_idx]
            tfm_score = compute_tfm(delta_aus)

            # Klasyfikuj emocję
            emotion_pred = classify_emotion_from_delta_aus(delta_aus)

            peak_frames_data.append(
                {
                    "frame_idx": peak_idx,
                    "frame": frames_list[peak_idx],
                    "keypoints": keypoints_list[peak_idx],
                    "delta_aus": delta_aus,
                    "emotion": emotion_pred,
                    "tfm_score": tfm_score,
                }
            )

            print(
                f"  Peak {peak_idx}: {emotion_pred.emotion.upper()} "
                f"(conf={emotion_pred.confidence:.2f}, TFM={tfm_score:.3f})"
            )

        # Zbierz dane wszystkich klatek
        all_frames_data = []
        for i in range(len(frames_list)):
            all_frames_data.append(
                {
                    "frame_idx": i,
                    "keypoints": keypoints_list[i],
                    "head_pose": head_poses[i],
                    "delta_aus": delta_aus_list[i],
                }
            )

        print("\n" + "=" * 60)
        print("DATASET GENERATION COMPLETE")
        print("=" * 60)

        return {
            "neutral_frame_idx": neutral_idx,
            "neutral_keypoints": neutral_keypoints,
            "peak_frames": peak_frames_data,
            "all_frames_data": all_frames_data,
        }

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
        from PIL import Image, ImageDraw, ImageFont

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
