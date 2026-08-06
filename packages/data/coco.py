"""
Klasa do eksportu anotacji w formacie COCO.

Format COCO JSON dla projektu Dog FACS z rozszerzeniami
dla breed_id, emotion_id i keypoints.
"""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from packages.data.schemas import KEYPOINT_NAMES, NUM_KEYPOINTS, SKELETON_CONNECTIONS

# Typ pojedynczej wartości w polu au_analysis: nowy format (słownik z wiarygodnością)
# lub stary (samo ratio) — zbiory sprzed wprowadzenia is_active/confidence.
AUValue = Union[float, dict]

# Liczba wartości na jeden punkt w zapisie COCO: x, y, widoczność
KEYPOINT_VALUES_PER_POINT: int = 3

# Długość płaskiego zapisu keypoints (46 punktów DogFLW po 3 wartości)
FLAT_KEYPOINTS_LENGTH: int = NUM_KEYPOINTS * KEYPOINT_VALUES_PER_POINT

# Ratio AU równe 1.0 oznacza brak zmiany względem klatki neutralnej
NEUTRAL_RATIO: float = 1.0

# Rola klatki w obrębie treku: baza AU albo szczyt mimiki
FRAME_ROLE_NEUTRAL: str = "neutral"
FRAME_ROLE_PEAK: str = "peak"
FRAME_ROLES: frozenset[str] = frozenset({FRAME_ROLE_NEUTRAL, FRAME_ROLE_PEAK})

# Źródło etykiety: pre-etykieta z reguł albo weryfikacja ręczna (Sprint 15)
LABEL_SOURCE_AUTO_RULES: str = "auto_rules"
LABEL_SOURCE_HUMAN_VERIFIED: str = "human_verified"
LABEL_SOURCES: frozenset[str] = frozenset(
    {LABEL_SOURCE_AUTO_RULES, LABEL_SOURCE_HUMAN_VERIFIED}
)

# Próg odsiewu aktywacji AU: sygnał musi być co najmniej równy jednemu odchyleniu
# standardowemu szumu zmierzonego w tym treku. Niżej nie da się odróżnić mimiki od
# drgania keypoints — mediana szumu ratio po wygładzaniu wynosi 0.166, a próg
# aktywacji AU to 0.15, więc samo `is_active` zapala się od szumu.
MIN_SIGNAL_TO_NOISE: float = 1.0


def au_ratio(value: AUValue) -> float:
    """
    Zwraca ratio AU niezależnie od formatu zapisu.

    Args:
        value: Wartość z au_analysis — float (stary format) lub słownik z kluczem "ratio"

    Returns:
        Ratio (stosunek cel/neutral)
    """
    if isinstance(value, dict):
        return float(value.get("ratio", 0.0))
    return float(value)


def au_analysis_from_delta_aus(
    delta_aus: dict,
    au_noise: Optional[Mapping[str, float]] = None,
) -> dict[str, dict]:
    """
    Serializuje Action Units do pola au_analysis wraz z wiarygodnością pomiaru.

    Sam ratio nie wystarcza: klamrowana wartość (np. ruchome ucho przy limicie 3.0)
    wygląda jak silna aktywacja, choć pomiar jest niewiarygodny. Nie wystarcza też
    `is_active`: przy błędzie naszego modelu keypoints (NME_iod 0.091) mediana szumu
    ratio PO wygładzeniu wynosi 0.166, a próg aktywacji to 0.15, więc dla typowo 13 z
    21 AU flaga zapala się od drgania punktów. Dlatego przy podanym `au_noise` obok
    ratio jedzie zmierzony szum tego AU w tym treku i stosunek sygnału do szumu.

    Args:
        delta_aus: Słownik {nazwa AU: DeltaActionUnit}
        au_noise: Odchylenie standardowe ratio na trek (`TrackResult.au_noise`).
            AU o zbyt małej liczbie pomiarów NIE ma tam klucza — dostaje wtedy
            `noise=None` i `snr=None`, czyli „nie zmierzono", a nie „szum zerowy".
            Pominięcie argumentu daje dokładnie stary zestaw pól.

    Returns:
        Słownik {nazwa AU: {"ratio", "is_active", "confidence"}}, a przy podanym
        `au_noise` dodatkowo {"noise": float | None, "snr": float | None}
    """
    return {name: _au_entry(name, au, au_noise) for name, au in delta_aus.items()}


def au_signal_above_noise(value: AUValue) -> Optional[bool]:
    """
    Mówi, czy sygnał AU przewyższa zmierzony szum treku.

    Trójstanowa odpowiedź jest celowa: brak zmierzonego szumu (stary zbiór, AU o zbyt
    małej liczbie pomiarów, szum poniżej rozdzielczości) to brak wiedzy, a nie „szum
    zerowy". Zwrócenie w takim wypadku `False` po cichu wyrzuciłoby dobre próbki,
    a `True` — wpuściło do treningu etykiety z drgania punktów.

    Args:
        value: Wartość z au_analysis (nowy format słownikowy albo stary float)

    Returns:
        True/False gdy szum zmierzono, None gdy nie ma jak porównać
    """
    if not isinstance(value, dict):
        return None
    snr = value.get("snr")
    if not isinstance(snr, (int, float)):
        return None
    return bool(snr >= MIN_SIGNAL_TO_NOISE)


def _au_entry(
    name: str,
    au: "DeltaActionUnit",  # type: ignore[name-defined]  # noqa: F821
    au_noise: Optional[Mapping[str, float]],
) -> dict:
    """
    Buduje wpis jednego AU w polu au_analysis.

    Args:
        name: Nazwa AU (klucz w delta_aus — to ona wiąże pomiar ze słownikiem szumu)
        au: Pomiar AU
        au_noise: Szum ratio na trek albo None, gdy szumu nie mierzono

    Returns:
        Słownik pól tego AU
    """
    entry: dict = {
        "ratio": float(au.ratio),
        "is_active": bool(au.is_active),
        "confidence": float(au.confidence),
    }
    if au_noise is None:
        return entry

    noise = au_noise.get(name)
    entry["noise"] = None if noise is None else float(noise)
    entry["snr"] = _signal_to_noise(entry["ratio"], noise)
    return entry


def _signal_to_noise(ratio: float, noise: Optional[float]) -> Optional[float]:
    """
    Liczy stosunek sygnału AU do szumu treku: |ratio - 1| / sigma.

    None oznacza „nie da się policzyć": szumu nie zmierzono albo wyszedł zerowy
    (ratio stałe w całym treku, np. klamrowane). Dzielenie przez zero dałoby
    `Infinity`, czyli niepoprawny JSON i pozorną pewność najgorszego pomiaru.

    Args:
        ratio: Stosunek cel/neutral z pomiaru AU
        noise: Odchylenie standardowe ratio w treku albo None

    Returns:
        Stosunek sygnału do szumu albo None
    """
    if noise is None or not math.isfinite(ratio):
        return None
    noise_value = float(noise)
    if not math.isfinite(noise_value) or noise_value <= 0.0:
        return None
    return abs(ratio - NEUTRAL_RATIO) / noise_value


@dataclass(frozen=True)
class TrackAnnotation:
    """
    Pola anotacji wynikające z przynależności klatki do treku jednego psa.

    Zebrane w jednym obiekcie, bo `add_annotation` ma już kilkanaście parametrów —
    dokładanie pięciu kolejnych rozjechałoby się z limitem z CLAUDE.md. Ten sam
    wzorzec zastosowano dla `TrackQuality` i `VideoDatasetConfig`.

    Attributes:
        track_id: Identyfikator psa w obrębie wideo (`TrackResult.track_id`)
        frame_role: `neutral` (baza AU treku) albo `peak` (szczyt mimiki)
        au_noise: Odchylenie standardowe ratio na trek, per AU. Pusty słownik znaczy
            „nie zmierzono" (np. trek za krótki), a NIE „szum zerowy".
        au_sample_count: Liczba pomiarów, z których policzono szum. Bez niej nie da
            się skorygować obciążenia krótkich treków (sigma z 3 klatek ma ~11%
            obciążenia i CV ~50%), więc jest wymagana razem z `au_noise`.
        label_source: `auto_rules` (pre-etykieta z reguł) albo `human_verified`
            (po weryfikacji ręcznej, Sprint 15)
        neutral_source: Skąd wzięła się klatka neutralna treku (`auto` albo `manual`).
            Przy kilku psach ręcznie wskazana klatka trafia tylko w treki, które ją
            zawierają — bez tego pola trening miesza dwa różne reżimy wyboru bazy AU,
            nie mając jak ich rozdzielić.
        procrustes_keypoints: Kanoniczny kształt twarzy (138 wartości) — planowane
            wejście sieci AU. Punkty niewidoczne są tam przekształcone, ale
            niewiarygodne; konsument ma je odfiltrować po widoczności.
    """

    track_id: int
    frame_role: str
    au_noise: dict[str, float] = field(default_factory=dict)
    au_sample_count: dict[str, int] = field(default_factory=dict)
    label_source: str = LABEL_SOURCE_AUTO_RULES
    neutral_source: Optional[str] = None
    procrustes_keypoints: Optional[list[float]] = None

    def __post_init__(self) -> None:
        """
        Sprawdza spójność pól treku.

        Raises:
            ValueError: Gdy rola, źródło etykiety, komplet szumu albo długość
                kanonicznego kształtu są nieprawidłowe
        """
        if self.frame_role not in FRAME_ROLES:
            raise ValueError(
                f"Nieznana rola klatki: {self.frame_role!r} (dozwolone: {sorted(FRAME_ROLES)})"
            )
        if self.label_source not in LABEL_SOURCES:
            raise ValueError(
                f"Nieznane źródło etykiety: {self.label_source!r} "
                f"(dozwolone: {sorted(LABEL_SOURCES)})"
            )

        missing_counts = sorted(set(self.au_noise) - set(self.au_sample_count))
        if missing_counts:
            raise ValueError(
                f"Szum AU bez liczby prób dla: {missing_counts}. Sigma z kilku klatek "
                "jest obciążona, więc bez liczby prób nie da się jej skorygować."
            )

        if (
            self.procrustes_keypoints is not None
            and len(self.procrustes_keypoints) != FLAT_KEYPOINTS_LENGTH
        ):
            raise ValueError(
                f"Kanoniczny kształt musi mieć {FLAT_KEYPOINTS_LENGTH} wartości, "
                f"otrzymano {len(self.procrustes_keypoints)}"
            )

    def to_dict(self) -> dict:
        """
        Zwraca pola treku do dopisania w anotacji COCO.

        `au_noise` i `au_sample_count` zapisujemy zawsze — również puste. Brak pola
        i pusty słownik znaczyłyby to samo dla czytnika, ale pusty słownik jawnie
        mówi, że trek przez pomiar szumu przeszedł.

        Returns:
            Słownik pól COCO
        """
        fields: dict = {
            "track_id": self.track_id,
            "frame_role": self.frame_role,
            "label_source": self.label_source,
            "au_noise": dict(self.au_noise),
            "au_sample_count": dict(self.au_sample_count),
        }
        if self.neutral_source is not None:
            fields["neutral_source"] = self.neutral_source
        if self.procrustes_keypoints is not None:
            fields["procrustes_keypoints"] = list(self.procrustes_keypoints)
        return fields


@dataclass
class COCOInfo:
    """Sekcja 'info' w COCO JSON."""

    description: str = "Dog FACS Dataset"
    url: str = "https://github.com/pg-weti/dog-facs"
    version: str = "1.0"
    year: int = field(default_factory=lambda: datetime.now().year)
    contributor: str = "Politechnika Gdańska WETI"
    date_created: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d")
    )

    def to_dict(self) -> dict:
        """Konwertuje do słownika."""
        return {
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "year": self.year,
            "contributor": self.contributor,
            "date_created": self.date_created,
        }


@dataclass
class COCOCategory:
    """Kategoria w COCO."""

    id: int
    name: str
    supercategory: str = "animal"
    keypoints: list[str] = field(default_factory=list)
    skeleton: list[tuple[int, int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Konwertuje do słownika."""
        return {
            "id": self.id,
            "name": self.name,
            "supercategory": self.supercategory,
            "keypoints": self.keypoints,
            "skeleton": [[a, b] for a, b in self.skeleton],
        }


class COCODataset:
    """
    Klasa do tworzenia i eksportu datasetu w formacie COCO.

    Użycie:
        dataset = COCODataset()

        # Dodaj obraz
        image_id = dataset.add_image(
            file_name="video1_frame_000001.jpg",
            width=1920,
            height=1080,
            source_video="video1.mp4",
            frame_number=0,
        )

        # Dodaj anotację
        dataset.add_annotation(
            image_id=image_id,
            bbox=[100, 150, 200, 250],
            keypoints=[x1, y1, v1, x2, y2, v2, ...],
            breed_id=15,
            emotion_id=0,
            confidence={"bbox": 0.95, "breed": 0.87, "emotion": 0.72},
        )

        # Zapisz do JSON
        dataset.save("annotations.json")
    """

    def __init__(self, info: Optional[COCOInfo] = None) -> None:
        """
        Inicjalizuje dataset.

        Args:
            info: Informacje o datasecie (opcjonalnie)
        """
        self.info = info or COCOInfo()

        # Wewnętrzne struktury
        self._images: list[dict] = []
        self._annotations: list[dict] = []
        self._licenses: list[dict] = []

        # Liczniki ID
        self._next_image_id = 1
        self._next_annotation_id = 1

        # Domyślna kategoria: dog
        self._categories = [
            COCOCategory(
                id=1,
                name="dog",
                supercategory="animal",
                keypoints=KEYPOINT_NAMES,
                skeleton=list(SKELETON_CONNECTIONS),
            )
        ]

    @property
    def num_images(self) -> int:
        """Zwraca liczbę obrazów."""
        return len(self._images)

    @property
    def num_annotations(self) -> int:
        """Zwraca liczbę anotacji."""
        return len(self._annotations)

    def add_image(
        self,
        file_name: str,
        width: int,
        height: int,
        source_video: Optional[str] = None,
        frame_number: Optional[int] = None,
        **extra_fields,
    ) -> int:
        """
        Dodaje obraz do datasetu.

        Args:
            file_name: Nazwa pliku obrazu
            width: Szerokość w pikselach
            height: Wysokość w pikselach
            source_video: Nazwa źródłowego wideo (opcjonalnie)
            frame_number: Numer klatki w wideo (opcjonalnie)
            **extra_fields: Dodatkowe pola

        Returns:
            ID dodanego obrazu
        """
        image_id = self._next_image_id
        self._next_image_id += 1

        image_dict = {
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
        }

        if source_video:
            image_dict["source_video"] = source_video
        if frame_number is not None:
            image_dict["frame_number"] = frame_number

        image_dict.update(extra_fields)
        self._images.append(image_dict)

        return image_id

    def add_annotation(
        self,
        image_id: int,
        bbox: list[int] | tuple[int, int, int, int],
        keypoints: Optional[list[float]] = None,
        num_keypoints: Optional[int] = None,
        breed_id: Optional[int] = None,
        breed_name: Optional[str] = None,
        emotion_id: Optional[int] = None,
        emotion_name: Optional[str] = None,
        confidence: Optional[dict[str, float]] = None,
        iscrowd: int = 0,
        au_analysis: Optional[dict[str, AUValue]] = None,
        neutral_frame_id: Optional[int] = None,
        emotion_rule_applied: Optional[str] = None,
        track: Optional[TrackAnnotation] = None,
        **extra_fields,
    ) -> int:
        """
        Dodaje anotację do datasetu.

        Args:
            image_id: ID obrazu
            bbox: Bounding box [x, y, width, height]
            keypoints: Lista keypoints [x1, y1, v1, x2, y2, v2, ...]
            num_keypoints: Liczba widocznych keypoints
            breed_id: ID rasy
            breed_name: Nazwa rasy
            emotion_id: ID emocji
            emotion_name: Nazwa emocji
            confidence: Słownik z pewnością {"bbox": 0.95, ...}
            iscrowd: Czy to tłum (domyślnie 0)
            au_analysis: Analiza Action Units
                {"AU101": {"ratio": 1.15, "is_active": True, "confidence": 0.9}, ...}
                (stary format — samo ratio jako float — jest nadal odczytywany)
            neutral_frame_id: ID neutral frame (reference do image_id)
            emotion_rule_applied: Nazwa zastosowanej reguły emocji
            track: Pola treku psa (`TrackAnnotation`) — który pies, rola klatki,
                źródło etykiety, zmierzony szum AU i kanoniczny kształt.
                Anotacje spoza pipeline'u wideo zostają bez tych pól.
            **extra_fields: Dodatkowe pola

        Returns:
            ID dodanej anotacji
        """
        annotation_id = self._next_annotation_id
        self._next_annotation_id += 1

        bbox_list = list(bbox)
        x, y, w, h = bbox_list
        area = w * h

        ann_dict = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,  # dog
            "bbox": bbox_list,
            "area": area,
            "iscrowd": iscrowd,
        }

        # Keypoints
        if keypoints:
            ann_dict["keypoints"] = keypoints
            if num_keypoints is not None:
                ann_dict["num_keypoints"] = num_keypoints
            else:
                # Policz widoczne keypoints (visibility > 0)
                visible = sum(
                    1 for i in range(2, len(keypoints), 3) if keypoints[i] > 0
                )
                ann_dict["num_keypoints"] = visible

        # Breed
        if breed_id is not None:
            ann_dict["breed_id"] = breed_id
        if breed_name:
            ann_dict["breed"] = breed_name

        # Emotion
        if emotion_id is not None:
            ann_dict["emotion_id"] = emotion_id
        if emotion_name:
            ann_dict["emotion"] = emotion_name

        # Confidence
        if confidence:
            ann_dict["confidence"] = confidence

        # DogFACS Dataset Generator extensions
        if au_analysis:
            ann_dict["au_analysis"] = au_analysis
        if neutral_frame_id is not None:
            ann_dict["neutral_frame_id"] = neutral_frame_id
        if emotion_rule_applied:
            ann_dict["emotion_rule_applied"] = emotion_rule_applied

        # Pola treku psa — potrzebne do trenowania własnej sieci AU (Sprint 16)
        if track is not None:
            ann_dict.update(track.to_dict())

        ann_dict.update(extra_fields)
        self._annotations.append(ann_dict)

        return annotation_id

    def add_annotation_from_dog(
        self,
        image_id: int,
        dog_annotation: "DogAnnotation",  # type: ignore[name-defined]  # noqa: F821
    ) -> int:
        """
        Dodaje anotację z obiektu DogAnnotation.

        Args:
            image_id: ID obrazu
            dog_annotation: Obiekt DogAnnotation z pipeline

        Returns:
            ID dodanej anotacji
        """
        # Keypoints
        keypoints = None
        num_keypoints = None
        if dog_annotation.keypoints:
            keypoints = dog_annotation.keypoints.to_coco_format()
            num_keypoints = dog_annotation.keypoints.num_detected

        # Breed
        breed_id = None
        breed_name = None
        if dog_annotation.breed:
            breed_id = dog_annotation.breed.class_id
            breed_name = dog_annotation.breed.class_name

        # Emotion
        emotion_id = None
        emotion_name = None
        emotion_rule_applied = None
        au_analysis = None
        if dog_annotation.emotion:
            emotion_id = dog_annotation.emotion.emotion_id
            emotion_name = dog_annotation.emotion.emotion
            # DogFACS extensions
            if dog_annotation.emotion.rule_applied:
                emotion_rule_applied = dog_annotation.emotion.rule_applied
            if dog_annotation.emotion.action_units:
                # Extract delta values for AU analysis
                au_analysis = dog_annotation.emotion.action_units.copy()

        # Confidence
        confidence = {"bbox": dog_annotation.bbox_confidence}
        if dog_annotation.breed:
            confidence["breed"] = dog_annotation.breed.confidence
        if dog_annotation.keypoints:
            confidence["keypoints"] = dog_annotation.keypoints.confidence
        if dog_annotation.emotion:
            confidence["emotion"] = dog_annotation.emotion.confidence

        return self.add_annotation(
            image_id=image_id,
            bbox=list(dog_annotation.bbox),
            keypoints=keypoints,
            num_keypoints=num_keypoints,
            breed_id=breed_id,
            breed_name=breed_name,
            emotion_id=emotion_id,
            emotion_name=emotion_name,
            confidence=confidence,
            au_analysis=au_analysis,
            emotion_rule_applied=emotion_rule_applied,
        )

    def to_dict(self) -> dict:
        """
        Konwertuje dataset do słownika COCO.

        Returns:
            Słownik w formacie COCO
        """
        return {
            "info": self.info.to_dict(),
            "licenses": self._licenses,
            "images": self._images,
            "annotations": self._annotations,
            "categories": [cat.to_dict() for cat in self._categories],
        }

    def save(self, output_path: Path | str, indent: int = 2) -> None:
        """
        Zapisuje dataset do pliku JSON.

        Args:
            output_path: Ścieżka do pliku wyjściowego
            indent: Wcięcie JSON (domyślnie 2)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            # allow_nan=False: NaN/Infinity to NIEPOPRAWNY JSON — część parserów
            # odrzuci taki plik, a cichy zapis oznaczałby zbiór nie do wczytania.
            # Moduł broni się przed NaN przy liczeniu szumu; to ostatnia bramka.
            json.dump(
                self.to_dict(), f, indent=indent, ensure_ascii=False, allow_nan=False
            )

        print(f"Dataset zapisany: {output_path}")
        print(f"  Obrazów: {self.num_images}")
        print(f"  Anotacji: {self.num_annotations}")

    @classmethod
    def load(cls, json_path: Path | str) -> "COCODataset":
        """
        Wczytuje dataset z pliku JSON.

        Args:
            json_path: Ścieżka do pliku JSON

        Returns:
            Obiekt COCODataset
        """
        json_path = Path(json_path)

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        dataset = cls()

        # Info
        if "info" in data:
            info_data = data["info"]
            dataset.info = COCOInfo(
                description=info_data.get("description", ""),
                url=info_data.get("url", ""),
                version=info_data.get("version", "1.0"),
                year=info_data.get("year", datetime.now().year),
                contributor=info_data.get("contributor", ""),
                date_created=info_data.get("date_created", ""),
            )

        # Images
        dataset._images = data.get("images", [])
        if dataset._images:
            dataset._next_image_id = max(img["id"] for img in dataset._images) + 1

        # Annotations
        dataset._annotations = data.get("annotations", [])
        if dataset._annotations:
            dataset._next_annotation_id = (
                max(ann["id"] for ann in dataset._annotations) + 1
            )

        # Licenses
        dataset._licenses = data.get("licenses", [])

        return dataset

    def validate(self) -> dict:
        """
        Waliduje dataset.

        Returns:
            Słownik z wynikami walidacji
        """
        errors = []
        warnings = []

        # Sprawdź czy są obrazy
        if not self._images:
            errors.append("Brak obrazów w datasecie")

        # Sprawdź czy są anotacje
        if not self._annotations:
            warnings.append("Brak anotacji w datasecie")

        # Sprawdź unikalne ID obrazów
        image_ids = [img["id"] for img in self._images]
        if len(image_ids) != len(set(image_ids)):
            errors.append("Duplikaty ID obrazów")

        # Sprawdź unikalne ID anotacji
        ann_ids = [ann["id"] for ann in self._annotations]
        if len(ann_ids) != len(set(ann_ids)):
            errors.append("Duplikaty ID anotacji")

        # Sprawdź czy anotacje odnoszą się do istniejących obrazów
        image_id_set = set(image_ids)
        for ann in self._annotations:
            if ann["image_id"] not in image_id_set:
                errors.append(
                    f"Anotacja {ann['id']} odnosi się do "
                    f"nieistniejącego obrazu {ann['image_id']}"
                )

        # Sprawdź bounding boxy
        for ann in self._annotations:
            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                errors.append(f"Anotacja {ann['id']} ma nieprawidłowy bbox")
            elif bbox[2] <= 0 or bbox[3] <= 0:
                warnings.append(f"Anotacja {ann['id']} ma bbox o zerowej wielkości")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "stats": {
                "images": len(self._images),
                "annotations": len(self._annotations),
            },
        }

    def get_statistics(self) -> dict:
        """
        Oblicza statystyki datasetu.

        Returns:
            Słownik ze statystykami
        """
        stats = {
            "total_images": len(self._images),
            "total_annotations": len(self._annotations),
            "annotations_per_image": {},
            "breeds": {},
            "emotions": {},
            "emotion_rules": {},
            "action_units": {},
        }

        # Anotacje per obraz
        ann_counts = {}
        for ann in self._annotations:
            img_id = ann["image_id"]
            ann_counts[img_id] = ann_counts.get(img_id, 0) + 1

        if ann_counts:
            stats["annotations_per_image"] = {
                "min": min(ann_counts.values()),
                "max": max(ann_counts.values()),
                "avg": sum(ann_counts.values()) / len(ann_counts),
            }

        # Rasy
        for ann in self._annotations:
            breed = ann.get("breed", "unknown")
            stats["breeds"][breed] = stats["breeds"].get(breed, 0) + 1

        # Emocje
        for ann in self._annotations:
            emotion = ann.get("emotion", "unknown")
            stats["emotions"][emotion] = stats["emotions"].get(emotion, 0) + 1

        # Emotion rules (DogFACS extension)
        for ann in self._annotations:
            rule = ann.get("emotion_rule_applied")
            if rule:
                stats["emotion_rules"][rule] = stats["emotion_rules"].get(rule, 0) + 1

        # Action Units (DogFACS extension)
        for ann in self._annotations:
            au_analysis = ann.get("au_analysis", {})
            for au_name, au_value in au_analysis.items():
                if au_name not in stats["action_units"]:
                    stats["action_units"][au_name] = {
                        "count": 0,
                        "total_delta": 0.0,
                        "avg_delta": 0.0,
                    }
                stats["action_units"][au_name]["count"] += 1
                stats["action_units"][au_name]["total_delta"] += au_ratio(au_value)

        # Oblicz średnie dla AU
        for au_name in stats["action_units"]:
            count = stats["action_units"][au_name]["count"]
            total = stats["action_units"][au_name]["total_delta"]
            stats["action_units"][au_name]["avg_delta"] = total / count if count > 0 else 0.0

        return stats
