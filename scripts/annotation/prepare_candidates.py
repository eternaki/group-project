#!/usr/bin/env python3
"""
Przygotowuje kandydatow do RECZNEGO wyboru: kadry odrzucone przez bramke.

Bramka jakosci odrzuca dzis 63% kandydatow, a obejrzane probki pokazuja, ze
czesc z nich jest w pelni czytelna — zwlaszcza kadry odsiane na "za duzo
niepewnych keypoints" przy szerokiej mordzie. Zamiast zgadywac prog, oddajemy
decyzje czlowiekowi: ten skrypt wycina mordy z NARYSOWANYMI punktami, zeby
bylo widac, jak model potraktowal kadr, i zapisuje liste do wyboru.

Klatka neutralna NIE jest przedmiotem wyboru i to jest celowe: para powstaje
z klatki szczytowej, a jej baza AU jedzie razem z nia (`neutral_frame_id`).
Wybranie peaku zawsze zabiera wlasciwa klatke neutralna.

Uzycie:
    python -m scripts.annotation.prepare_candidates --min-face 45
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from packages.pipeline.quality_gate import QualityThresholds, assess_frame
from scripts.annotation.cropping import face_box, read_image, write_jpeg
from scripts.annotation.curate_for_review import (
    FRAME_ROLE_NEUTRAL,
    FRAME_ROLE_PEAK,
    REVIEW_MAX_ASYMMETRY,
    REVIEW_MIN_FACE_WIDTH,
)

DEFAULT_DATASET: str = "data/dataset_final/annotations.json"
DEFAULT_OUTPUT: str = "data/kandydaci"

# Zapas wokol punktow przy wycinaniu mordy do podgladu
PREVIEW_MARGIN: float = 0.30

# Bok miniatury w podgladzie
THUMBNAIL_PX: int = 520

# Ponizej tej pewnosci punkt rysujemy inaczej — to wlasnie te punkty decyduja
# o odrzuceniu na "za duzo niepewnych", wiec czlowiek musi je odroznic
WEAK_POINT_CONFIDENCE: float = 0.30


# Promien punktu na GOTOWEJ miniaturze. Rysowanie przed zmniejszeniem dawalo
# kropki wielkosci piksela: wycinek 800 px z promieniem 3 po sprowadzeniu do
# 260 px zostawial punkt niewidoczny golym okiem.
POINT_RADIUS_PX: int = 6


def _draw_keypoints(
    image: np.ndarray,
    keypoints: list[float],
    origin: tuple[int, int],
    scale: tuple[float, float],
) -> None:
    """
    Rysuje punkty na GOTOWEJ miniaturze: pewne zielone, niepewne czerwone.

    Skala jest PARA, osobno dla kazdej osi. Wycinek mordy nie jest kwadratem
    (zmierzone: stosunki od 0.87 do 1.01), a miniatura jest — wiec jeden
    wspolny mnoznik przesuwa punkty w pionie nawet o 15% wysokosci kadru.

    Args:
        image: Miniatura po zmniejszeniu (modyfikowana w miejscu)
        keypoints: Punkty w ukladzie PELNEJ klatki
        origin: Lewy gorny rog wycinka w pelnej klatce
        scale: Mnozniki (poziomy, pionowy) z wycinka na miniature
    """
    points = np.asarray(keypoints, dtype=float).reshape(-1, 3)
    for x, y, confidence in points:
        position = (
            int((x - origin[0]) * scale[0]),
            int((y - origin[1]) * scale[1]),
        )
        if not (0 <= position[0] < image.shape[1] and 0 <= position[1] < image.shape[0]):
            continue
        pewny = confidence > WEAK_POINT_CONFIDENCE
        kolor = (0, 210, 0) if pewny else (0, 0, 255)
        # Obwodka w kontrze — punkt musi byc widoczny i na jasnej, i na ciemnej sierscie
        cv2.circle(image, position, POINT_RADIUS_PX + 2, (255, 255, 255), -1)
        cv2.circle(image, position, POINT_RADIUS_PX, kolor, -1)


def build(dataset: Path, output: Path, min_face: float) -> int:
    """
    Wycina podglady kandydatow i zapisuje ich liste.

    Args:
        dataset: Surowy COCO po anotacji wsadowej
        output: Katalog na miniatury i liste
        min_face: Minimalna szerokosc mordy OBU klatek pary

    Returns:
        Liczba przygotowanych kandydatow
    """
    coco = json.loads(dataset.read_text(encoding="utf-8"))
    images = {image["id"]: image["file_name"] for image in coco["images"]}
    neutrals = {
        annotation["image_id"]: annotation
        for annotation in coco["annotations"]
        if annotation.get("frame_role") == FRAME_ROLE_NEUTRAL
    }
    frames_dir = dataset.parent / "frames"
    loose = QualityThresholds(min_face_width=0.0, max_asymmetry=99.0, max_weak_ratio=1.0)
    strict = QualityThresholds(
        max_asymmetry=REVIEW_MAX_ASYMMETRY, min_face_width=REVIEW_MIN_FACE_WIDTH
    )

    (output / "thumbs").mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    seen: set[str] = set()

    for peak in coco["annotations"]:
        if peak.get("frame_role") != FRAME_ROLE_PEAK:
            continue
        name = images[peak["image_id"]]
        if name in seen:
            continue
        seen.add(name)
        neutral = neutrals.get(peak.get("neutral_frame_id"))
        if neutral is None:
            continue
        if (
            assess_frame(peak.get("keypoints"), strict).is_usable
            and assess_frame(neutral.get("keypoints"), strict).is_usable
        ):
            continue  # ta para juz jest w kolejce

        peak_quality = assess_frame(peak.get("keypoints"), loose)
        neutral_quality = assess_frame(neutral.get("keypoints"), loose)
        if peak_quality.face_width < min_face or neutral_quality.face_width < min_face:
            continue

        thumb = _render(frames_dir / name, peak.get("keypoints"))
        if thumb is None:
            continue
        thumb_name = f"{len(entries):05d}.jpg"
        write_jpeg(output / "thumbs" / thumb_name, thumb, 82)
        entries.append(
            {
                "thumb": thumb_name,
                "peak": name,
                "neutral": images[neutral["image_id"]],
                "face_px": round(peak_quality.face_width),
                "weak": round(getattr(peak_quality, "weak_ratio", 0.0), 2),
                "video": name.rsplit("/", 1)[0],
            }
        )

    entries.sort(key=lambda item: -item["face_px"])
    (output / "candidates.json").write_text(
        json.dumps(entries, ensure_ascii=False), encoding="utf-8"
    )
    return len(entries)


def _render(frame: Path, keypoints: list[float]) -> Optional[np.ndarray]:
    """
    Wycina morde z klatki i rysuje na niej punkty.

    Args:
        frame: Sciezka pelnej klatki
        keypoints: Punkty w ukladzie tej klatki

    Returns:
        Miniatura albo None, gdy klatki nie da sie odczytac
    """
    image = read_image(frame)
    if image is None:
        return None
    height, width = image.shape[:2]
    box = face_box(keypoints, width, height, PREVIEW_MARGIN)
    if box is None:
        return None
    crop = image[box.y0 : box.y1, box.x0 : box.x1].copy()
    if crop.size == 0:
        return None
    thumb = cv2.resize(crop, (THUMBNAIL_PX, THUMBNAIL_PX))
    skala = (
        THUMBNAIL_PX / max(crop.shape[1], 1),
        THUMBNAIL_PX / max(crop.shape[0], 1),
    )
    _draw_keypoints(thumb, keypoints, (box.x0, box.y0), skala)
    return thumb


def main() -> None:
    """Punkt wejscia CLI."""
    parser = argparse.ArgumentParser(description="Przygotowuje kandydatow do recznego wyboru")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--min-face", type=float, default=45.0)
    args = parser.parse_args()

    ile = build(Path(args.dataset), Path(args.output), args.min_face)
    print(f"Przygotowano {ile} kandydatow w {args.output}")


if __name__ == "__main__":
    main()
