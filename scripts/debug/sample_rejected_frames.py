#!/usr/bin/env python3
"""
Podgląd materiału odrzuconego przez bramkę jakości — po grupach, z keypoints.

Bramka podaje liczby ("1884 profil lub obrót głowy"), ale liczba nie mówi, czy
odrzucony kadr był naprawdę nie do odczytania, czy tylko lekko przekręcony.
Ten skrypt renderuje PRÓBKĘ każdej grupy z narysowanymi punktami, żeby dało się
ocenić stratę okiem.

Próbka jest brana RÓWNOMIERNIE po całym zakresie tkliwości grupy, nie losowo
i nie od najlepszych — inaczej podgląd pokazywałby, że wszystko jest w porządku,
niezależnie od tego, jak jest naprawdę.

Użycie:
    python -m scripts.debug.sample_rejected_frames --per-group 60
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from packages.pipeline.quality_gate import (
    QualityThresholds,
    assess_frame,
    hide_out_of_frame,
    split_keypoints,
)
from scripts.annotation.curate_for_review import (
    REVIEW_MAX_ASYMMETRY,
    REVIEW_MIN_FACE_WIDTH,
)

DEFAULT_DATASET: str = "data/dataset_v2"
DEFAULT_OUTPUT: str = "data/review_samples"

# Kadr mordy powiększamy o ten margines — anotator też ogląda z zapasem.
CROP_MARGIN: float = 0.18
THUMBNAIL_WIDTH: int = 260
JPEG_QUALITY: int = 72

# Punkt pewny / niepewny / ukryty
_COLOR_STRONG: tuple[int, int, int] = (0, 220, 60)
_COLOR_WEAK: tuple[int, int, int] = (0, 165, 255)
_CONFIDENT: float = 0.5
_VISIBLE: float = 0.3


@dataclass(frozen=True)
class Sample:
    """
    Jedna klatka wybrana do podglądu.

    Attributes:
        path: Ścieżka pliku klatki względem katalogu `frames`
        group: Nazwa grupy (powód odrzucenia albo "przyjęte")
        severity: Wartość, po której grupa jest sortowana
        caption: Podpis pod miniaturą
    """

    path: str
    group: str
    severity: float
    caption: str


def _crop_box(coords: np.ndarray, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """
    Wylicza kadr wokół widocznych punktów.

    Args:
        coords: Współrzędne keypoints (N, 2)
        shape: (wysokość, szerokość) obrazu

    Returns:
        Krotka (x0, y0, x1, y1) przyciętą do obrazu
    """
    height, width = shape
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    margin_x = (x_max - x_min) * CROP_MARGIN
    margin_y = (y_max - y_min) * CROP_MARGIN
    return (
        int(max(0, x_min - margin_x)),
        int(max(0, y_min - margin_y)),
        int(min(width, x_max + margin_x)),
        int(min(height, y_max + margin_y)),
    )


def render(frames_root: Path, sample: Sample, keypoints: list[float]) -> Optional[np.ndarray]:
    """
    Rysuje keypoints na kadrze mordy i skaluje do miniatury.

    Args:
        frames_root: Katalog z klatkami
        sample: Opis wybranej klatki
        keypoints: Lista [x, y, conf] * N

    Returns:
        Obraz BGR albo None, gdy pliku nie ma
    """
    image = cv2.imread(str(frames_root / sample.path))
    if image is None:
        return None

    coords, confidences = split_keypoints(keypoints)
    visible = confidences >= _VISIBLE
    if not visible.any():
        return None

    x0, y0, x1, y1 = _crop_box(coords[visible], image.shape[:2])
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None

    canvas = image[y0:y1, x0:x1].copy()
    scale = THUMBNAIL_WIDTH / canvas.shape[1]
    canvas = cv2.resize(canvas, (THUMBNAIL_WIDTH, max(1, int(canvas.shape[0] * scale))))

    for (x, y), conf in zip(coords, confidences):
        if conf < _VISIBLE:
            continue
        point = (int((x - x0) * scale), int((y - y0) * scale))
        color = _COLOR_STRONG if conf >= _CONFIDENT else _COLOR_WEAK
        cv2.circle(canvas, point, 2, color, -1, lineType=cv2.LINE_AA)
    return canvas


def _group_of(quality, accepted: bool) -> tuple[str, float]:
    """
    Przypisuje klatkę do grupy i podaje jej tkliwość.

    Args:
        quality: Ocena z `assess_frame`
        accepted: Czy klatka przeszła kurację

    Returns:
        Krotka (nazwa grupy, tkliwość do sortowania)
    """
    if accepted:
        return "przyjete", quality.asymmetry
    if len(quality.reasons) > 1:
        return "wiele_powodow", quality.asymmetry
    reason = quality.reasons[0] if quality.reasons else "inne"
    if "profil" in reason:
        return "obrot_glowy", quality.asymmetry
    if "niepewnych" in reason:
        return "slabe_punkty", quality.weak_ratio
    if "morda" in reason:
        return "mala_morda", -quality.face_width
    # Bez powodów, a jednak poza kuracją: sam kadr przeszedł, ale para odpadła,
    # bo w treku nie było godnej klatki neutralnej. Strata nie leży w tym kadrze.
    return "zla_neutralna", quality.asymmetry


def _evenly(items: list, count: int) -> list:
    """
    Wybiera `count` elementów rozłożonych po całej liście.

    Args:
        items: Lista posortowana po tkliwości
        count: Ile elementów zwrócić

    Returns:
        Podlista rozłożona równomiernie
    """
    if len(items) <= count:
        return items
    step = len(items) / count
    return [items[int(index * step)] for index in range(count)]


def collect(dataset: Path) -> tuple[dict[str, list[tuple[Sample, list[float]]]], dict]:
    """
    Dzieli peaki zbioru na grupy według powodu odrzucenia.

    Args:
        dataset: Katalog zbioru z `annotations.json` i `curated.json`

    Returns:
        Krotka (grupy, statystyki liczbowe)
    """
    raw = json.loads((dataset / "annotations.json").read_text(encoding="utf-8"))
    curated = json.loads((dataset / "curated.json").read_text(encoding="utf-8"))

    raw_images = {img["id"]: img for img in raw["images"]}
    curated_images = {img["id"]: img for img in curated["images"]}
    kept = {curated_images[a["image_id"]]["file_name"] for a in curated["annotations"]}

    thresholds = QualityThresholds(
        max_asymmetry=REVIEW_MAX_ASYMMETRY,
        min_face_width=REVIEW_MIN_FACE_WIDTH,
    )

    groups: dict[str, list[tuple[Sample, list[float]]]] = {}
    totals: dict[str, int] = {}
    for annotation in raw["annotations"]:
        if annotation.get("frame_role") != "peak":
            continue
        keypoints = annotation.get("keypoints")
        if not keypoints:
            continue
        image = raw_images[annotation["image_id"]]
        file_name = image["file_name"]
        cleaned = hide_out_of_frame(
            keypoints,
            image_size=(image.get("width"), image.get("height")),
            bbox=annotation.get("bbox"),
        )
        quality = assess_frame(cleaned, thresholds)
        group, severity = _group_of(quality, file_name in kept)
        caption = (
            f"asym {quality.asymmetry:.2f} | slabe {quality.weak_ratio:.0%} "
            f"| morda {quality.face_width:.0f}px"
        )
        totals[group] = totals.get(group, 0) + 1
        groups.setdefault(group, []).append(
            (Sample(file_name, group, severity, caption), cleaned)
        )

    for bucket in groups.values():
        bucket.sort(key=lambda item: item[0].severity)
    return groups, totals


def main() -> None:
    """Punkt wejścia: renderuje próbki i zapisuje je na dysk."""
    parser = argparse.ArgumentParser(description="Podglad odrzuconych klatek")
    parser.add_argument("--dataset", type=Path, default=Path(DEFAULT_DATASET))
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--per-group", type=int, default=60)
    args = parser.parse_args()

    groups, totals = collect(args.dataset)
    frames_root = args.dataset / "frames"
    args.out.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, list[dict]] = {}
    for group, bucket in sorted(groups.items()):
        chosen = _evenly(bucket, args.per_group)
        target = args.out / group
        target.mkdir(exist_ok=True)
        entries: list[dict] = []
        for index, (sample, keypoints) in enumerate(chosen):
            canvas = render(frames_root, sample, keypoints)
            if canvas is None:
                continue
            name = f"{index:03d}.jpg"
            cv2.imwrite(
                str(target / name), canvas, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            entries.append(
                {"file": f"{group}/{name}", "caption": sample.caption, "src": sample.path}
            )
        manifest[group] = entries
        print(f"{group:16} w zbiorze {totals.get(group, 0):5}  w probce {len(entries):3}")

    (args.out / "manifest.json").write_text(
        json.dumps({"totals": totals, "groups": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nZapisano do {args.out}")


if __name__ == "__main__":
    main()
