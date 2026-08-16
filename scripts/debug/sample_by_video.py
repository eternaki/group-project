#!/usr/bin/env python3
"""
Podgląd zbioru NAGRANIAMI: ile klatek dało jedno nagranie i ile z nich przeżyło.

Widok po grupach odrzucenia mówi, co jest nie tak z pojedynczą klatką. Nie mówi
natomiast, czego szukać przy zbieraniu materiału — a to osobne pytanie, bo
klatki nie są niezależne: jedno dobre nagranie daje ich kilkanaście, a nagranie
z psem w profilu nie da żadnej, choćby było w 4K.

Skrypt renderuje WSZYSTKIE peaki wybranych nagrań (przyjęte i odrzucone razem),
żeby dało się zobaczyć, czym różni się nagranie wydajne od jałowego.

Użycie:
    python -m scripts.debug.sample_by_video --videos 36
"""

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from packages.pipeline.quality_gate import (
    QualityThresholds,
    assess_frame,
    hide_out_of_frame,
)
from scripts.annotation.curate_for_review import (
    REVIEW_MAX_ASYMMETRY,
    REVIEW_MIN_FACE_WIDTH,
)
from scripts.debug.sample_rejected_frames import (
    JPEG_QUALITY,
    Sample,
    _group_of,
    render,
)

DEFAULT_DATASET: str = "data/dataset_v2"
DEFAULT_OUTPUT: str = "data/review_by_video"

# Ile najbardziej i najmniej wydajnych nagrań pokazać z miniaturami. Reszta
# wchodzi tylko do statystyki — 1181 nagrań nie da się obejrzeć.
SHOWN_TOP: int = 18
SHOWN_BOTTOM: int = 18


@dataclass
class VideoStats:
    """
    Bilans jednego nagrania.

    Attributes:
        video: Nazwa katalogu nagrania
        frames: Lista (Sample, keypoints, czy przyjęte)
        reasons: Licznik powodów odrzucenia
    """

    video: str
    frames: list = field(default_factory=list)
    reasons: Counter = field(default_factory=Counter)

    @property
    def total(self) -> int:
        """Ile peaków wykrył pipeline w tym nagraniu."""
        return len(self.frames)

    @property
    def kept(self) -> int:
        """Ile peaków przeszło do kolejki anotatora."""
        return sum(1 for item in self.frames if item[2])

    @property
    def yield_ratio(self) -> float:
        """Udział peaków przyjętych — miara wydajności nagrania."""
        return self.kept / self.total if self.total else 0.0


def collect(dataset: Path) -> dict[str, VideoStats]:
    """
    Grupuje peaki zbioru po nagraniu i liczy bilans każdego.

    Args:
        dataset: Katalog z `annotations.json` i `curated.json`

    Returns:
        Mapa nazwa nagrania -> bilans
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

    videos: dict[str, VideoStats] = {}
    for annotation in raw["annotations"]:
        if annotation.get("frame_role") != "peak":
            continue
        keypoints = annotation.get("keypoints")
        if not keypoints:
            continue
        image = raw_images[annotation["image_id"]]
        file_name = image["file_name"]
        video = Path(file_name).parent.name
        cleaned = hide_out_of_frame(
            keypoints,
            image_size=(image.get("width"), image.get("height")),
            bbox=annotation.get("bbox"),
        )
        accepted = file_name in kept
        quality = assess_frame(cleaned, thresholds)
        group, severity = _group_of(quality, accepted)
        caption = (
            f"asym {quality.asymmetry:.2f} | slabe {quality.weak_ratio:.0%} "
            f"| morda {quality.face_width:.0f}px"
        )
        stats = videos.setdefault(video, VideoStats(video))
        stats.frames.append(
            (Sample(file_name, group, severity, caption), cleaned, accepted)
        )
        if not accepted:
            stats.reasons[group] += 1
    return videos


def distribution(videos: dict[str, VideoStats]) -> list[dict]:
    """
    Liczy rozkład: ile nagrań dało ile przyjętych klatek.

    Args:
        videos: Bilanse nagrań

    Returns:
        Lista wierszy rozkładu
    """
    counts = Counter(stats.kept for stats in videos.values())
    rows = []
    for kept in sorted(counts):
        rows.append(
            {
                "kept": kept,
                "videos": counts[kept],
                "frames": kept * counts[kept],
            }
        )
    return rows


def _pick(videos: dict[str, VideoStats]) -> list[VideoStats]:
    """
    Wybiera nagrania do pokazania: najwydajniejsze i te, które straciły najwięcej.

    Args:
        videos: Bilanse nagrań

    Returns:
        Lista bilansów w kolejności prezentacji
    """
    productive = sorted(
        (s for s in videos.values() if s.kept > 0),
        key=lambda s: (-s.kept, -s.yield_ratio),
    )[:SHOWN_TOP]
    # Nagrania, w których pipeline znalazł DUŻO peaków, a przeżył jeden lub
    # żaden — tu leży największy pojedynczy zysk z poluzowania progów.
    wasted = sorted(
        (s for s in videos.values() if s.total >= 4 and s.kept <= 1),
        key=lambda s: -(s.total - s.kept),
    )[:SHOWN_BOTTOM]
    return productive + wasted


def main() -> None:
    """Punkt wejścia: renderuje miniatury nagraniami i zapisuje manifest."""
    parser = argparse.ArgumentParser(description="Podglad zbioru nagraniami")
    parser.add_argument("--dataset", type=Path, default=Path(DEFAULT_DATASET))
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUTPUT))
    args = parser.parse_args()

    videos = collect(args.dataset)
    frames_root = args.dataset / "frames"
    args.out.mkdir(parents=True, exist_ok=True)

    shown: list[dict] = []
    for index, stats in enumerate(_pick(videos)):
        target = args.out / f"v{index:02d}"
        target.mkdir(exist_ok=True)
        entries: list[dict] = []
        ordered = sorted(stats.frames, key=lambda item: (not item[2], item[0].severity))
        for position, (sample, keypoints, accepted) in enumerate(ordered):
            canvas = render(frames_root, sample, keypoints)
            if canvas is None:
                continue
            name = f"{position:03d}.jpg"
            cv2.imwrite(
                str(target / name), canvas, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            entries.append(
                {
                    "file": f"v{index:02d}/{name}",
                    "caption": sample.caption,
                    "accepted": accepted,
                    "group": sample.group,
                }
            )
        shown.append(
            {
                "video": stats.video,
                "total": stats.total,
                "kept": stats.kept,
                "reasons": dict(stats.reasons),
                "frames": entries,
            }
        )

    per_video = [s.total for s in videos.values()]
    summary = {
        "videos": len(videos),
        "peaks": sum(per_video),
        "kept": sum(s.kept for s in videos.values()),
        "videos_with_any": sum(1 for s in videos.values() if s.kept),
        "distribution": distribution(videos),
    }
    (args.out / "manifest.json").write_text(
        json.dumps({"summary": summary, "shown": shown}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"nagrań: {summary['videos']}, peaków: {summary['peaks']}, "
          f"przyjętych: {summary['kept']}")
    print(f"nagrań z choć jedną przyjętą klatką: {summary['videos_with_any']}")
    for row in summary["distribution"][:12]:
        print(f"  {row['kept']:2} przyjętych -> {row['videos']:4} nagrań")


if __name__ == "__main__":
    main()
