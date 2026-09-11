#!/usr/bin/env python3
"""
Finalny mały zbiór (cel 50/emocja) ze WSZYSTKIMI punktami prowadzącego + AU na
KAŻDYM kadrze.

Emocja pochodzi z werdyktu człowieka (dziennik `video_<kto>.jsonl`), a kadry i
geometria z gotowego `annotations_full.json`. AU liczone są tu na miejscu z
keypoints (pik względem klatki neutralnej TEGO psa) — dzięki temu mamy AU także
tam, gdzie bramka kuracji ich wcześniej nie policzyła. Nic nie jest zmyślane:
- emocja = człowiek,
- AU = geometria pik/neutral (ratio, is_active, confidence),
- 2 kadry na wideo = pierwszy i ostatni pik (onset/offset ekspresji),
- pełne klatki + bbox psa, foldery klas, licencje, losowe nazwy plików.

`confidence` przy AU niesie wiarygodność (mała/obrócona morda -> niska pewność),
więc niskiej jakości pomiary są przezroczyste, nie ukryte.
"""

import argparse
import csv
import json
import logging
import shutil
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from packages.models.delta_action_units import DeltaActionUnitsExtractor
from scripts.annotation.build_structured import infer_license

REPO = Path(__file__).resolve().parent.parent.parent
DATASET = "dataset_final"
SRC = REPO / "data" / DATASET / "work" / "annotations_full.json"
FRAMES = REPO / "data" / DATASET / "work" / "frames"
LABELS = REPO / "data" / "labels" / DATASET
OUTPUT = REPO / "data" / DATASET / "release_50"

# 6 docelowych emocji (jak w narzędziu weryfikacji wideo)
EMOTIONS = ["neutral", "sad", "happy", "surprise", "angry", "fearful"]
FRAMES_PER_VIDEO = 2  # onset + offset ekspresji (wymóg: 2 kadry na wideo)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def human_emotions() -> dict[str, str]:
    """Mapa source_video -> emocja człowieka (ostatni usable werdykt wygrywa)."""
    out: dict[str, str] = {}
    for path in LABELS.glob("video_*.jsonl"):
        for line in path.open(encoding="utf-8"):
            if line.strip():
                record = json.loads(line)
                if record.get("emotion") and record.get("usable", True):
                    out[record["video"]] = record["emotion"]
    return out


def _neutral_keypoints(
    ann_by_img: dict[int, list[dict]], neutral_id: int, track_id
) -> np.ndarray | None:
    """Keypoints klatki neutralnej DLA TEGO psa (dopasowanie po track_id)."""
    candidates = ann_by_img.get(neutral_id, [])
    match = next((a for a in candidates if a.get("track_id") == track_id), None)
    # gdy brak dopasowania po treku, a jest tylko jeden pies — bierzemy jego
    if match is None and len(candidates) == 1:
        match = candidates[0]
    if match is None or not match.get("keypoints"):
        return None
    return np.array(match["keypoints"], dtype=float)


def _au_analysis(peak_kp: list[float], neutral_kp: np.ndarray) -> dict[str, dict]:
    """Liczy AU jako delta pik/neutral. Zwraca {AU: {ratio, is_active, confidence}}."""
    extractor = DeltaActionUnitsExtractor(neutral_kp)
    result = extractor.extract(np.array(peak_kp, dtype=float))
    return {
        name: {"ratio": au.ratio, "is_active": au.is_active, "confidence": au.confidence}
        for name, au in result.items()
    }


def _pick_peaks(peaks: list[dict], images: dict[int, dict]) -> list[dict]:
    """Dominujący pies (najwięcej pików), z niego pierwszy i ostatni pik po numerze klatki."""
    by_track: dict[object, list[dict]] = defaultdict(list)
    for annotation in peaks:
        by_track[annotation.get("track_id")].append(annotation)
    main = max(by_track.values(), key=len)
    main.sort(key=lambda a: images[a["image_id"]].get("frame_number", 0))
    if len(main) <= FRAMES_PER_VIDEO:
        return main
    return [main[0], main[-1]]


def build() -> None:
    """Składa zbiór release_50."""
    coco = json.loads(SRC.read_text(encoding="utf-8"))
    images = {i["id"]: i for i in coco["images"]}
    ann_by_img: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        ann_by_img[annotation["image_id"]].append(annotation)

    peaks_by_video: dict[str, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        if annotation.get("frame_role") != "peak":
            continue
        image = images[annotation["image_id"]]
        video = image.get("source_video") or image["file_name"].split("/")[-2]
        peaks_by_video[video].append(annotation)

    human = human_emotions()

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    for emotion in EMOTIONS:
        (OUTPUT / emotion).mkdir()

    out_images: list[dict] = []
    out_anns: list[dict] = []
    lic_rows: list[dict] = []
    seen_lic: set[str] = set()
    per_emotion: Counter = Counter()
    skipped_no_neutral = 0
    img_id = ann_id = 0

    for video, emotion in human.items():
        if emotion not in EMOTIONS or video not in peaks_by_video:
            continue
        platform, lic, link = infer_license(video)
        if video not in seen_lic:
            lic_rows.append({"video_id": video, "platform": platform, "license": lic, "link": link})
            seen_lic.add(video)

        for annotation in _pick_peaks(peaks_by_video[video], images):
            image = images[annotation["image_id"]]
            src = FRAMES / image["file_name"]
            neutral_kp = _neutral_keypoints(
                ann_by_img, annotation.get("neutral_frame_id"), annotation.get("track_id")
            )
            if not src.is_file() or neutral_kp is None:
                skipped_no_neutral += 1
                continue

            rand = uuid.uuid4().hex + ".jpg"
            shutil.copy2(src, OUTPUT / emotion / rand)
            img_id += 1
            out_images.append(
                {
                    "id": img_id,
                    "file_name": f"{emotion}/{rand}",
                    "width": image.get("width"),
                    "height": image.get("height"),
                    "source_video": video,
                    "license": lic,
                    "license_link": link,
                }
            )
            ann_id += 1
            out_anns.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": annotation.get("bbox"),
                    "area": annotation.get("area"),
                    "iscrowd": 0,
                    "keypoints": annotation.get("keypoints"),
                    "num_keypoints": annotation.get("num_keypoints"),
                    "au_analysis": _au_analysis(annotation["keypoints"], neutral_kp),
                    "breed": annotation.get("breed"),
                    "emotion": emotion,
                    "label_source": "human_verified",
                    "source_video": video,
                }
            )
            per_emotion[emotion] += 1

    _write_outputs(out_images, out_anns, lic_rows, per_emotion, human, skipped_no_neutral)


def _write_outputs(out_images, out_anns, lic_rows, per_emotion, human, skipped) -> None:
    """Zapisuje COCO, licencje, README i raport."""
    coco_out = {
        "info": {
            "description": "Dog FACS Dataset — zbiór finalny (50/emocja)",
            "date_created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "contributor": "Politechnika Gdańska WETI",
        },
        "categories": [
            {"id": 1, "name": "dog", "supercategory": "animal", "keypoints": [], "skeleton": []}
        ],
        "images": out_images,
        "annotations": out_anns,
    }
    (OUTPUT / "annotations.json").write_text(json.dumps(coco_out, ensure_ascii=False), encoding="utf-8")

    with (OUTPUT / "licenses.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "platform", "license", "link"])
        writer.writeheader()
        writer.writerows(lic_rows)

    videos = sum(1 for v in human if human[v] in EMOTIONS)
    lines = [
        "# Dog FACS Dataset — zbiór finalny",
        "",
        f"Złożono: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Struktura",
        "- Foldery = klasy emocji; w każdym pełne klatki wideo tej emocji.",
        "- Pliki nazwane losowym ciągiem. `annotations.json` (COCO): bbox psa,",
        "  46 keypoints (DogFLW), 21 AU (DogFACS), emocja, `source_video`.",
        "- `licenses.csv`: licencja + link źródła dla każdego wideo.",
        "",
        "## Etykiety",
        "- **Emocja** = werdykt człowieka (każde wideo obejrzane, `label_source=human_verified`).",
        "- **AU** = delta geometrii pik/neutral tego samego psa (ratio, is_active, confidence).",
        "  Pole `confidence` niesie wiarygodność pomiaru — niska przy małej/obróconej mordzie.",
        "- **2 kadry na wideo** = pierwszy i ostatni pik ekspresji (onset/offset).",
        "",
        "## Liczby (kadry na emocję)",
    ]
    for emotion in EMOTIONS:
        lines.append(f"- **{emotion}**: {per_emotion.get(emotion, 0)} kadrów")
    lines += [
        "",
        f"Wideo (człowiek): {videos} · kadrów łącznie: {len(out_images)} · "
        f"AU policzone na: {len(out_anns)}/{len(out_anns)} kadrach (100%).",
    ]
    (OUTPUT / "README.md").write_text("\n".join(lines), encoding="utf-8")

    logger.info("=== release_50 gotowy: %s ===", OUTPUT)
    for emotion in EMOTIONS:
        logger.info("  %-9s %3d kadrów", emotion, per_emotion.get(emotion, 0))
    logger.info("Kadrów: %d | AU na wszystkich | pominięto (brak neutral/pliku): %d",
                len(out_images), skipped)


if __name__ == "__main__":
    argparse.ArgumentParser(description="Zbiór finalny Dog FACS (50/emocja)").parse_args()
    build()
