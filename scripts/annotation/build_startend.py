#!/usr/bin/env python3
"""
Finalny zbiór wg wymogu prowadzącego: DOKŁADNIE 2 kadry na wideo — POCZĄTEK i
KONIEC emocji, w sekundach zaznaczonych ręcznie (przyciski ①/② w narzędziu).

Dla każdego wideo:
  1. otwieramy plik z cache (/tmp/dogvids, pobrany przez narzędzie weryfikacji),
  2. bierzemy klatkę w sekundzie start i w sekundzie end,
  3. jeśli w tej klatce psa nie ma / jest rozmyta / słabe keypoints — dobieramy
     NAJBLIŻSZĄ dobrą klatkę z tego samego wideo (okno ±0.3/±0.6/±1.0 s),
  4. liczymy bbox + keypoints (modele) oraz AU (delta względem klatki neutralnej
     tego wideo z annotations_full),
  5. emocja = werdykt człowieka.

Zawsze wychodzą 2 kadry na wideo (chyba że w całym oknie nie ma psa — wtedy
bierzemy najlepszą dostępną klatkę, nic nie gubimy po cichu).
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

import cv2
import numpy as np

from packages.models.delta_action_units import DeltaActionUnitsExtractor
from packages.pipeline.inference import InferencePipeline, PipelineConfig
from scripts.annotation.build_structured import infer_license

REPO = Path(__file__).resolve().parent.parent.parent
DATASET = "dataset_final"
FULL = REPO / "data" / DATASET / "work" / "annotations_full.json"
LABELS = REPO / "data" / "labels" / DATASET
CACHE = Path("/tmp/dogvids")
OUTPUT = REPO / "data" / DATASET / "release_startend"

EMOTIONS = ["neutral", "sad", "happy", "surprise", "angry", "fearful"]
# Progi jakości klatki (te same co pipeline): pewność keypoints i ostrość (anty-blur)
MIN_KP_CONF = 0.5
MIN_SHARPNESS = 60.0
# Okno dobierania zastępczej klatki, w sekundach (0 = dokładna sekunda)
SEARCH_OFFSETS = [0.0, 0.3, -0.3, 0.6, -0.6, 1.0, -1.0]

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def human_labels() -> dict[str, dict]:
    """Mapa source_video -> ostatni usable werdykt człowieka (emocja + czasy)."""
    out: dict[str, dict] = {}
    for path in LABELS.glob("video_*.jsonl"):
        for line in path.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("emotion") and r.get("usable", True):
                    out[r["video"]] = r
    return out


def neutral_by_video() -> dict[str, np.ndarray]:
    """Keypoints klatki neutralnej dominującego psa dla każdego wideo."""
    coco = json.loads(FULL.read_text(encoding="utf-8"))
    images = {i["id"]: i for i in coco["images"]}
    ann_by_img: dict[int, list[dict]] = defaultdict(list)
    for a in coco["annotations"]:
        ann_by_img[a["image_id"]].append(a)
    peaks_by_video: dict[str, list[dict]] = defaultdict(list)
    for a in coco["annotations"]:
        if a.get("frame_role") == "peak":
            img = images[a["image_id"]]
            v = img.get("source_video") or img["file_name"].split("/")[-2]
            peaks_by_video[v].append(a)

    out: dict[str, np.ndarray] = {}
    for v, peaks in peaks_by_video.items():
        by_track: dict[object, list[dict]] = defaultdict(list)
        for p in peaks:
            by_track[p.get("track_id")].append(p)
        main = max(by_track.values(), key=len)
        peak = main[0]
        cands = ann_by_img.get(peak.get("neutral_frame_id"), [])
        match = next((a for a in cands if a.get("track_id") == peak.get("track_id")), None)
        if match is None and len(cands) == 1:
            match = cands[0]
        if match and match.get("keypoints"):
            out[v] = np.array(match["keypoints"], dtype=float)
    return out


def _sharpness(crop: np.ndarray) -> float:
    """Wariancja Laplasjanu = miara ostrości (im wyżej, tym mniej rozmycia)."""
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _read_at(cap: cv2.VideoCapture, t: float) -> np.ndarray | None:
    """Czyta klatkę wideo w sekundzie t."""
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
    ok, frame = cap.read()
    return frame if ok else None


def _pick_dog(pipeline: InferencePipeline, frame: np.ndarray):
    """Największy pies z keypoints; zwraca (annotation, kp_conf, sharpness) lub None."""
    result = pipeline.process_frame(frame)
    best = None
    for ann in result.annotations:
        if ann.keypoints is None:
            continue
        x, y, w, h = ann.bbox
        area = w * h
        if best is None or area > best[0]:
            best = (area, ann)
    if best is None:
        return None
    ann = best[1]
    x, y, w, h = ann.bbox
    sharp = _sharpness(frame[y : y + h, x : x + w])
    return ann, float(ann.keypoints.confidence), sharp


def best_frame(pipeline: InferencePipeline, cap: cv2.VideoCapture, t: float, duration: float):
    """
    Zwraca (frame, annotation) najlepszej klatki blisko sekundy t.

    Najpierw próbuje dokładnej sekundy; jeśli pies rozmyty/słaby/brak — dobiera
    najbliższą dobrą klatkę z okna. Gdy nic nie przejdzie progu, oddaje najlepszą
    dostępną (nic nie gubimy). None tylko gdy w całym oknie nie ma psa.
    """
    fallback = None  # (score, frame, ann)
    for i, off in enumerate(SEARCH_OFFSETS):
        tt = min(max(0.0, t + off), max(0.0, duration - 0.05))
        frame = _read_at(cap, tt)
        if frame is None:
            continue
        picked = _pick_dog(pipeline, frame)
        if picked is None:
            continue
        ann, conf, sharp = picked
        if conf >= MIN_KP_CONF and sharp >= MIN_SHARPNESS:
            return frame, ann, i > 0  # dobra klatka (i>0 => inna sekunda niż zaznaczona)
        score = conf * min(sharp / MIN_SHARPNESS, 1.0)
        if fallback is None or score > fallback[0]:
            fallback = (score, frame, ann)
    if fallback is not None:
        return fallback[1], fallback[2], True  # najlepsza dostępna, poniżej progu
    return None, None, False


def _au_analysis(kp_coco: list[float], neutral_kp: np.ndarray) -> dict[str, dict]:
    """AU jako delta pik/neutral z keypoints (ratio, is_active, confidence)."""
    extractor = DeltaActionUnitsExtractor(neutral_kp)
    result = extractor.extract(np.array(kp_coco, dtype=float))
    return {
        name: {"ratio": au.ratio, "is_active": au.is_active, "confidence": au.confidence}
        for name, au in result.items()
    }


def build() -> None:
    """Składa zbiór release_startend."""
    labels = human_labels()
    neutrals = neutral_by_video()
    logger.info("Wideo z etykietą: %d | z bazą neutralną: %d", len(labels), len(neutrals))

    config = PipelineConfig(device="cpu")
    pipeline = InferencePipeline(config)
    pipeline.load()
    logger.info("Modele załadowane (cpu).")

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
    substituted = 0
    no_dog = 0
    img_id = ann_id = 0

    videos = [v for v, r in labels.items() if r["emotion"] in EMOTIONS and v in neutrals]
    for n, video in enumerate(videos, 1):
        record = labels[video]
        emotion = record["emotion"]
        mp4 = CACHE / (video.replace("/", "_") + ".mp4")
        if not mp4.is_file():
            logger.warning("brak w cache: %s", video)
            continue
        cap = cv2.VideoCapture(str(mp4))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames_n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = frames_n / fps if fps else 0.0
        neutral_kp = neutrals[video]

        platform, lic, link = infer_license(video)
        if video not in seen_lic:
            lic_rows.append({"video_id": video, "platform": platform, "license": lic, "link": link})
            seen_lic.add(video)

        for role, t in (("start", record.get("start_time")), ("end", record.get("end_time"))):
            if t is None:
                continue
            frame, ann, was_substituted = best_frame(pipeline, cap, float(t), duration)
            if frame is None:
                no_dog += 1
                continue
            if was_substituted:
                substituted += 1

            height, width = frame.shape[:2]
            rand = uuid.uuid4().hex + ".jpg"
            cv2.imwrite(str(OUTPUT / emotion / rand), frame)
            img_id += 1
            out_images.append(
                {
                    "id": img_id,
                    "file_name": f"{emotion}/{rand}",
                    "width": width,
                    "height": height,
                    "source_video": video,
                    "license": lic,
                    "license_link": link,
                }
            )
            x, y, w, h = ann.bbox
            ann_id += 1
            out_anns.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "keypoints": ann.keypoints.to_coco_format(),
                    "num_keypoints": ann.keypoints.num_detected,
                    "au_analysis": _au_analysis(ann.keypoints.to_coco_format(), neutral_kp),
                    "breed": ann.breed.class_name if ann.breed else None,
                    "emotion": emotion,
                    "frame_role": role,
                    "frame_time_s": round(float(t), 2),
                    "label_source": "human_verified",
                    "source_video": video,
                }
            )
            per_emotion[emotion] += 1
        cap.release()
        if n % 20 == 0:
            logger.info("  ... %d/%d wideo, kadrów %d", n, len(videos), len(out_images))

    _write_outputs(out_images, out_anns, lic_rows, per_emotion, len(videos), substituted, no_dog)


def _write_outputs(out_images, out_anns, lic_rows, per_emotion, n_videos, substituted, no_dog) -> None:
    """Zapisuje COCO, licencje, README, raport."""
    coco_out = {
        "info": {
            "description": "Dog FACS Dataset — kadry początku i końca emocji",
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

    # frames.csv — czytelna mapa: które wideo, początek/koniec, sekunda, plik
    file_by_img = {i["id"]: i["file_name"] for i in out_images}
    frame_rows = sorted(
        (
            {
                "source_video": a["source_video"],
                "emotion": a["emotion"],
                "frame_role": a["frame_role"],
                "frame_time_s": a["frame_time_s"],
                "file_name": file_by_img[a["image_id"]],
            }
            for a in out_anns
        ),
        key=lambda r: (r["emotion"], r["source_video"], r["frame_role"]),
    )
    with (OUTPUT / "frames.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["source_video", "emotion", "frame_role", "frame_time_s", "file_name"]
        )
        writer.writeheader()
        writer.writerows(frame_rows)

    lines = [
        "# Dog FACS Dataset — kadry początku i końca emocji",
        "",
        f"Złożono: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Zasada",
        "- **2 kadry na wideo**: POCZĄTEK i KONIEC emocji (sekundy zaznaczone ręcznie).",
        "- Gdy w danej sekundzie pies był rozmyty/niewidoczny — dobrana najbliższa",
        "  dobra klatka z tego samego wideo (`frame_role` = start/end zachowane).",
        "- **Emocja** = werdykt człowieka (`label_source=human_verified`).",
        "- **keypoints** (46 DogFLW) i **bbox** liczone modelami na klatce.",
        "- **AU** (21 DogFACS) = delta względem klatki neutralnej tego psa.",
        "- Pełne klatki + bbox, foldery klas, licencje, losowe nazwy plików.",
        "",
        "## Liczby (kadry na emocję)",
    ]
    for emotion in EMOTIONS:
        lines.append(f"- **{emotion}**: {per_emotion.get(emotion, 0)} kadrów")
    lines += [
        "",
        f"Wideo: {n_videos} · kadrów: {len(out_images)} · AU na wszystkich · "
        f"klatek dobranych (zamiast rozmytej): {substituted} · bez psa w oknie: {no_dog}.",
    ]
    (OUTPUT / "README.md").write_text("\n".join(lines), encoding="utf-8")

    logger.info("=== release_startend gotowy: %s ===", OUTPUT)
    for emotion in EMOTIONS:
        logger.info("  %-9s %3d kadrów", emotion, per_emotion.get(emotion, 0))
    logger.info("Kadrów: %d | dobranych zamiast rozmytej: %d | bez psa: %d",
                len(out_images), substituted, no_dog)


if __name__ == "__main__":
    argparse.ArgumentParser(description="Zbiór start/end Dog FACS").parse_args()
    build()
