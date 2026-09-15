#!/usr/bin/env python3
"""
Zbiór finalny "150+" — pełne przetworzenie WSZYSTKICH ręcznie oznaczonych wideo.

Źródła etykiet:
- stare: data/labels/dataset_final/video_*.jsonl  (video = nazwa na Drive)
- nowe:  data/labels/dataset_final/select_*.jsonl  (fid = ID pliku na Drive)

Dla każdego wideo (jak przy pierwszych 50, tyle że neutralną wykrywamy sami):
  1. pobranie z Drive (po fid / po nazwie) + transkodowanie do H.264 gdy trzeba,
  2. wykrycie klatki NEUTRALNEJ (próbkowanie wideo + NeutralFrameDetector),
  3. kadr POCZĄTKU i KOŃCA emocji (z podmianą na ostry, gdy rozmyty),
  4. bbox + keypoints + rasa (modele), AU = delta względem neutralnej,
  5. nazwy <emocja>_<NNN>_1/2, foldery klas, licencje z linkiem, frames.csv,
     annotations_<klasa>.json + zbiorczy.

fearful i surprise rozdzielone. Uruchamiać: python -m scripts.annotation.build_150plus [--limit N]
"""

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from packages.models.delta_action_units import DeltaActionUnitsExtractor
from packages.pipeline.inference import InferencePipeline, PipelineConfig
from packages.pipeline.neutral_frame import NeutralFrameDetector
from scripts.annotation.refine_startend import direct_license
from scripts.download.tiktok.config import (
    GDRIVE_CREDENTIALS_PATH,
    GDRIVE_FOLDER_ID,
    GDRIVE_TOKEN_PATH,
)
from scripts.download.tiktok.drive_uploader import GoogleDriveUploader

REPO = Path(__file__).resolve().parent.parent.parent
LABELS = REPO / "data" / "labels" / "dataset_final"
CACHE = Path("/tmp/dogvids")
CACHE.mkdir(exist_ok=True)
OUTPUT = REPO / "data" / "dataset_final" / "release_150plus"

EMOTIONS = ["neutral", "sad", "happy", "surprise", "angry", "fearful"]
MERGE: dict[str, str] = {}  # fearful i surprise rozdzielone
MIN_KP_CONF = 0.5
MIN_SHARPNESS = 60.0
SEARCH_OFFSETS = [0.0, 0.3, -0.3, 0.6, -0.6, 1.0, -1.0]
NEUTRAL_SAMPLES = 8  # ile klatek próbkujemy do wykrycia neutralnej

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_drive = GoogleDriveUploader(GDRIVE_CREDENTIALS_PATH, GDRIVE_TOKEN_PATH, GDRIVE_FOLDER_ID)
_name2id: dict[str, str | None] = {}
CHECKPOINT = OUTPUT / "_checkpoint.json"


def _retry(fn, tries: int = 4):
    """Powtarza operację Drive przy błędach sieci/SSL, z ponowną autoryzacją."""
    for i in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — sieć bywa kapryśna przy długim biegu
            if i == tries - 1:
                raise
            logger.warning("Drive błąd (%s), ponawiam %d/%d", type(exc).__name__, i + 1, tries)
            time.sleep(3 * (i + 1))
            try:
                _drive.authenticate()
            except Exception:  # noqa: BLE001
                pass


def _load_records() -> list[dict]:
    """Wszystkie etykiety (nowe select_ + stare video_), dedup, emocja scalona."""
    records: list[dict] = []
    seen_names: set[str] = set()
    # nowe (fid)
    seen_fid: set[str] = set()
    for f in sorted(LABELS.glob("select_*.jsonl")):
        for line in f.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if not r.get("emotion") or r["fid"] in seen_fid:
                    continue
                if r.get("start_time") is None or r.get("end_time") is None:
                    continue
                seen_fid.add(r["fid"])
                seen_names.add(r.get("video"))
                records.append({"fid": r["fid"], "name": r.get("video"),
                                "emotion": MERGE.get(r["emotion"], r["emotion"]),
                                "start": r["start_time"], "end": r["end_time"]})
    # stare (nazwa)
    seen_old: set[str] = set()
    for f in sorted(LABELS.glob("video_*.jsonl")):
        for line in f.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if not r.get("emotion") or r["video"] in seen_old or r["video"] in seen_names:
                    continue
                if r.get("start_time") is None or r.get("end_time") is None:
                    continue
                seen_old.add(r["video"])
                records.append({"fid": None, "name": r["video"],
                                "emotion": MERGE.get(r["emotion"], r["emotion"]),
                                "start": r["start_time"], "end": r["end_time"]})
    return records


def _find_drive(name: str) -> str | None:
    """ID pliku na Drive po nazwie (cache)."""
    if name in _name2id:
        return _name2id[name]
    for cand in (f"{name}.mp4", f"{name}.webm", f"{name}.mov", name):
        safe = cand.replace("'", "\\'")
        r = _retry(lambda q=safe: _drive._service.files().list(
            q=f"name = '{q}' and trashed = false",
            fields="files(id)", pageSize=1).execute()).get("files", [])
        if r:
            _name2id[name] = r[0]["id"]
            return r[0]["id"]
    _name2id[name] = None
    return None


def _video_codec(path: Path) -> str:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30)
        return r.stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _ensure_cached(fid: str) -> Path | None:
    """Pobiera z Drive po ID i (gdy trzeba) transkoduje do H.264 dla cv2."""
    dst = CACHE / f"{fid}.mp4"
    if dst.is_file():
        return dst
    raw = CACHE / f"{fid}.raw"
    try:
        _retry(lambda: _drive.download_file(fid, raw))
    except Exception:  # noqa: BLE001
        raw.unlink(missing_ok=True)
        return None
    if _video_codec(raw) == "h264":
        raw.rename(dst)
    else:
        try:
            subprocess.run(["ffmpeg", "-y", "-i", str(raw), "-c:v", "libx264", "-preset",
                            "veryfast", "-crf", "23", "-c:a", "aac", "-movflags", "+faststart",
                            str(dst)], capture_output=True, timeout=600, check=True)
        except Exception:  # noqa: BLE001
            raw.unlink(missing_ok=True)
            dst.unlink(missing_ok=True)
            return None
        raw.unlink(missing_ok=True)
    return dst


def _pick_dog(pipeline: InferencePipeline, frame: np.ndarray):
    """Największy pies z keypoints -> (annotation, kp_conf, sharpness) lub None."""
    result = pipeline.process_frame(frame)
    best = None
    for ann in result.annotations:
        if ann.keypoints is None:
            continue
        x, y, w, h = ann.bbox
        if best is None or w * h > best[0]:
            best = (w * h, ann)
    if best is None:
        return None
    ann = best[1]
    x, y, w, h = ann.bbox
    sharp = cv2.Laplacian(cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY),
                          cv2.CV_64F).var() if frame[y:y + h, x:x + w].size else 0.0
    return ann, float(ann.keypoints.confidence), float(sharp)


def _read_at(cap: cv2.VideoCapture, t: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
    ok, frame = cap.read()
    return frame if ok else None


def _neutral_kp(pipeline: InferencePipeline, cap: cv2.VideoCapture, duration: float) -> np.ndarray | None:
    """Wykrywa keypoints klatki neutralnej: próbkuje wideo i pyta detektor."""
    frames, kps = [], []
    for i in range(1, NEUTRAL_SAMPLES + 1):
        t = duration * i / (NEUTRAL_SAMPLES + 1)
        frame = _read_at(cap, t)
        if frame is None:
            continue
        picked = _pick_dog(pipeline, frame)
        frames.append(frame)
        kps.append(np.array(picked[0].keypoints.to_coco_format()) if picked else None)
    if not any(k is not None for k in kps):
        return None
    try:
        idx = NeutralFrameDetector().detect_auto(frames, kps)
    except Exception:  # noqa: BLE001
        # fallback: klatka o najwyższej pewności keypoints
        idx = max(range(len(kps)), key=lambda i: -1 if kps[i] is None
                  else float(np.mean(kps[i].reshape(-1, 3)[:, 2])))
    return kps[idx]


def _best_frame(pipeline: InferencePipeline, cap: cv2.VideoCapture, t: float, duration: float):
    """Kadr blisko sekundy t (podmiana na ostry, gdy rozmyty)."""
    fallback = None
    for off in SEARCH_OFFSETS:
        tt = min(max(0.0, t + off), max(0.0, duration - 0.05))
        frame = _read_at(cap, tt)
        if frame is None:
            continue
        picked = _pick_dog(pipeline, frame)
        if picked is None:
            continue
        ann, conf, sharp = picked
        if conf >= MIN_KP_CONF and sharp >= MIN_SHARPNESS:
            return frame, ann
        score = conf * min(sharp / MIN_SHARPNESS, 1.0)
        if fallback is None or score > fallback[0]:
            fallback = (score, frame, ann)
    return (fallback[1], fallback[2]) if fallback else (None, None)


def build(limit: int | None = None) -> None:
    records = _load_records()
    if limit:
        records = records[:limit]
    logger.info("Etykiet do przetworzenia: %d", len(records))

    _drive.authenticate()
    pipeline = InferencePipeline(PipelineConfig(device="cpu"))
    pipeline.load()
    logger.info("Modele załadowane (cpu).")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for e in EMOTIONS:
        (OUTPUT / e).mkdir(exist_ok=True)

    st = _load_checkpoint()
    out_images, out_anns, lic_rows = st["out_images"], st["out_anns"], st["lic_rows"]
    counters, stats = Counter(st["counters"]), Counter(st["stats"])
    iid, aid = st["iid"], st["aid"]
    done_keys: set = set(st["done_keys"])
    if done_keys:
        logger.info("Wznawiam — już zrobione: %d", len(done_keys))

    for n, rec in enumerate(records, 1):
        key = rec["fid"] or rec["name"]
        if key in done_keys:
            continue
        try:
            fid = rec["fid"] or _find_drive(rec["name"])
            path = _ensure_cached(fid) if fid else None
            if path is None:
                stats["brak_pliku"] += 1
            else:
                cap = cv2.VideoCapture(str(path))
                try:
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps
                    neutral = _neutral_kp(pipeline, cap, duration)
                    if neutral is None:
                        stats["brak_neutralnej"] += 1
                    else:
                        extractor = DeltaActionUnitsExtractor(neutral)
                        emotion = rec["emotion"]
                        counters[emotion] += 1
                        base = f"{emotion}_{counters[emotion]:03d}"
                        made = 0
                        for role, t in (("start", rec["start"]), ("end", rec["end"])):
                            frame, ann = _best_frame(pipeline, cap, float(t), duration)
                            if frame is None:
                                continue
                            kp = ann.keypoints.to_coco_format()
                            au = {name: {"ratio": a.ratio, "is_active": a.is_active,
                                         "confidence": a.confidence}
                                  for name, a in extractor.extract(np.array(kp, dtype=float)).items()}
                            h, w = frame.shape[:2]
                            fname = f"{emotion}/{base}_{1 if role == 'start' else 2}.jpg"
                            cv2.imwrite(str(OUTPUT / fname), frame)
                            iid += 1
                            aid += 1
                            x, y, bw, bh = ann.bbox
                            out_images.append({"id": iid, "file_name": fname, "width": w,
                                               "height": h, "source_video": rec["name"]})
                            out_anns.append({"id": aid, "image_id": iid, "category_id": 1,
                                             "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0,
                                             "keypoints": kp, "num_keypoints": ann.keypoints.num_detected,
                                             "au_analysis": au,
                                             "breed": ann.breed.class_name if ann.breed else None,
                                             "emotion": emotion, "frame_role": role,
                                             "frame_time_s": round(float(t), 2),
                                             "label_source": "human_verified",
                                             "source_video": rec["name"]})
                            made += 1
                        if made == 0:
                            counters[emotion] -= 1
                            stats["brak_psa"] += 1
                        else:
                            lic_rows[rec["name"]] = direct_license(rec["name"])
                            stats["ok"] += 1
                finally:
                    cap.release()
        except Exception as exc:  # noqa: BLE001 — nie wywracamy całego biegu
            logger.warning("  ! błąd %s: %s", rec["name"], exc)
            stats["blad"] += 1
        done_keys.add(key)
        if n % 20 == 0:
            logger.info("  ... %d/%d | kadrów %d | %s", n, len(records), len(out_images), dict(stats))
        if n % 50 == 0:
            _save_checkpoint(out_images, out_anns, lic_rows, counters, stats, iid, aid, done_keys)

    _write(out_images, out_anns, lic_rows, counters, stats)
    CHECKPOINT.unlink(missing_ok=True)


def _load_checkpoint() -> dict:
    if CHECKPOINT.is_file():
        d = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        logger.info("Znaleziono checkpoint: %d kadrów", len(d.get("out_images", [])))
        return d
    return {"out_images": [], "out_anns": [], "lic_rows": {}, "counters": {},
            "stats": {}, "iid": 0, "aid": 0, "done_keys": []}


def _save_checkpoint(out_images, out_anns, lic_rows, counters, stats, iid, aid, done_keys) -> None:
    CHECKPOINT.write_text(json.dumps({
        "out_images": out_images, "out_anns": out_anns, "lic_rows": lic_rows,
        "counters": dict(counters), "stats": dict(stats), "iid": iid, "aid": aid,
        "done_keys": sorted(done_keys)}, ensure_ascii=False), encoding="utf-8")


def _write(out_images, out_anns, lic_rows, counters, stats) -> None:
    info = {"description": "Dog FACS Dataset 150+", "contributor": "Politechnika Gdańska WETI",
            "date_created": datetime.now(timezone.utc).strftime("%Y-%m-%d")}
    categories = [{"id": 1, "name": "dog", "supercategory": "animal", "keypoints": [], "skeleton": []}]
    imgs_by_id = {i["id"]: i for i in out_images}
    per_class: dict[str, dict] = defaultdict(lambda: {"images": [], "annotations": []})
    for a in out_anns:
        per_class[a["emotion"]]["annotations"].append(a)
        per_class[a["emotion"]]["images"].append(imgs_by_id[a["image_id"]])
    for emotion, data in per_class.items():
        (OUTPUT / f"annotations_{emotion}.json").write_text(
            json.dumps({"info": info, "categories": categories, **data}, ensure_ascii=False),
            encoding="utf-8")
    (OUTPUT / "annotations.json").write_text(
        json.dumps({"info": info, "categories": categories, "images": out_images,
                    "annotations": out_anns}, ensure_ascii=False), encoding="utf-8")
    with (OUTPUT / "licenses.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["source_video", "platform", "license", "link"])
        w.writeheader()
        for sv, (plat, lic, link) in sorted(lic_rows.items()):
            w.writerow({"source_video": sv, "platform": plat, "license": lic, "link": link})
    with (OUTPUT / "frames.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["source_video", "emotion", "frame_role", "frame_time_s", "file_name"])
        w.writeheader()
        rows = [{"source_video": a["source_video"], "emotion": a["emotion"],
                 "frame_role": a["frame_role"], "frame_time_s": a["frame_time_s"],
                 "file_name": imgs_by_id[a["image_id"]]["file_name"]} for a in out_anns]
        w.writerows(sorted(rows, key=lambda r: (r["emotion"], r["file_name"])))

    logger.info("=== release_150plus gotowy: %s ===", OUTPUT)
    for e in EMOTIONS:
        logger.info("  %-9s %3d wideo (%d kadrów)", e, counters.get(e, 0), len(per_class[e]["images"]))
    logger.info("Kadrów: %d | statusy: %s", len(out_images), dict(stats))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Zbiór 150+ (pełne przetworzenie)")
    ap.add_argument("--limit", type=int, default=None, help="Przetwórz tylko N pierwszych (test)")
    args = ap.parse_args()
    build(args.limit)
