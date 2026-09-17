#!/usr/bin/env python3
"""
Wersja build_150plus pod Google Colab (GPU) — bez Drive API i bez OAuth.

Różnice względem build_150plus:
- wideo czyta z LOKALNYCH folderów (zamontowany Dysk: happy_final, sad_final, ...),
  dopasowując po nazwie z etykiety — nie pobiera nic przez API,
- wynik (kadry + COCO + checkpoint) pisze na zamontowany Dysk, żeby przetrwał
  rozłączenie Colaba (liczy się CZĘŚCIAMI — checkpoint co 50 wideo),
- device = 'cuda'.

Ścieżki podajemy zmiennymi środowiskowymi (ustawia je notebook):
  COLAB_VIDEO_DIRS  — foldery z wideo, oddzielone ';' (np. .../happy_final;.../sad_final)
  COLAB_LABELS_DIR  — folder z *.jsonl (etykiety)
  COLAB_OUTPUT      — folder wynikowy na Dysku
  COLAB_DEVICE      — 'cuda' (domyślnie) lub 'cpu'
  COLAB_EMOTIONS    — które emocje robić, po przecinku (domyślnie 'happy,sad')

Uruchom w Colabie:  python -m scripts.annotation.build_colab
"""

import csv
import json
import logging
import os
import shutil
import subprocess
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from packages.models.delta_action_units import DeltaActionUnitsExtractor
from packages.pipeline.inference import InferencePipeline, PipelineConfig
from packages.pipeline.neutral_frame import NeutralFrameDetector
from scripts.annotation.refine_startend import direct_license

VIDEO_DIRS = [Path(p) for p in os.environ.get("COLAB_VIDEO_DIRS", "").split(";") if p]
LABELS = Path(os.environ.get("COLAB_LABELS_DIR", "data/labels/dataset_final"))
# OUTPUT (na Dysku) to CEL końcowy; liczymy LOKALNIE (WORK) i synchronizujemy na Dysk,
# bo zapis wprost na zamontowany Dysk pada przy dłuższym biegu (Errno 107).
DRIVE_OUT = Path(os.environ.get("COLAB_OUTPUT", "release_colab"))
OUTPUT = Path(os.environ.get("COLAB_WORK", "/content/dogfacs_out"))
DEVICE = os.environ.get("COLAB_DEVICE", "cuda")
EMOTIONS_DO = set(os.environ.get("COLAB_EMOTIONS", "happy,sad").split(","))
CACHE = Path("/tmp/dogvids_colab")
CACHE.mkdir(exist_ok=True, parents=True)

EMOTIONS = ["neutral", "sad", "happy", "surprise", "angry", "fearful"]
MIN_KP_CONF = 0.5
MIN_SHARPNESS = 60.0
SEARCH_OFFSETS = [0.0, 0.3, -0.3, 0.6, -0.6, 1.0, -1.0]
NEUTRAL_SAMPLES = 8
VIDEO_EXT = (".mp4", ".mov", ".webm", ".m4v", ".mkv", ".avi")
COPY_TRIES = 4  # ile razy próbujemy skopiować plik z Dysku (FUSE bywa zrywa)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
CHECKPOINT = OUTPUT / "_checkpoint.json"


def _index_videos() -> dict[str, Path]:
    """Mapa nazwa/stem -> ścieżka pliku wideo we wszystkich folderach źródłowych."""
    idx: dict[str, Path] = {}
    for d in VIDEO_DIRS:
        if not d.is_dir():
            logger.warning("brak folderu wideo: %s", d)
            continue
        for f in d.iterdir():
            if f.suffix.lower() in VIDEO_EXT:
                idx.setdefault(f.name, f)
                idx.setdefault(f.stem, f)
    return idx


def _load_records() -> list[dict]:
    """Etykiety (nowe select_ z fid+nazwą, stare video_ z nazwą), dedup."""
    records: list[dict] = []
    seen_names: set[str] = set()
    seen_fid: set[str] = set()
    for f in sorted(LABELS.glob("select_*.jsonl")):
        if "skip" in f.name:
            continue
        for line in f.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("emotion") or r.get("fid") in seen_fid:
                continue
            if r.get("start_time") is None or r.get("end_time") is None:
                continue
            seen_fid.add(r["fid"])
            if r.get("video"):
                seen_names.add(r["video"])
            records.append({"name": r.get("video"), "emotion": r["emotion"],
                            "start": r["start_time"], "end": r["end_time"]})
    seen_old: set[str] = set()
    for f in sorted(LABELS.glob("video_*.jsonl")):
        for line in f.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            v = r.get("video")
            if not r.get("emotion") or v in seen_old or v in seen_names:
                continue
            if r.get("start_time") is None or r.get("end_time") is None:
                continue
            seen_old.add(v)
            records.append({"name": v, "emotion": r["emotion"],
                            "start": r["start_time"], "end": r["end_time"]})
    return [r for r in records if r["emotion"] in EMOTIONS_DO]


def _video_codec(path: Path) -> str:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=60)
        return r.stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _unzip_video(path: Path) -> Path | None:
    """Część pobrań z Envato to ZIP z wideo w środku — wypakowuje je do /tmp."""
    try:
        with zipfile.ZipFile(path) as z:
            vids = [n for n in z.namelist() if n.lower().endswith(VIDEO_EXT)]
            if not vids:
                return None
            inner = CACHE / f"{path.stem}.inner{Path(vids[0]).suffix}"
            with z.open(vids[0]) as s, inner.open("wb") as d:
                shutil.copyfileobj(s, d)
    except Exception:  # noqa: BLE001
        return None
    return inner


def _copy_from_drive(src: Path, dst: Path) -> bool:
    """Kopiuje plik z zamontowanego Dysku lokalnie — sekwencyjnie, z ponowieniem.

    cv2 czytając wprost z Dysku robi mnóstwo losowych seeków i zrywa FUSE
    (Errno 107). Jedna sekwencyjna kopia jest dużo stabilniejsza; przy zerwaniu
    dajemy FUSE chwilę i próbujemy jeszcze raz.
    """
    for attempt in range(COPY_TRIES):
        try:
            with src.open("rb") as s, dst.open("wb") as d:
                shutil.copyfileobj(s, d, length=8 << 20)
            return True
        except OSError as exc:
            dst.unlink(missing_ok=True)
            logger.warning("  kopia %s nieudana (%s), próba %d/%d",
                           src.name, exc, attempt + 1, COPY_TRIES)
            if attempt == COPY_TRIES - 1:
                return False
            time.sleep(3 * (attempt + 1))
    return False


def _fetch_local(src: Path, stem: str) -> Path | None:
    """Ściąga wideo z Dysku do /tmp (rozpakowuje ZIP), zwraca plik gotowy dla cv2."""
    with src.open("rb") as fh:
        is_zip = fh.read(4) == b"PK\x03\x04"  # część pobrań z Envato to ZIP
    if is_zip:
        raw = _unzip_video(src)
        if raw is None:
            return None
    else:
        raw = CACHE / f"{stem}{src.suffix.lower()}"
        if not _copy_from_drive(src, raw):
            return None
    # cv2 lubi H.264/mp4 — resztę (av1/vp9/mov) przekodowujemy lokalnie
    if _video_codec(raw) == "h264" and raw.suffix.lower() == ".mp4":
        return raw
    dst = CACHE / f"{stem}.mp4"
    try:
        subprocess.run(["ffmpeg", "-y", "-i", str(raw), "-c:v", "libx264", "-preset",
                        "veryfast", "-crf", "23", "-an", str(dst)],
                       capture_output=True, timeout=600, check=True)
    except Exception:  # noqa: BLE001
        dst.unlink(missing_ok=True)
        return None
    finally:
        raw.unlink(missing_ok=True)
    return dst


def _pick_dog(pipeline: InferencePipeline, frame: np.ndarray):
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
    crop = frame[y:y + h, x:x + w]
    sharp = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() \
        if crop.size else 0.0
    return ann, float(ann.keypoints.confidence), float(sharp)


def _read_at(cap: cv2.VideoCapture, t: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
    ok, frame = cap.read()
    return frame if ok else None


def _neutral_kp(pipeline: InferencePipeline, cap: cv2.VideoCapture, duration: float):
    frames, kps = [], []
    for i in range(1, NEUTRAL_SAMPLES + 1):
        frame = _read_at(cap, duration * i / (NEUTRAL_SAMPLES + 1))
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
        idx = max(range(len(kps)), key=lambda i: -1 if kps[i] is None
                  else float(np.mean(kps[i].reshape(-1, 3)[:, 2])))
    return kps[idx]


def _best_frame(pipeline: InferencePipeline, cap: cv2.VideoCapture, t: float, duration: float):
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


def _load_checkpoint() -> dict:
    # Po restarcie środowiska Colaba /content znika — checkpoint (i kadry) bierzemy
    # z Dysku, żeby po każdym "Reconnect + Run all" liczyć DALEJ, a nie od zera.
    if not CHECKPOINT.is_file():
        drive_ck = DRIVE_OUT / "_checkpoint.json"
        try:
            if drive_ck.is_file():
                OUTPUT.mkdir(parents=True, exist_ok=True)
                shutil.copy(drive_ck, CHECKPOINT)
                for sub in EMOTIONS:  # zsynchronizowane wcześniej kadry ściągamy lokalnie
                    src = DRIVE_OUT / sub
                    if src.is_dir():
                        (OUTPUT / sub).mkdir(parents=True, exist_ok=True)
                        for jpg in src.glob("*.jpg"):
                            shutil.copy(jpg, OUTPUT / sub / jpg.name)
                logger.info("Checkpoint odtworzony z Dysku.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("nie udało się wczytać checkpointu z Dysku: %s", exc)
    if CHECKPOINT.is_file():
        d = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
        logger.info("Checkpoint: %d kadrów, wznawiam", len(d.get("out_images", [])))
        return d
    return {"out_images": [], "out_anns": [], "lic_rows": {}, "counters": {},
            "stats": {}, "iid": 0, "aid": 0, "done_keys": []}


def _save_checkpoint(st: dict) -> None:
    CHECKPOINT.write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")
    try:  # kopia na Dysk — przeżyje rozłączenie środowiska
        DRIVE_OUT.mkdir(parents=True, exist_ok=True)
        shutil.copy(CHECKPOINT, DRIVE_OUT / "_checkpoint.json")
    except Exception:  # noqa: BLE001
        pass


def build() -> None:
    records = _load_records()
    idx = _index_videos()
    logger.info("Etykiet (%s): %d | plików wideo w folderach: %d",
                ",".join(sorted(EMOTIONS_DO)), len(records), len(set(idx.values())))

    device = DEVICE
    if device == "cuda":  # gdy limit GPU w Colab — schodzimy na CPU zamiast się wywalić
        try:
            import torch
            if not torch.cuda.is_available():
                device = "cpu"
                logger.warning("GPU niedostępne (limit Colab?) — liczę na CPU (wolniej).")
        except Exception:  # noqa: BLE001
            device = "cpu"
    pipeline = InferencePipeline(PipelineConfig(device=device))
    pipeline.load()
    logger.info("Modele załadowane (%s).", device)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for e in EMOTIONS:
        (OUTPUT / e).mkdir(exist_ok=True)

    st = _load_checkpoint()
    out_images, out_anns, lic_rows = st["out_images"], st["out_anns"], st["lic_rows"]
    counters, stats = Counter(st["counters"]), Counter(st["stats"])
    iid, aid = st["iid"], st["aid"]
    done_keys = set(st["done_keys"])

    for n, rec in enumerate(records, 1):
        key = rec["name"]
        if key in done_keys:
            continue
        processed = False  # tylko realnie przetworzone (odczytane) oznaczamy jako zrobione
        try:
            src = idx.get(rec["name"]) or idx.get(Path(rec["name"]).stem if rec["name"] else "")
            if src is None:
                stats["brak_w_indeksie"] += 1
                logger.warning("  ? brak w folderach: %s", rec["name"])
            else:
                path = _fetch_local(src, Path(rec["name"]).stem)
                if path is None:
                    stats["brak_odczytu"] += 1
                    logger.warning("  ? nie odczytano (ZIP/uszkodzone?): %s", rec["name"])
                else:
                    processed = True
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
                                      for name, a in extractor.extract(
                                          np.array(kp, dtype=float)).items()}
                                h, w = frame.shape[:2]
                                fname = f"{emotion}/{base}_{1 if role == 'start' else 2}.jpg"
                                cv2.imwrite(str(OUTPUT / fname), frame)
                                iid += 1
                                aid += 1
                                x, y, bw, bh = ann.bbox
                                out_images.append({"id": iid, "file_name": fname, "width": w,
                                                   "height": h, "source_video": rec["name"]})
                                out_anns.append({"id": aid, "image_id": iid, "category_id": 1,
                                                 "bbox": [x, y, bw, bh], "area": bw * bh,
                                                 "iscrowd": 0, "keypoints": kp,
                                                 "num_keypoints": ann.keypoints.num_detected,
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
                        path.unlink(missing_ok=True)  # lokalna kopia już niepotrzebna
        except Exception as exc:  # noqa: BLE001
            logger.warning("  ! błąd %s: %s", rec["name"], exc)
            stats["blad"] += 1
            processed = False
        if processed:
            done_keys.add(key)
        if n % 10 == 0:
            logger.info("  ... %d/%d | kadrów %d | %s", n, len(records),
                        len(out_images), dict(stats))
        if n % 50 == 0:
            _save_checkpoint({"out_images": out_images, "out_anns": out_anns,
                              "lic_rows": lic_rows, "counters": dict(counters),
                              "stats": dict(stats), "iid": iid, "aid": aid,
                              "done_keys": sorted(done_keys)})
        if n % 100 == 0:
            _sync_to_drive()  # co jakiś czas zrzucamy wynik na Dysk (na wypadek zerwania)

    _write(out_images, out_anns, lic_rows, counters, stats)
    CHECKPOINT.unlink(missing_ok=True)
    _sync_to_drive()


def _sync_to_drive() -> None:
    """Kopiuje lokalny wynik (OUTPUT) na Dysk (DRIVE_OUT). Nie wywraca biegu, gdy FUSE padnie."""
    try:
        DRIVE_OUT.mkdir(parents=True, exist_ok=True)
        for item in OUTPUT.rglob("*"):
            rel = item.relative_to(OUTPUT)
            target = DRIVE_OUT / rel
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
        logger.info("  ~ zsynchronizowano wynik na Dysk: %s", DRIVE_OUT)
    except Exception as exc:  # noqa: BLE001
        logger.warning("  ~ sync na Dysk nieudany (%s) — wynik jest lokalnie w %s",
                       exc, OUTPUT)


def _write(out_images, out_anns, lic_rows, counters, stats) -> None:
    info = {"description": "Dog FACS Dataset (Colab)", "contributor": "Politechnika Gdańska WETI",
            "date_created": datetime.now(timezone.utc).strftime("%Y-%m-%d")}
    categories = [{"id": 1, "name": "dog", "supercategory": "animal",
                   "keypoints": [], "skeleton": []}]
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
        w = csv.DictWriter(h, fieldnames=["source_video", "emotion", "frame_role",
                                          "frame_time_s", "file_name"])
        w.writeheader()
        rows = [{"source_video": a["source_video"], "emotion": a["emotion"],
                 "frame_role": a["frame_role"], "frame_time_s": a["frame_time_s"],
                 "file_name": imgs_by_id[a["image_id"]]["file_name"]} for a in out_anns]
        w.writerows(sorted(rows, key=lambda r: (r["emotion"], r["file_name"])))

    logger.info("=== GOTOWE: %s ===", OUTPUT)
    for e in EMOTIONS:
        logger.info("  %-9s %3d wideo (%d kadrów)", e, counters.get(e, 0),
                    len(per_class[e]["images"]))
    logger.info("Kadrów: %d | statusy: %s", len(out_images), dict(stats))


if __name__ == "__main__":
    build()
