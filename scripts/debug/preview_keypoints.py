#!/usr/bin/env python3
"""
Podgląd jakości keypoints: bierze obraz LUB wideo, znajduje najlepszą klatkę
z psem, wykrywa 46 keypoints i renderuje krupny plan mordy ze szkieletem.

Użycie:
    python scripts/debug/preview_keypoints.py <ścieżka> [nazwa_wyjścia]
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from packages.data.schemas import SKELETON_CONNECTIONS
from packages.pipeline import InferencePipeline, PipelineConfig

VIDEO_EXT = {".mp4", ".webm", ".mkv", ".mov", ".avi", ".MP4", ".MOV"}


def sample_video_frames(path: Path, n: int = 12) -> list[np.ndarray]:
    """Pobiera n równomiernie rozłożonych klatek z wideo."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = [int(total * (i + 1) / (n + 1)) for i in range(n)]
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    return frames


def best_dog_frame(pipeline: InferencePipeline, frames: list[np.ndarray]):
    """Wybiera klatkę z najpewniejszą detekcją + keypoints."""
    best = None
    for fr in frames:
        res = pipeline.process_frame(fr, frame_id=0)
        if not res.annotations:
            continue
        ann = res.annotations[0]
        score = ann.bbox_confidence + (ann.keypoints.confidence if ann.keypoints else 0)
        if best is None or score > best[0]:
            best = (score, fr, ann)
    return best


def render(frame: np.ndarray, ann, out_path: str) -> None:
    """Rysuje krupny plan mordy z keypoints i szkieletem."""
    x, y, w, h = ann.bbox
    pad = int(0.2 * max(w, h))
    H, W = frame.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    crop = frame[y0:y1, x0:x1].copy()

    scale = max(1, int(800 / max(crop.shape[:2])))
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Keypoints w układzie pełnego obrazu -> crop
    pts = []
    for kp in ann.keypoints.keypoints:
        cx, cy = int((kp.x - x0) * scale), int((kp.y - y0) * scale)
        pts.append((cx, cy, kp.visibility))

    # Szkielet (0-indeksowane pary w SKELETON_CONNECTIONS)
    for a, b in SKELETON_CONNECTIONS:
        if 0 <= a < len(pts) and 0 <= b < len(pts):
            ax, ay, av = pts[a]
            bx, by, bv = pts[b]
            if av > 0 and bv > 0:
                cv2.line(crop, (ax, ay), (bx, by), (0, 220, 0), 1, cv2.LINE_AA)

    # Punkty
    for cx, cy, v in pts:
        if v > 0:
            cv2.circle(crop, (cx, cy), 4, (0, 165, 255), -1)
            cv2.circle(crop, (cx, cy), 4, (0, 0, 0), 1)

    # Etykieta
    breed = ann.breed.class_name if ann.breed else "?"
    bconf = ann.breed.confidence if ann.breed else 0
    nkp = ann.keypoints.num_detected
    kconf = ann.keypoints.confidence
    txt = f"{breed} ({bconf:.2f}) | kp {nkp}/46 conf {kconf:.2f}"
    cv2.rectangle(crop, (0, 0), (crop.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(crop, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, crop)
    print(f"saved {out_path} {crop.shape} | bbox={ann.bbox} conf={ann.bbox_confidence:.2f} "
          f"| {breed} {bconf:.2f} | kp {nkp}/46 conf {kconf:.2f}")


def main() -> int:
    src = Path(sys.argv[1])
    out = sys.argv[2] if len(sys.argv) > 2 else f"data/preview_{src.stem}.png"

    pipeline = InferencePipeline(PipelineConfig(device="cpu"))
    pipeline.load()

    if src.suffix in VIDEO_EXT:
        frames = sample_video_frames(src)
        print(f"Wideo: {src.name} -> {len(frames)} próbek klatek")
        best = best_dog_frame(pipeline, frames)
        if best is None:
            print("Nie wykryto psa w żadnej klatce")
            return 1
        _, frame, ann = best
    else:
        frame = cv2.imread(str(src))
        if frame is None:
            print(f"Nie można wczytać: {src}")
            return 1
        res = pipeline.process_frame(frame, frame_id=0)
        if res.annotations:
            ann = res.annotations[0]
        else:
            # Fallback: obraz to prawdopodobnie gotowy crop mordy — uruchom
            # keypoints/rasę bezpośrednio na całym obrazie (bez detekcji bbox).
            print("Brak detekcji psa — keypoints bezpośrednio na całym obrazie (crop mordy)")
            from packages.pipeline.inference import DogAnnotation

            h, w = frame.shape[:2]
            ann = DogAnnotation(dog_id=0, bbox=(0, 0, w, h), bbox_confidence=0.0)
            ann.keypoints = pipeline.keypoints_model.predict(frame)
            if pipeline.breed_model is not None:
                ann.breed = pipeline.breed_model.predict(frame)

    if ann.keypoints is None:
        print("Brak keypoints")
        return 1
    render(frame, ann, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
