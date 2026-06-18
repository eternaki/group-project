#!/usr/bin/env python3
"""
Trening detektora MORDY psa (YOLOv8n) na DogFLW — lokalnie na CPU.

DogFLW to pełne sceny z ramką mordy (bbox liczony z 46 landmarków + margines).
Detektor zwraca region mordy → pipeline kadruje go pod keypoints (zamiast
zawodnego two-pass z bboxa całego ciała).

Użycie:
    python scripts/training/train_dogface_detector.py \
        --dogflw_path data/dogflw_raw --epochs 40 --imgsz 512
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np


def find_root(base: str) -> str | None:
    """Znajduje katalog z train/labels."""
    for d, subs, _ in os.walk(base):
        if "labels" in subs and "images" in subs and os.path.basename(d) == "train":
            return os.path.dirname(d)
    # fallback: szukaj dowolnego train/labels
    for d, subs, _ in os.walk(base):
        if os.path.isdir(os.path.join(d, "train", "labels")):
            return d
    return None


def build_yolo_dataset(root: str, out: str, margin: float = 0.08) -> str:
    """Buduje dataset YOLO (1 klasa: dogface) z ramek liczonych z landmarków."""
    for split, sub in [("train", "train"), ("test", "val")]:
        idir = os.path.join(root, split, "images")
        ldir = os.path.join(root, split, "labels")
        oi = os.path.join(out, sub, "images")
        ol = os.path.join(out, sub, "labels")
        os.makedirs(oi, exist_ok=True)
        os.makedirs(ol, exist_ok=True)
        n = 0
        for lf in os.listdir(ldir):
            if not lf.endswith(".json"):
                continue
            name = lf[:-5]
            ip = next(
                (os.path.join(idir, name + e)
                 for e in (".jpg", ".jpeg", ".png", ".JPEG")
                 if os.path.exists(os.path.join(idir, name + e))),
                None,
            )
            if ip is None:
                continue
            try:
                lm = np.array(json.load(open(os.path.join(ldir, lf)))["landmarks"], float)
                im = cv2.imread(ip)
                h, w = im.shape[:2]
                x0, y0 = lm[:, 0].min(), lm[:, 1].min()
                x1, y1 = lm[:, 0].max(), lm[:, 1].max()
                mw, mh = (x1 - x0) * margin, (y1 - y0) * margin
                x0, y0 = max(0, x0 - mw), max(0, y0 - mh)
                x1, y1 = min(w, x1 + mw), min(h, y1 + mh)
                cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
                bw, bh = (x1 - x0) / w, (y1 - y0) / h
                shutil.copy(ip, os.path.join(oi, name + ".jpg"))
                with open(os.path.join(ol, name + ".txt"), "w") as f:
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                n += 1
            except Exception:
                pass
        print(f"{sub}: {n} próbek")

    yaml_path = os.path.join(out, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(
            f"path: {os.path.abspath(out)}\n"
            "train: train/images\nval: val/images\nnc: 1\nnames: [dogface]\n"
        )
    return yaml_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dogflw_path", default="data/dogflw_raw")
    p.add_argument("--out", default="data/dogface_yolo")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--model_out", default="models/dogface_yolo.pt")
    args = p.parse_args()

    root = find_root(args.dogflw_path)
    print(f"DogFLW root: {root}")
    if not root:
        raise SystemExit("Nie znaleziono DogFLW (train/labels) w " + args.dogflw_path)

    yaml_path = build_yolo_dataset(root, args.out)

    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device="cpu",
        project="data/dogface_runs",
        name="train",
        exist_ok=True,
        verbose=True,
        cache=False,  # bez RAM-cache (maszyna używana do testów webapp)
        workers=4,
    )
    m = model.val()
    print(f"mAP50={m.box.map50:.4f} mAP50-95={m.box.map:.4f}")

    best = "data/dogface_runs/train/weights/best.pt"
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    if os.path.exists(best):
        shutil.copy(best, args.model_out)
        print(f"Zapisano detektor: {args.model_out}")
    else:
        print("best.pt nie znaleziono")


if __name__ == "__main__":
    main()
