#!/usr/bin/env python3
"""Krupny plan mordy psa z 46 keypoints i szkieletem (podgląd wyniku)."""

import json
from pathlib import Path

import cv2

d = json.load(open("data/annotations/annotations.json", encoding="utf-8"))
img_rel = Path(d["images"][0]["file_name"])
ann = d["annotations"][0]
cat = d["categories"][0]
kp_names = cat.get("keypoints", [])
skeleton = cat.get("skeleton", [])

img = cv2.imread(str(Path("data/frames") / img_rel))

# Wytnij obszar bbox z marginesem
x, y, w, h = ann["bbox"]
pad = int(0.25 * max(w, h))
H, W = img.shape[:2]
x0, y0 = max(0, x - pad), max(0, y - pad)
x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
crop = img[y0:y1, x0:x1]

# Upscale do czytelnego rozmiaru (docelowo ~700px na dłuższym boku)
scale = max(1, int(700 / max(crop.shape[:2])))
crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

# Keypoints (pełny obraz) -> układ cropa
kp = ann["keypoints"]
pts = []
for i in range(0, len(kp), 3):
    px, py, v = kp[i], kp[i + 1], kp[i + 2]
    cx, cy = int((px - x0) * scale), int((py - y0) * scale)
    pts.append((cx, cy, v))

# Szkielet (połączenia 1-indeksowane)
for a, b in skeleton:
    if 1 <= a <= len(pts) and 1 <= b <= len(pts):
        ax, ay, av = pts[a - 1]
        bx, by, bv = pts[b - 1]
        if av > 0 and bv > 0:
            cv2.line(crop, (ax, ay), (bx, by), (0, 200, 0), 1, cv2.LINE_AA)

# Punkty + numery
for idx, (cx, cy, v) in enumerate(pts):
    if v > 0:
        cv2.circle(crop, (cx, cy), 4, (0, 165, 255), -1)
        cv2.circle(crop, (cx, cy), 4, (0, 0, 0), 1)
        cv2.putText(crop, str(idx + 1), (cx + 4, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(crop, str(idx + 1), (cx + 4, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1, cv2.LINE_AA)

out = "data/face_keypoints_preview.png"
cv2.imwrite(out, crop)
print("saved", out, crop.shape, "| scale x", scale)
print("keypoints:", ann["num_keypoints"], "| skeleton links:", len(skeleton))
print("breed:", ann.get("breed"), "| emotion:", ann.get("emotion"))
