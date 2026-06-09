#!/usr/bin/env python3
"""Wizualizacja anotacji COCO na klatce (podgląd wyniku)."""

import json
from pathlib import Path

import cv2

d = json.load(open("data/annotations/annotations.json", encoding="utf-8"))
img_rel = Path(d["images"][0]["file_name"])
ann = d["annotations"][0]
img = cv2.imread(str(Path("data/frames") / img_rel))

# bbox
x, y, w, h = ann["bbox"]
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# keypoints (format [x, y, v, ...])
kp = ann["keypoints"]
for i in range(0, len(kp), 3):
    px, py, v = kp[i], kp[i + 1], kp[i + 2]
    if v > 0:
        cv2.circle(img, (int(px), int(py)), 2, (0, 165, 255), -1)

# etykieta: emocja + rasa
emo = ann.get("emotion_name") or ann.get("emotion")
breed = ann.get("breed_name") or ann.get("breed")
conf = ann["confidence"].get("emotion", 0)
label = f"{emo} ({conf:.2f}) | {breed}"
cv2.rectangle(img, (x, y - 22), (x + max(190, 7 * len(label)), y), (0, 255, 0), -1)
cv2.putText(img, label, (x + 3, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

# top-5 AU wg |wartości|
au = ann.get("au_analysis", {})
top = sorted(au.items(), key=lambda kv: -abs(kv[1]))[:5]
for j, (name, val) in enumerate(top):
    txt = f"{name}: {val:+.2f}"
    cv2.putText(img, txt, (8, 22 + j * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, txt, (8, 22 + j * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

out = "data/annotation_preview.png"
cv2.imwrite(out, img)
print("saved", out, img.shape)
print(f"emotion={emo} breed={breed} keypoints={ann['num_keypoints']} AU={len(au)}")
