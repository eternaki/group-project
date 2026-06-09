#!/usr/bin/env python3
"""
EKSPERYMENT: porównanie jakości keypoints.
  (A) baseline       — predykcja na crop bbox (obecne zachowanie, Resize sztaucha)
  (B) square         — crop bbox dopełniony do kwadratu (bez zniekształcenia proporcji)
  (C) two-pass face  — pass1 -> region mordy z keypoints -> kwadratowy crop -> pass2

Zapisuje obraz porównawczy obok siebie.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from packages.pipeline import InferencePipeline, PipelineConfig


def square_crop(img, x, y, w, h, expand=1.0):
    """Wytnij kwadratowy region (cx,cy, bok) z marginesem, w granicach obrazu."""
    H, W = img.shape[:2]
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * expand
    x0 = int(max(0, cx - side / 2))
    y0 = int(max(0, cy - side / 2))
    x1 = int(min(W, cx + side / 2))
    y1 = int(min(H, cy + side / 2))
    return img[y0:y1, x0:x1], x0, y0


def kp_face_region(kps, conf_min):
    """Bbox pewnych keypoints (region mordy) w układzie wejścia predykcji."""
    pts = [(kp.x, kp.y) for kp in kps if kp.visibility >= conf_min]
    if len(pts) < 5:
        pts = [(kp.x, kp.y) for kp in kps]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return x0, y0, x1 - x0, y1 - y0


def draw(crop, kps, scale, ox, oy, title):
    """Narysuj keypoints (w układzie oryginału) na wyskalowanym cropie."""
    vis = crop.copy()
    visn = sum(1 for kp in kps if kp.visibility > 0.15)
    conf = float(np.mean([kp.visibility for kp in kps]))
    for kp in kps:
        if kp.visibility > 0.10:
            px, py = int((kp.x - ox) * scale), int((kp.y - oy) * scale)
            r = 5 if kp.visibility > 0.3 else 3
            cv2.circle(vis, (px, py), r, (0, 165, 255), -1)
            cv2.circle(vis, (px, py), r, (0, 0, 0), 1)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(vis, f"{title}  kp {visn}/46 conf {conf:.2f}", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def main():
    src = Path(sys.argv[1])
    pipe = InferencePipeline(PipelineConfig(device="cpu"))
    pipe.load()
    km = pipe.keypoints_model

    img = cv2.imread(str(src))
    H, W = img.shape[:2]

    # bbox: z detektora lub cały obraz
    dets = pipe.bbox_model.predict(img)
    if dets:
        x, y, w, h = dets[0].bbox
        x, y = max(0, x), max(0, y)
        w, h = min(w, W - x), min(h, H - y)
    else:
        x, y, w, h = 0, 0, W, H

    panels = []
    target = 360  # rozmiar panelu

    # (A) baseline: crop bbox bezpośrednio (jak teraz)
    cropA = img[y:y + h, x:x + w]
    predA = km.predict(cropA)  # keypoints w układzie cropA
    sA = target / max(cropA.shape[:2])
    panels.append(draw(cv2.resize(cropA, None, fx=sA, fy=sA), predA.keypoints, sA, 0, 0, "A baseline"))

    # (B) square: bbox dopełniony do kwadratu
    cropB, bx, by = square_crop(img, x, y, w, h, expand=1.0)
    predB = km.predict(cropB)
    sB = target / max(cropB.shape[:2])
    panels.append(draw(cv2.resize(cropB, None, fx=sB, fy=sB), predB.keypoints, sB, 0, 0, "B square"))

    # (C) two-pass: region mordy z pass1 (na square) -> kwadratowy crop -> pass2
    fx, fy, fw, fh = kp_face_region(predB.keypoints, conf_min=0.2)
    # przelicz region z układu cropB na oryginał
    cropC, cx0, cy0 = square_crop(img, bx + fx, by + fy, fw, fh, expand=1.6)
    if cropC.size > 0 and min(cropC.shape[:2]) > 10:
        predC = km.predict(cropC)
        sC = target / max(cropC.shape[:2])
        panels.append(draw(cv2.resize(cropC, None, fx=sC, fy=sC), predC.keypoints, sC, 0, 0, "C two-pass"))

    # Zrównaj wysokości i sklej poziomo
    hmax = max(p.shape[0] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, hmax - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40)) for p in panels]
    combo = np.hstack(panels)
    out = f"data/exp_{src.stem}.png"
    cv2.imwrite(out, combo)
    print(f"saved {out} {combo.shape}")
    print(f"A conf {np.mean([k.visibility for k in predA.keypoints]):.3f} | "
          f"B conf {np.mean([k.visibility for k in predB.keypoints]):.3f}", end="")
    if 'predC' in dir():
        print(f" | C conf {np.mean([k.visibility for k in predC.keypoints]):.3f}")
    else:
        print()


if __name__ == "__main__":
    main()
