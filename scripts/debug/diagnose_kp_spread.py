"""По-кадровый замер горизонтального разброса парных keypoints.

Отличает настоящий коллапс лево/право (баг HorizontalFlip без перестановки пар)
от артефакта усреднения по мордам разной ориентации.

На КАЖДОЙ морде печатает горизонтальное расстояние (норм. к ширине морды) между:
- кончиками ушей ch6/ch7  (ожид. широко: ~0.4-0.7 на фронтальной морде)
- внешними углами глаз ch18/ch19 (ожид. ~0.2-0.4)
Если на всех мордах эти расстояния ~0 -> модель схлопывает пары к центру.
"""
import json
import subprocess

import cv2
import numpy as np

from packages.models.keypoints import KeypointsConfig, KeypointsModel
from packages.pipeline import InferencePipeline, PipelineConfig


def curl(url):
    return subprocess.run(["curl", "-s", "-m", "30", url],
                          capture_output=True, text=True).stdout


def curlb(url, fn):
    subprocess.run(["curl", "-s", "-m", "30", "-o", fn, url])


pipe = InferencePipeline(PipelineConfig(device="cpu", keypoints_two_pass=True))
pipe.load()
km = KeypointsModel(KeypointsConfig(
    weights_path="models/keypoints_dogflw.pt", device="cpu", use_tta=False))
km.load()

import os
os.makedirs("data/tmpl", exist_ok=True)
breeds = ["pug", "boxer", "bulldog/french", "rottweiler", "beagle",
          "doberman", "bullterrier", "pointer/german"]

print(f"{'face':24s} {'ear_dx':>7s} {'eye_dx':>7s} {'medvis':>7s}")
ear_dxs, eye_dxs = [], []
n = 0
for b in breeds:
    for k in range(3):
        try:
            r = json.loads(curl(f"https://dog.ceo/api/breed/{b}/images/random"))
            if r.get("status") != "success":
                continue
            fn = f"data/tmpl/spread_{b.replace('/', '_')}_{k}.jpg"
            curlb(r["message"], fn)
            img = cv2.imread(fn)
            if img is None:
                continue
            dets = pipe.bbox_model.predict(img)
            if not dets:
                continue
            x, y, w, h = dets[0].bbox
            c1, ox1, oy1 = pipe._square_crop(img, x, y, w, h, 1.1)
            p1 = km.predict(c1)
            fx, fy, fw, fh = pipe._keypoints_face_region(p1.keypoints)
            c2, ox2, oy2 = pipe._square_crop(
                img, int(fx + ox1), int(fy + oy1), int(fw), int(fh), 1.6)
            if c2.size == 0 or min(c2.shape[:2]) <= 10:
                continue
            H, W = c2.shape[:2]
            p2 = km.predict(c2).keypoints
            vis = np.median([kp.visibility for kp in p2])
            if vis < 0.2:
                continue
            ear_dx = abs(p2[6].x - p2[7].x) / W
            eye_dx = abs(p2[18].x - p2[19].x) / W
            ear_dxs.append(ear_dx)
            eye_dxs.append(eye_dx)
            n += 1
            print(f"{b+'_'+str(k):24s} {ear_dx:7.3f} {eye_dx:7.3f} {vis:7.2f}")
        except Exception:
            pass

print(f"\nN={n}")
print(f"ear_tip_dx:  mean={np.mean(ear_dxs):.3f} max={np.max(ear_dxs):.3f}")
print(f"eye_outer_dx: mean={np.mean(eye_dxs):.3f} max={np.max(eye_dxs):.3f}")
print("\nИнтерпретация: для нормальной фронтальной модели ear_dx ~0.4-0.7,")
print("eye_dx ~0.2-0.4. Значения ~0 на ВСЕХ мордах = коллапс пар (баг flip).")
