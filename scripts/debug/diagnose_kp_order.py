"""Диагностика: совпадает ли порядок каналов модели keypoints со schemas.py.

Строит усреднённый нормализованный шаблон позиций 46 каналов по многим
фронтальным мордам (TTA выключен -> сырой порядок каналов модели) и сравнивает
фактическое расположение групп с семантикой, объявленной в schemas.py.

Запуск:
    .venv/Scripts/python.exe scripts/debug/diagnose_kp_order.py
"""
import json
import subprocess

import cv2
import numpy as np

from packages.data.schemas import KEYPOINT_NAMES, NUM_KEYPOINTS
from packages.models.keypoints import KeypointsConfig, KeypointsModel
from packages.pipeline import InferencePipeline, PipelineConfig


def curl(url: str) -> str:
    return subprocess.run(
        ["curl", "-s", "-m", "30", url], capture_output=True, text=True
    ).stdout


def curlb(url: str, fn: str) -> None:
    subprocess.run(["curl", "-s", "-m", "30", "-o", fn, url])


pipe = InferencePipeline(PipelineConfig(device="cpu", keypoints_two_pass=True))
pipe.load()
km = KeypointsModel(
    KeypointsConfig(
        weights_path="models/keypoints_dogflw.pt", device="cpu", use_tta=False
    )
)
km.load()

breeds = [
    "pug", "beagle", "boxer", "labrador", "bulldog/french", "retriever/golden",
    "chihuahua", "pomeranian", "husky", "rottweiler", "spaniel/cocker",
    "dalmatian", "doberman", "bullterrier", "pointer/german", "weimaraner",
]

import os
os.makedirs("data/tmpl", exist_ok=True)
acc = np.zeros((NUM_KEYPOINTS, 2))
accv = np.zeros(NUM_KEYPOINTS)
n_used = 0

for b in breeds:
    for k in range(3):
        try:
            r = json.loads(curl(f"https://dog.ceo/api/breed/{b}/images/random"))
            if r.get("status") != "success":
                continue
            fn = f"data/tmpl/{b.replace('/', '_')}_{k}.jpg"
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
                img, int(fx + ox1), int(fy + oy1), int(fw), int(fh), 1.6
            )
            if c2.size == 0 or min(c2.shape[:2]) <= 10:
                continue
            H, W = c2.shape[:2]
            p2 = km.predict(c2)
            vis = np.array([kp.visibility for kp in p2.keypoints])
            if np.median(vis) < 0.2:
                continue
            for i, kp in enumerate(p2.keypoints):
                wgt = max(kp.visibility, 0.0)
                acc[i, 0] += wgt * (kp.x / W)
                acc[i, 1] += wgt * (kp.y / H)
                accv[i] += wgt
            n_used += 1
        except Exception:
            pass

print(f"USED_FACES={n_used}")
if n_used < 5:
    print("Недостаточно морд для надёжного шаблона.")
    raise SystemExit(1)

mean = np.zeros((NUM_KEYPOINTS, 2))
for i in range(NUM_KEYPOINTS):
    if accv[i] > 0:
        mean[i] = acc[i] / accv[i]

json.dump(mean.tolist(), open("data/kp_template.json", "w"))

# Рендер шаблона с номерами каналов
S = 700
canvas = np.full((S, S, 3), 255, np.uint8)
for i in range(NUM_KEYPOINTS):
    px, py = int(mean[i, 0] * S), int(mean[i, 1] * S)
    cv2.circle(canvas, (px, py), 4, (0, 0, 200), -1)
    cv2.putText(canvas, str(i), (px + 3, py - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
cv2.imwrite("data/kp_template.png", canvas)


def region(y: float) -> str:
    if y < 0.33:
        return "TOP"
    if y < 0.66:
        return "MID"
    return "BOT"


# Группы по schemas.py и где они ДОЛЖНЫ быть на фронтальной морде
groups = {
    "uszy 0-13 (ОЖИД: TOP)": range(0, 14),
    "brwi 14-15 (ОЖИД: верх-MID)": range(14, 16),
    "oczy 16-23 (ОЖИД: верх-MID)": range(16, 24),
    "nos/pysk 24-37 (ОЖИД: MID-центр)": range(24, 38),
    "wargi/podbr. 38-45 (ОЖИД: BOT)": range(38, 46),
}
print("\n=== Фактическое усреднённое положение групп (по schemas.py) ===")
for label, idxs in groups.items():
    ys = [mean[i, 1] for i in idxs]
    print(f"{label:38s} mean_y={np.mean(ys):.2f}  -> {region(np.mean(ys))}")

print("\n=== Отдельные каналы (норм. x,y внутри морды) ===")
for i in range(NUM_KEYPOINTS):
    print(f"ch{i:2d} ({KEYPOINT_NAMES[i]:22s}) x={mean[i,0]:.2f} y={mean[i,1]:.2f} {region(mean[i,1])}")

print("\nСохранено: data/kp_template.png, data/kp_template.json")
