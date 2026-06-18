"""Weryfikacja pipeline na najlepszym modelu keypoints.

Sprawdza: konfigurację modelu, liczbę i nazwy WSZYSTKICH 46 keypoints,
renderuje obraz z numerami punktów. Pobiera czołową mordę psa (lub bierze
ścieżkę z argumentu).

Użycie: python scripts/debug/verify_keypoints_pipeline.py [ścieżka_obrazu]
"""
import json
import subprocess
import sys

import cv2

from packages.data.schemas import KEYPOINT_NAMES, NUM_KEYPOINTS
from packages.pipeline import InferencePipeline, PipelineConfig


def fetch_frontal_dog(fn: str) -> str | None:
    for breed in ["pug", "boxer", "bulldog/french", "beagle"]:
        try:
            r = json.loads(
                subprocess.run(
                    ["curl", "-s", "-m", "30",
                     f"https://dog.ceo/api/breed/{breed}/images/random"],
                    capture_output=True, text=True,
                ).stdout
            )
            if r.get("status") == "success":
                subprocess.run(["curl", "-s", "-m", "30", "-o", fn, r["message"]])
                if cv2.imread(fn) is not None:
                    return fn
        except Exception:
            pass
    return None


def main() -> int:
    cfg = PipelineConfig(device="cpu")
    print("=== KONFIGURACJA KEYPOINTS ===")
    print(f"  wagi:        {cfg.keypoints_weights}")
    print(f"  two_pass:    {cfg.keypoints_two_pass}")
    print(f"  TTA skale:   {cfg.keypoints_tta_expands}")

    pipe = InferencePipeline(cfg)
    pipe.load()
    kcfg = pipe.keypoints_model.config
    print(f"  backbone:    {kcfg.model_name}")
    print(f"  img_size:    {kcfg.img_size} | heatmap: {kcfg.heatmap_size}")
    print(f"  use_tta:     {kcfg.use_tta} | use_dark: {kcfg.use_dark}")
    print(f"  NUM_KEYPOINTS (schemas): {NUM_KEYPOINTS}")

    src = sys.argv[1] if len(sys.argv) > 1 else fetch_frontal_dog("data/tmpl/verify.jpg")
    if not src:
        print("Nie udało się pobrać obrazu")
        return 1
    img = cv2.imread(src)
    res = pipe.process_frame(img, frame_id=0)
    if not res.annotations or res.annotations[0].keypoints is None:
        print("Brak detekcji psa / keypoints")
        return 1

    ann = res.annotations[0]
    kps = ann.keypoints.keypoints
    print(f"\n=== KEYPOINTS: zwrócono {len(kps)} / oczekiwano {NUM_KEYPOINTS} ===")
    thr = kcfg.confidence_threshold
    vis_cnt = sum(1 for k in kps if k.visibility > thr)
    print(f"powyżej progu {thr}: {vis_cnt}/{len(kps)} | num_detected={ann.keypoints.num_detected}")
    print(f"\n{'idx':>3} {'nazwa':24s} {'x':>7} {'y':>7} {'vis':>5}")
    for i, k in enumerate(kps):
        name = KEYPOINT_NAMES[i] if i < len(KEYPOINT_NAMES) else "?"
        print(f"{i:>3} {name:24s} {k.x:7.1f} {k.y:7.1f} {k.visibility:5.2f}")

    # render z numerami
    x, y, w, h = ann.bbox
    pad = int(0.25 * max(w, h))
    H, W = img.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    crop = img[y0:y1, x0:x1].copy()
    sc = max(1, int(700 / max(crop.shape[:2])))
    crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
    for i, k in enumerate(kps):
        cx, cy = int((k.x - x0) * sc), int((k.y - y0) * sc)
        col = (0, 165, 255) if k.visibility > thr else (160, 160, 160)
        cv2.circle(crop, (cx, cy), 3, col, -1)
        cv2.putText(crop, str(i), (cx + 3, cy - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
    out = "data/kp_results/verify_full_labeled.png"
    cv2.imwrite(out, crop)
    print(f"\nzapisano render z numerami: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
