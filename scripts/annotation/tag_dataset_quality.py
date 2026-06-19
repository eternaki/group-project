"""
Tagowanie jakości zestawu danych COCO (filtry automatyczne, bez udziału człowieka).

Dla każdej anotacji liczy z keypoints: średnią pewność, kąt głowy (yaw/pitch),
spójność anatomiczną oraz "strict_ok" (czysta frontalna morda). Zapisuje wzbogacony
COCO + czysty podzbiór (strict_ok) + statystyki.

Użycie:
    python scripts/annotation/tag_dataset_quality.py \
        --coco data/dataset_final/annotations/annotations.json \
        --out-dir data/dataset_final
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from packages.data.schemas import KP, NUM_KEYPOINTS
from packages.models.head_pose import estimate_head_pose

# Grupy keypoints do oceny regionalnej
_L_EYE = [KP.LEFT_EYE_INNER, KP.LEFT_EYE_OUTER, KP.LEFT_EYE_TOP, KP.LEFT_EYE_BOTTOM]
_R_EYE = [KP.RIGHT_EYE_INNER, KP.RIGHT_EYE_OUTER, KP.RIGHT_EYE_TOP, KP.RIGHT_EYE_BOTTOM]
_NOSE = [KP.NOSE_LEFT_WING, KP.NOSE_RIGHT_WING, KP.NOSE_TIP]
_MOUTH = [KP.MOUTH_LEFT_CORNER, KP.MOUTH_RIGHT_CORNER]


def _anatomy_ok(k: np.ndarray) -> bool:
    """Spójność geometrii twarzy: porządek oczy<nos<pysk wzdłuż osi twarzy."""
    eye_l = (k[KP.LEFT_EYE_INNER, :2] + k[KP.LEFT_EYE_OUTER, :2]) / 2
    eye_r = (k[KP.RIGHT_EYE_INNER, :2] + k[KP.RIGHT_EYE_OUTER, :2]) / 2
    eye_c = (eye_l + eye_r) / 2
    nose = k[KP.NOSE_TIP, :2]
    mouth = (k[KP.MOUTH_LEFT_CORNER, :2] + k[KP.MOUTH_RIGHT_CORNER, :2]) / 2
    eye_dist = float(np.linalg.norm(eye_l - eye_r))
    if eye_dist < 1e-3:
        return False
    axis = mouth - eye_c
    axis_len = float(np.linalg.norm(axis))
    if axis_len < 0.3 * eye_dist:
        return False
    axis = axis / axis_len
    nose_p = float(np.dot(nose - eye_c, axis))
    mouth_p = float(np.dot(mouth - eye_c, axis))
    return 0.05 * mouth_p < nose_p < 0.95 * mouth_p


def _quality(k: np.ndarray) -> dict:
    """Liczy metryki jakości jednej anotacji z keypoints (46,3)."""
    mean_conf = float(np.mean(k[:, 2]))
    hp = estimate_head_pose(k.flatten())
    cL, cR = float(np.mean(k[_L_EYE, 2])), float(np.mean(k[_R_EYE, 2]))
    cN, cM = float(np.mean(k[_NOSE, 2])), float(np.mean(k[_MOUTH, 2]))
    anat = _anatomy_ok(k)

    strict = (
        abs(hp.yaw) <= 20 and abs(hp.pitch) <= 25 and abs(hp.roll) <= 20
        and cL >= 0.55 and cR >= 0.55 and cN >= 0.5 and cM >= 0.5
        and abs(cL - cR) <= 0.22
        and anat
    )
    return {
        "mean_conf": round(mean_conf, 3),
        "yaw": round(hp.yaw, 1),
        "pitch": round(hp.pitch, 1),
        "eye_conf_l": round(cL, 3),
        "eye_conf_r": round(cR, 3),
        "anatomy_ok": bool(anat),
        "strict_ok": bool(strict),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    coco = json.loads(args.coco.read_text(encoding="utf-8"))
    anns = coco.get("annotations", [])
    print(f"Anotacji: {len(anns)}")

    strict_ok = 0
    anat_ok = 0
    for a in anns:
        kp = a.get("keypoints")
        if not kp or len(kp) != NUM_KEYPOINTS * 3:
            a["quality"] = {"strict_ok": False, "anatomy_ok": False, "note": "brak keypoints"}
            continue
        k = np.array(kp, dtype=np.float32).reshape(NUM_KEYPOINTS, 3)
        q = _quality(k)
        a["quality"] = q
        strict_ok += q["strict_ok"]
        anat_ok += q["anatomy_ok"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # wzbogacony COCO
    full_path = args.out_dir / "annotations_quality.json"
    full_path.write_text(json.dumps(coco, ensure_ascii=False, indent=1), encoding="utf-8")

    # czysty podzbiór (strict_ok)
    strict_ids = {a["image_id"] for a in anns if a.get("quality", {}).get("strict_ok")}
    clean = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco.get("categories", []),
        "images": [im for im in coco.get("images", []) if im["id"] in strict_ids],
        "annotations": [a for a in anns if a.get("quality", {}).get("strict_ok")],
    }
    clean_path = args.out_dir / "annotations_clean.json"
    clean_path.write_text(json.dumps(clean, ensure_ascii=False, indent=1), encoding="utf-8")

    emos = Counter(a.get("emotion") for a in clean["annotations"])
    breeds = Counter(a.get("breed") for a in clean["annotations"])
    print("\n==== STATYSTYKI ====")
    print(f"anotacje (peak frames):   {len(anns)}")
    print(f"anatomy_ok:               {anat_ok} ({100*anat_ok/max(1,len(anns)):.1f}%)")
    print(f"strict_ok (czysty zbiór): {strict_ok} ({100*strict_ok/max(1,len(anns)):.1f}%)")
    print(f"emocje (clean):           {dict(emos)}")
    print(f"rasy (clean, top5):       {dict(breeds.most_common(5))}")
    print(f"\nzapisano: {full_path.name} (pełny+quality), {clean_path.name} (strict)")


if __name__ == "__main__":
    main()
