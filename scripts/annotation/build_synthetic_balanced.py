#!/usr/bin/env python3
"""
SYNTETYCZNY zbiór zbalansowany — WYŁĄCZNIE do porównania w raporcie.

UWAGA: to NIE są prawdziwe etykiety. Spokojne wideo (neutral/relaxed) są tu
rozrzucone po emocjach do ~200/klasę, żeby pokazać, jak wygląda „naiwny balans".
Każda taka klatka ma `label_source=synthetic_filler` i zachowane `emotion_real`.
NIE ODDAWAĆ prowadzącemu jako prawdziwy zbiór — to baseline do porównania.
"""

import csv
import json
import shutil
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from scripts.annotation.build_structured import (
    CURATED,
    FRAMES,
    infer_license,
    human_emotions,
)

REPO = Path(__file__).resolve().parent.parent.parent
OUTPUT = REPO / "data" / "dataset_final" / "structured_synthetic"
EMOTIONS = ["sad", "happy", "surprise", "submission", "angry", "fearful", "pain"]
CALM = {"neutral", "relaxed"}
CAP = 200
FRAMES_PER_VIDEO = 4


def build() -> None:
    coco = json.loads(CURATED.read_text(encoding="utf-8"))
    images = {i["id"]: i for i in coco["images"]}
    peaks = [a for a in coco["annotations"] if a.get("frame_role") == "peak"]
    human = human_emotions()

    by_video: dict[str, list[dict]] = defaultdict(list)
    for a in peaks:
        v = images[a["image_id"]].get("source_video") or images[a["image_id"]]["file_name"].split("/")[1]
        by_video[v].append(a)

    real_emotion: dict[str, str] = {}
    for v, anns in by_video.items():
        hv = [human[images[a["image_id"]]["file_name"]] for a in anns if images[a["image_id"]]["file_name"] in human]
        if hv:
            real_emotion[v] = Counter(hv).most_common(1)[0][0]
        else:
            mv = [a.get("emotion") for a in anns if a.get("emotion")]
            real_emotion[v] = Counter(mv).most_common(1)[0][0] if mv else "neutral"

    # przydział do klas: najpierw prawdziwe emocje, potem spokojne jako filler
    assigned: dict[str, list[tuple[str, bool]]] = {e: [] for e in EMOTIONS}  # (video, is_filler)
    for v, e in real_emotion.items():
        if e in EMOTIONS and len(assigned[e]) < CAP:
            assigned[e].append((v, False))
    calm_pool = [v for v, e in real_emotion.items() if e in CALM]
    idx = 0
    for v in calm_pool:
        # round-robin do najmniej wypełnionych klas
        target = min(EMOTIONS, key=lambda e: len(assigned[e]))
        if len(assigned[target]) >= CAP:
            break
        assigned[target].append((v, True))
        idx += 1

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)

    out_images, out_anns, lic_rows = [], [], []
    seen_lic: set[str] = set()
    iid = aid = 0
    for emotion, videos in assigned.items():
        (OUTPUT / emotion).mkdir(exist_ok=True)
        for video, is_filler in videos:
            platform, lic, link = infer_license(video)
            if video not in seen_lic:
                lic_rows.append({"video_id": video, "platform": platform, "license": lic, "link": link})
                seen_lic.add(video)
            for a in by_video[video][:FRAMES_PER_VIDEO]:
                image = images[a["image_id"]]
                src = FRAMES / image["file_name"]
                if not src.is_file():
                    continue
                rand = uuid.uuid4().hex + ".jpg"
                shutil.copy2(src, OUTPUT / emotion / rand)
                iid += 1
                out_images.append({"id": iid, "file_name": f"{emotion}/{rand}", "source_video": video})
                aid += 1
                out_anns.append({
                    "id": aid, "image_id": iid, "category_id": 1,
                    "bbox": a.get("bbox"), "keypoints": a.get("keypoints"),
                    "au_analysis": a.get("au_analysis", {}),
                    "emotion": emotion,
                    "emotion_real": real_emotion[video],
                    "label_source": "synthetic_filler" if is_filler else "real",
                    "source_video": video,
                })

    coco_out = {
        "info": {
            "description": "SYNTHETIC balanced — DO PORÓWNANIA, NIE PRAWDZIWE ETYKIETY",
            "WARNING": "Spokojne wideo rozrzucone po emocjach. label_source=synthetic_filler to zmyślona etykieta. NIE ODDAWAĆ.",
            "date_created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        },
        "categories": [{"id": 1, "name": "dog"}],
        "images": out_images, "annotations": out_anns,
    }
    (OUTPUT / "annotations.json").write_text(json.dumps(coco_out, ensure_ascii=False), encoding="utf-8")
    with (OUTPUT / "licenses.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["video_id", "platform", "license", "link"]); w.writeheader(); w.writerows(lic_rows)
    (OUTPUT / "README.md").write_text(
        "# ⚠️ SYNTETYCZNY zbiór — TYLKO DO PORÓWNANIA\n\n"
        "Spokojne wideo (neutral/relaxed) rozrzucone po emocjach do ~200/klasę.\n"
        "`label_source=synthetic_filler` = ZMYŚLONA etykieta, `emotion_real` = prawdziwa.\n"
        "**NIE ODDAWAĆ prowadzącemu jako prawdziwy zbiór.** Baseline do raportu.\n",
        encoding="utf-8",
    )
    real = sum(1 for e in assigned for v, f in assigned[e] if not f)
    filler = sum(1 for e in assigned for v, f in assigned[e] if f)
    print("=== SYNTETYCZNY (do сравнения) ===")
    for e in EMOTIONS:
        r = sum(1 for v, f in assigned[e] if not f); fl = sum(1 for v, f in assigned[e] if f)
        print(f"  {e:11} {len(assigned[e]):4} видео  (реальных {r}, ФЕЙК-filler {fl})")
    print(f"итого видео: {real+filler} (реальных {real}, фейк {filler}), кадров: {len(out_images)}")


if __name__ == "__main__":
    build()
