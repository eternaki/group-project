#!/usr/bin/env python3
"""
Scala kilka wyników obróbki (release_*) w jeden zbiór COCO.

Po co: obróbkę można rozbić na kilka maszyn (my liczymy happy/sad na Colabie,
druga osoba neutral/angry/surprise/fearful na swoim GPU). Każda maszyna oddaje
swój katalog wyników; ten skrypt składa je w jedno, USUWAJĄC DUBLE po
`source_video` (gdyby ktoś przez pomyłkę policzył to samo wideo dwa razy —
wygrywa wariant z policzonym AU, potem pierwszy napotkany).

Wejście: katalogi z annotations.json + podfolderami klas (jpg).
Wyjście: jeden katalog z annotations.json, annotations_<klasa>.json, kadrami,
licenses.csv, frames.csv.

Uruchom:
  python -m scripts.annotation.merge_releases OUT release_A release_B [...]
"""

import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path


def _has_au(ann: dict) -> bool:
    """Czy anotacja ma policzone AU (delta względem klatki neutralnej)."""
    au = ann.get("au_analysis")
    return bool(au) and isinstance(au, dict) and len(au) > 0


def _load(release: Path) -> tuple[dict, dict]:
    """(mapa image_id->image, lista annotations) z jednego release."""
    coco = json.loads((release / "annotations.json").read_text(encoding="utf-8"))
    images = {i["id"]: i for i in coco["images"]}
    return images, coco["annotations"]


def merge(out_dir: Path, releases: list[Path]) -> None:
    # grupujemy anotacje po source_video z każdego release (para start/end razem)
    by_video: dict[str, dict] = {}  # source_video -> {"release", "images", "anns"}
    for rel in releases:
        if not (rel / "annotations.json").is_file():
            print(f"  ! pomijam (brak annotations.json): {rel}")
            continue
        images, anns = _load(rel)
        grouped: dict[str, list[dict]] = defaultdict(list)
        for a in anns:
            grouped[a["source_video"]].append(a)
        for sv, group in grouped.items():
            has_au = any(_has_au(a) for a in group)
            prev = by_video.get(sv)
            # wygrywa wariant z AU; przy remisie zostaje pierwszy
            if prev is None or (has_au and not prev["has_au"]):
                by_video[sv] = {"release": rel, "has_au": has_au,
                                "images": images, "anns": group}

    # składamy na nowo z ciągłą numeracją + kopiujemy kadry
    out_dir.mkdir(parents=True, exist_ok=True)
    out_images: list[dict] = []
    out_anns: list[dict] = []
    lic_rows: dict[str, dict] = {}
    counters: dict[str, int] = defaultdict(int)
    iid = aid = 0
    dups = 0

    for sv in sorted(by_video):
        entry = by_video[sv]
        rel, images, group = entry["release"], entry["images"], entry["anns"]
        emotion = group[0]["emotion"]
        counters[emotion] += 1
        base = f"{emotion}_{counters[emotion]:03d}"
        group_sorted = sorted(group, key=lambda a: 0 if a.get("frame_role") == "start" else 1)
        for a in group_sorted:
            img = images[a["image_id"]]
            old_path = rel / img["file_name"]
            num = 1 if a.get("frame_role") == "start" else 2
            new_name = f"{emotion}/{base}_{num}.jpg"
            (out_dir / emotion).mkdir(parents=True, exist_ok=True)
            if old_path.is_file():
                shutil.copy(old_path, out_dir / new_name)
            iid += 1
            aid += 1
            out_images.append({**{k: img[k] for k in img if k not in ("id", "file_name")},
                               "id": iid, "file_name": new_name})
            na = dict(a)
            na["id"] = aid
            na["image_id"] = iid
            out_anns.append(na)
        # licencja, jeśli była w źródłowym release
        lic_path = rel / "licenses.csv"
        if lic_path.is_file():
            for row in csv.DictReader(lic_path.open(encoding="utf-8")):
                key = row.get("source_video") or row.get("video_id")
                if key == sv:
                    lic_rows[sv] = row

    _write(out_dir, out_images, out_anns, lic_rows, counters, dups)


def _write(out_dir, out_images, out_anns, lic_rows, counters, dups) -> None:
    info = {"description": "Dog FACS Dataset — scalony", "contributor": "Politechnika Gdańska WETI"}
    categories = [{"id": 1, "name": "dog", "supercategory": "animal", "keypoints": [], "skeleton": []}]
    imgs_by_id = {i["id"]: i for i in out_images}
    per_class: dict[str, dict] = defaultdict(lambda: {"images": [], "annotations": []})
    for a in out_anns:
        per_class[a["emotion"]]["annotations"].append(a)
        per_class[a["emotion"]]["images"].append(imgs_by_id[a["image_id"]])
    for emotion, data in per_class.items():
        (out_dir / f"annotations_{emotion}.json").write_text(
            json.dumps({"info": info, "categories": categories, **data}, ensure_ascii=False),
            encoding="utf-8")
    (out_dir / "annotations.json").write_text(
        json.dumps({"info": info, "categories": categories, "images": out_images,
                    "annotations": out_anns}, ensure_ascii=False), encoding="utf-8")
    if lic_rows:
        fields = list(next(iter(lic_rows.values())).keys())
        with (out_dir / "licenses.csv").open("w", encoding="utf-8", newline="") as h:
            w = csv.DictWriter(h, fieldnames=fields)
            w.writeheader()
            for sv in sorted(lic_rows):
                w.writerow(lic_rows[sv])
    with (out_dir / "frames.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["source_video", "emotion", "frame_role",
                                          "frame_time_s", "file_name"])
        w.writeheader()
        for a in out_anns:
            w.writerow({"source_video": a["source_video"], "emotion": a["emotion"],
                        "frame_role": a.get("frame_role"), "frame_time_s": a.get("frame_time_s"),
                        "file_name": imgs_by_id[a["image_id"]]["file_name"]})

    print(f"=== scalono: {out_dir} ===")
    total = len(out_images)
    for e in sorted(per_class):
        print(f"  {e:9} {counters[e]:3} wideo ({len(per_class[e]['images'])} kadrów)")
    print(f"Kadrów łącznie: {total}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("Użycie: python -m scripts.annotation.merge_releases OUT release_A release_B [...]")
    merge(Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]])
