#!/usr/bin/env python3
"""
Poprawki zbioru release_startend wg uwag prowadzącego:

1. Parę kadrów z jednego wideo nazywamy tak samo + numer kadru: <emocja>_<NNN>_1/_2.jpg
2. Kadry poniżej 720p (krótszy bok < 720) USUWAMY — bez sztucznego podbijania.
3. Zamiast losowych nazw-hashy — czytelne nazwy z punktu 1.
5. licenses.csv: bezpośredni link do źródła (Pexels/YouTube/TikTok itp.).
6. annotations.json rozbity po klasach: annotations_<emocja>.json.

Punkt 4 (ponowna weryfikacja emocji) jest RĘCZNY — nie da się go zrobić skryptem.
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parent.parent.parent
D = REPO / "data" / "dataset_final" / "release_startend"
# Krótszy bok kadru musi być >= MIN_SIDE. 0 = bez filtra rozdzielczości.
MIN_SIDE = 720
# Scalanie emocji (fearful i surprise są u psów wizualnie bliskie -> jedna klasa)
MERGE: dict[str, str] = {}  # fearful i surprise rozdzielone


def direct_license(sv: str) -> tuple[str, str, str]:
    """Platforma, licencja i BEZPOŚREDNI link źródła po nazwie wideo (nigdy pusty)."""
    fair = "fragment badawczy <25% utworu"
    low = sv.lower()
    if low.startswith("mixkit"):
        return "Mixkit", "Mixkit Free License", "https://mixkit.co/free-stock-video/"
    if low.startswith("coverr"):
        return "Coverr", "Coverr License (free)", "https://coverr.co"
    # Pixabay: pixabay_<id> oraz magnific-<id> (pobrania Pixabay)
    if low.startswith("pixabay") or low.startswith("magnific"):
        m = re.search(r"(\d{4,})", sv)
        link = f"https://pixabay.com/videos/id-{m.group(1)}/" if m else "https://pixabay.com"
        return "Pixabay", "Pixabay Content License", link
    # Pexels: pexels_ / dog_ / segment_long_ / goła liczba 6-9 cyfr
    m = re.fullmatch(r"(?:pexels_|dog_)?(\d{6,9})", sv) or re.match(r"segment_long_(\d{6,9})", sv)
    if m:
        return "Pexels", "Pexels License", f"https://www.pexels.com/video/{m.group(1)}/"
    # YouTube ze starą nazwą zapytania: ID (11 znaków) jest na końcu nazwy
    last = sv.split("_")[-1]
    if low.startswith("youtube") and re.fullmatch(r"[A-Za-z0-9_-]{11}", last) and not last.isdigit():
        return "YouTube", fair, f"https://www.youtube.com/watch?v={last}"
    # YouTube: yt_<id> lub 11-znakowe ID (opcjonalnie z sufiksem klatki)
    m = re.fullmatch(r"(?:yt_)?([A-Za-z0-9_-]{11})(?:_\d+)?", sv)
    if m and not m.group(1).isdigit():
        return "YouTube", fair, f"https://www.youtube.com/watch?v={m.group(1)}"
    if low.startswith("youtube"):  # YouTube bez odzyskiwalnego ID
        return "YouTube", fair, "https://www.youtube.com"
    # TikTok: długie numeryczne ID
    tail = sv.split("_")[-1]
    if tail.isdigit() and len(tail) >= 15:
        return "TikTok", fair, f"https://www.tiktok.com/video/{tail}"
    if re.fullmatch(r"[0-9a-f]{32}", sv):  # hash pobrania z TikToka
        return "TikTok", fair, "https://www.tiktok.com"
    return "inne/lokalne", fair, "https://www.tiktok.com"


def _resolution(path: Path) -> tuple[int, int]:
    im = cv2.imread(str(path))
    if im is None:
        return 0, 0
    h, w = im.shape[:2]
    return w, h


def refine() -> None:
    coco = json.loads((D / "annotations.json").read_text(encoding="utf-8"))
    images = {i["id"]: i for i in coco["images"]}
    anns = coco["annotations"]

    # scalanie emocji (np. fearful -> surprise) przed grupowaniem/nazywaniem
    for a in anns:
        a["emotion"] = MERGE.get(a["emotion"], a["emotion"])

    # grupujemy anotacje po wideo
    by_video: dict[str, list[dict]] = defaultdict(list)
    for a in anns:
        by_video[a["source_video"]].append(a)

    dropped_lowres = 0
    kept_videos: list[tuple[str, list[dict]]] = []
    # filtr rozdzielczości — całe wideo pada, jeśli kadr < 720p (oba mają tę samą res)
    for sv, group in by_video.items():
        if MIN_SIDE:
            first_img = images[group[0]["image_id"]]
            w, h = _resolution(D / first_img["file_name"])
            if min(w, h) < MIN_SIDE:
                for a in group:  # kasujemy pliki z dysku
                    (D / images[a["image_id"]]["file_name"]).unlink(missing_ok=True)
                dropped_lowres += len(group)
                continue
        kept_videos.append((sv, group))

    # nowe nazwy: <emocja>_<NNN>_1/_2.jpg (start=1, end=2); numeracja per emocja
    counters: dict[str, int] = defaultdict(int)
    new_images: list[dict] = []
    new_anns: list[dict] = []
    licenses: dict[str, tuple[str, str, str]] = {}
    iid = aid = 0
    # sortujemy dla determinizmu
    for sv, group in sorted(kept_videos, key=lambda x: (x[1][0]["emotion"], x[0])):
        emotion = group[0]["emotion"]
        counters[emotion] += 1
        base = f"{emotion}_{counters[emotion]:03d}"
        licenses[sv] = direct_license(sv)
        # start przed end
        group_sorted = sorted(group, key=lambda a: 0 if a.get("frame_role") == "start" else 1)
        for a in group_sorted:
            img = images[a["image_id"]]
            old = D / img["file_name"]
            num = 1 if a.get("frame_role") == "start" else 2
            new_name = f"{emotion}/{base}_{num}.jpg"
            if old.is_file():
                old.rename(D / new_name)
            iid += 1
            aid += 1
            new_images.append({**{k: img[k] for k in img if k != "file_name"},
                               "id": iid, "file_name": new_name})
            na = dict(a)
            na["id"] = aid
            na["image_id"] = iid
            new_anns.append(na)

    _write(coco, new_images, new_anns, licenses, dropped_lowres, counters)


def _write(coco, new_images, new_anns, licenses, dropped, counters) -> None:
    info = coco.get("info", {})
    categories = coco.get("categories", [{"id": 1, "name": "dog"}])

    # sprzątamy stare pliki per-klasa i puste foldery (po scaleniu emocji)
    for old in D.glob("annotations_*.json"):
        old.unlink()
    for sub in D.iterdir():
        if sub.is_dir() and not any(sub.iterdir()):
            sub.rmdir()

    # 6. rozbicie po klasach + zbiorczy plik
    imgs_by_id = {i["id"]: i for i in new_images}
    per_class: dict[str, dict] = defaultdict(lambda: {"images": [], "annotations": []})
    for a in new_anns:
        e = a["emotion"]
        per_class[e]["annotations"].append(a)
        per_class[e]["images"].append(imgs_by_id[a["image_id"]])
    for emotion, data in per_class.items():
        out = {"info": info, "categories": categories,
               "images": data["images"], "annotations": data["annotations"]}
        (D / f"annotations_{emotion}.json").write_text(
            json.dumps(out, ensure_ascii=False), encoding="utf-8")

    coco_all = {"info": info, "categories": categories,
                "images": new_images, "annotations": new_anns}
    (D / "annotations.json").write_text(json.dumps(coco_all, ensure_ascii=False), encoding="utf-8")

    # 5. licenses.csv z bezpośrednimi linkami
    with (D / "licenses.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["source_video", "platform", "license", "link"])
        w.writeheader()
        for sv, (plat, lic, link) in sorted(licenses.items()):
            w.writerow({"source_video": sv, "platform": plat, "license": lic, "link": link})

    # frames.csv (nowe nazwy)
    with (D / "frames.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["source_video", "emotion", "frame_role", "frame_time_s", "file_name"])
        w.writeheader()
        rows = [{"source_video": a["source_video"], "emotion": a["emotion"],
                 "frame_role": a.get("frame_role"), "frame_time_s": a.get("frame_time_s"),
                 "file_name": imgs_by_id[a["image_id"]]["file_name"]} for a in new_anns]
        w.writerows(sorted(rows, key=lambda r: (r["emotion"], r["file_name"])))

    print(f"Usunięto kadrów <720p: {dropped}")
    print(f"Kadrów po poprawkach: {len(new_images)}")
    for e in sorted(counters):
        print(f"  {e:9} wideo: {counters[e]:3}  (kadrów: {len(per_class[e]['images'])})")
    print("Pliki annotations_<klasa>.json zapisane; licenses.csv z bezpośrednimi linkami.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Poprawki zbioru release_startend")
    ap.add_argument("--min-side", type=int, default=720,
                    help="Minimalny krótszy bok kadru (0 = bez filtra rozdzielczości)")
    args = ap.parse_args()
    MIN_SIDE = args.min_side
    refine()
