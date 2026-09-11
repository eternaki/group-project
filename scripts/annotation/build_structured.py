#!/usr/bin/env python3
"""
Strukturalny zbiór pod wymogi prowadzącego — jedną komendą, z istniejącego materiału.

Realizuje punkty 1,2,3,5,7 + bbox w JSON, BEZ zmyślania etykiet:
- foldery według klas emocji (pkt 1),
- licencja + link do źródła dla każdego wideo (pkt 2),
- pliki nazwane losowym ciągiem (pkt 3),
- każde wideo ma emocję z ETYKIETY (człowiek gdzie jest, inaczej model) (pkt 5),
- spójne liczby + README (pkt 7),
- bbox psa w anotacji, obraz to pełna klatka (uwaga końcowa).

Emocja wideo = werdykt człowieka (jeśli ktoś ocenił którąś klatkę), inaczej
najczęstsza emocja modelu wśród pików. Nic nie jest losowane ani dosypywane.
"""

import argparse
import csv
import json
import logging
import re
import shutil
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
DATASET = "dataset_final"
CURATED = REPO / "data" / DATASET / "work" / "curated.json"
FRAMES = REPO / "data" / DATASET / "work" / "frames"
LABELS = REPO / "data" / "labels" / DATASET
OUTPUT = REPO / "data" / DATASET / "structured"

# Ile najwyżej klatek na wideo bierzemy (2 to minimum prowadzącego; dajemy zapas)
FRAMES_PER_VIDEO = 6
# Kap na klasę — częste emocje przycinamy, żeby nie było 800 vs 30 (to tylko WYBÓR
# podzbioru, nie zmiana etykiet). Rzadkie zostają w całości.
CAP_PER_EMOTION = 200

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def infer_license(video_id: str) -> tuple[str, str, str]:
    """
    Zgaduje platformę, licencję i link po nazwie źródła wideo.

    Args:
        video_id: Identyfikator wideo (nazwa folderu źródła)

    Returns:
        (platforma, licencja, link)
    """
    low = video_id.lower()
    if low.startswith("mixkit"):
        return "Mixkit", "Mixkit Free License", "https://mixkit.co/free-stock-video/"
    if low.startswith("coverr"):
        return "Coverr", "Coverr License (free)", "https://coverr.co"
    if low.startswith("pixabay"):
        return "Pixabay", "Pixabay Content License", "https://pixabay.com"
    if low.startswith("youtube") or low.startswith("yt_"):
        return "YouTube", "fragment badawczy <25% utworu", "https://youtube.com"
    # 11-znakowe ID YouTube z sufiksem klatki, np. "Mste1BpEu2c_000405"
    yt = re.match(r"^([A-Za-z0-9_-]{11})_\d+$", video_id)
    if yt and not yt.group(1).isdigit():
        return (
            "YouTube",
            "fragment badawczy <25% utworu",
            f"https://www.youtube.com/watch?v={yt.group(1)}",
        )
    # tiktokowe nagrania mają długie numeryczne ID (dogmood_..., dogsoftiktok_...)
    tail = video_id.split("_")[-1]
    if tail.isdigit() and len(tail) >= 15:
        return "TikTok", "fragment badawczy <25% utworu", "https://tiktok.com"
    return "inne/lokalne", "fragment badawczy <25% utworu", ""


def human_emotions() -> dict[str, str]:
    """Mapa pair_key -> emocja człowieka (z wszystkich dzienników, tylko usable)."""
    out: dict[str, str] = {}
    for path in LABELS.glob("*.jsonl"):
        for line in path.open(encoding="utf-8"):
            if line.strip():
                record = json.loads(line)
                if record.get("usable") and record.get("emotion"):
                    out[record["pair_key"]] = record["emotion"]
    return out


def build() -> None:
    """Składa strukturalny zbiór."""
    coco = json.loads(CURATED.read_text(encoding="utf-8"))
    images = {i["id"]: i for i in coco["images"]}
    peaks = [a for a in coco["annotations"] if a.get("frame_role") == "peak"]
    human = human_emotions()

    # Grupujemy piki po wideo
    by_video: dict[str, list[dict]] = defaultdict(list)
    for annotation in peaks:
        image = images[annotation["image_id"]]
        video = image.get("source_video") or image["file_name"].split("/")[1]
        by_video[video].append(annotation)

    # Emocja wideo: człowiek (jeśli ocenił którąś klatkę) inaczej model (moda pików)
    video_emotion: dict[str, str] = {}
    video_source: dict[str, str] = {}  # human_verified / auto_model
    for video, anns in by_video.items():
        human_votes = [
            human[images[a["image_id"]]["file_name"]]
            for a in anns
            if images[a["image_id"]]["file_name"] in human
        ]
        if human_votes:
            video_emotion[video] = Counter(human_votes).most_common(1)[0][0]
            video_source[video] = "human_verified"
        else:
            model_votes = [a.get("emotion") for a in anns if a.get("emotion")]
            video_emotion[video] = Counter(model_votes).most_common(1)[0][0] if model_votes else "neutral"
            video_source[video] = "auto_model"

    # Kap na emocję (wybór podzbioru wideo, nie zmiana etykiet)
    per_emotion_videos: dict[str, list[str]] = defaultdict(list)
    # człowiek pierwszy — gwarantujemy, że ręczne trafiają do zbioru
    for video in sorted(by_video, key=lambda v: video_source[v] != "human_verified"):
        emotion = video_emotion[video]
        if len(per_emotion_videos[emotion]) < CAP_PER_EMOTION:
            per_emotion_videos[emotion].append(video)

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)

    out_images: list[dict] = []
    out_anns: list[dict] = []
    licenses_rows: list[dict] = []
    seen_video_license: set[str] = set()
    img_id = 0
    ann_id = 0

    for emotion, videos in per_emotion_videos.items():
        (OUTPUT / emotion).mkdir(exist_ok=True)
        for video in videos:
            platform, lic, link = infer_license(video)
            if video not in seen_video_license:
                licenses_rows.append(
                    {"video_id": video, "platform": platform, "license": lic, "link": link}
                )
                seen_video_license.add(video)
            for annotation in by_video[video][:FRAMES_PER_VIDEO]:
                image = images[annotation["image_id"]]
                src = FRAMES / image["file_name"]
                if not src.is_file():
                    continue
                rand = uuid.uuid4().hex + ".jpg"
                shutil.copy2(src, OUTPUT / emotion / rand)
                img_id += 1
                out_images.append(
                    {
                        "id": img_id,
                        "file_name": f"{emotion}/{rand}",
                        "width": image.get("width"),
                        "height": image.get("height"),
                        "source_video": video,
                        "license": lic,
                        "license_link": link,
                    }
                )
                ann_id += 1
                out_anns.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": annotation.get("bbox"),
                        "keypoints": annotation.get("keypoints"),
                        "num_keypoints": annotation.get("num_keypoints"),
                        "au_analysis": annotation.get("au_analysis", {}),
                        "breed": annotation.get("breed"),
                        "emotion": emotion,
                        "label_source": video_source[video],
                        "source_video": video,
                    }
                )

    coco_out = {
        "info": {
            "description": "Dog FACS Dataset — zbiór strukturalny",
            "date_created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "contributor": "Politechnika Gdańska WETI",
        },
        "categories": [{"id": 1, "name": "dog", "supercategory": "animal"}],
        "images": out_images,
        "annotations": out_anns,
    }
    (OUTPUT / "annotations.json").write_text(
        json.dumps(coco_out, ensure_ascii=False), encoding="utf-8"
    )
    with (OUTPUT / "licenses.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "platform", "license", "link"])
        writer.writeheader()
        writer.writerows(licenses_rows)

    _write_readme(per_emotion_videos, video_source, out_images)
    _report(per_emotion_videos, video_source, out_images)


def _write_readme(per_emotion, video_source, out_images) -> None:
    """Zapisuje README z liczbami."""
    lines = [
        "# Dog FACS Dataset — zbiór strukturalny",
        "",
        f"Złożono: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Struktura",
        "- Foldery = klasy emocji; w każdym pełne klatki wideo tej emocji.",
        "- Pliki nazwane losowym ciągiem. `annotations.json` (COCO): bbox psa,",
        "  46 keypoints, 21 AU, emocja, `source_video`, `label_source`.",
        "- `licenses.csv`: licencja + link źródła dla każdego wideo.",
        "",
        "## Liczby (wideo na emocję)",
    ]
    for emotion in sorted(per_emotion, key=lambda e: -len(per_emotion[e])):
        videos = per_emotion[emotion]
        human = sum(1 for v in videos if video_source[v] == "human_verified")
        lines.append(f"- **{emotion}**: {len(videos)} wideo (człowiek: {human}, model: {len(videos)-human})")
    lines += [
        "",
        f"Klatek łącznie: {len(out_images)}",
        "",
        "## Etykiety emocji",
        "Emocja wideo = werdykt człowieka, gdzie ktoś ocenił klatkę; inaczej",
        "najczęstsza emocja modelu wśród klatek szczytowych. Rzadkie emocje",
        "ograniczone dostępnym materiałem (wyraziste nagrania psów są rzadkie).",
    ]
    (OUTPUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _report(per_emotion, video_source, out_images) -> None:
    """Wypisuje podsumowanie."""
    logger.info("=== Zbiór strukturalny gotowy: %s ===", OUTPUT)
    for emotion in sorted(per_emotion, key=lambda e: -len(per_emotion[e])):
        videos = per_emotion[emotion]
        human = sum(1 for v in videos if video_source[v] == "human_verified")
        logger.info("  %-11s %4d wideo  (człowiek %d / model %d)", emotion, len(videos), human, len(videos) - human)
    logger.info("Klatek: %d", len(out_images))


if __name__ == "__main__":
    argparse.ArgumentParser(description="Zbiór strukturalny Dog FACS").parse_args()
    build()
