#!/usr/bin/env python3
"""
Pełny zbiór 9k — wszystkie klatki, etykieta automatyczna, człowiek jako podzbiór.

    python -m scripts.annotation.build_full_dataset

`build_final_dataset` oddaje TYLKO pary zweryfikowane przez człowieka (528) —
złoty, ale mały rdzeń. Ten skrypt składa DRUGI artefakt: cały materiał 9150 klatek
z automatyczną etykietą AU po szumowym gejcie, a werdykt człowieka wchodzi tam,
gdzie istnieje, jako podzbiór oznaczony `label_source=human_verified`.

Obrazem są PEŁNE klatki z `work/frames/` (już w repozytorium) — nie kadrujemy
9150 klatek na nowo, bo dołożyłoby to ~200 MB, a klatki i tak są wersjonowane.
`file_name` w COCO wskazuje je względem `work/frames/`. Pomiar reguł (`au_analysis`,
`au_noise`) NIE jedzie do tego pliku — jest w `curated.json`; tutaj zostaje tylko
zwięzła etykieta, żeby COCO zmieściło się pod limitem 100 MB GitHuba.

Wynik w `data/dataset_final/release/`:
    annotations_full.json   COCO 9150 klatek: keypoints, rasa, emocja, AU (auto + człowiek)
    au_full_labels.csv      tabela AU na pik: werdykt człowieka albo auto (kolumna label_source)
    LICENSE                 CC BY-NC 4.0
"""

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "apps" / "webapp" / "backend"))

from packages.data.coco import au_auto_verdicts  # noqa: E402
from packages.data.schemas import (  # noqa: E402
    EMOTION_CLASSES,
    KEYPOINT_NAMES,
    SKELETON_CONNECTIONS,
)
from packages.models.au_geometry import AUGeometryModel, pair_features  # noqa: E402
from packages.models.delta_action_units import ACTION_UNIT_NAMES  # noqa: E402
from scripts.annotation.build_final_dataset import (  # noqa: E402
    DEFAULT_DATASET,
    LICENSES,
    VERDICT_TO_CSV,
    _neutral_of,
    _resolve_data,
    index_curated,
    resolve_labels,
)
from scripts.annotation.build_work_pack import shrink  # noqa: E402
from scripts.annotation.train_au_model import DEFAULT_WEIGHTS, load_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CC_BY_NC_TEXT: str = (
    "Dog FACS Dataset — pełny zbiór 9k\n"
    "Licencja: Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)\n"
    "https://creativecommons.org/licenses/by-nc/4.0/\n"
)


def build_full_coco(
    coco: dict, labels: dict, model: Optional[AUGeometryModel] = None
) -> tuple[dict, list[dict]]:
    """
    Składa COCO całego zbioru i wiersze CSV etykiet AU.

    Werdykt pary wiąże się po pliku PIKU (`pair_key`), a poprawka punktów po pliku
    KLATKI, którą poprawiano — także neutralnej. Dzięki temu nie gubi się ani
    werdyktu przy zamianie ról, ani poprawki bazy AU.

    Args:
        coco: Wczytany `curated.json`
        labels: Mapa pair_key -> werdykt człowieka
        model: Wyuczony model AU albo None, gdy nie ma zapisanych wag

    Returns:
        Para (COCO, wiersze CSV pików)
    """
    index = index_curated(coco)
    corrections = {key: record for key, record in labels.items() if record.keypoints}
    verdict_by_peak = {
        key: record
        for key, record in labels.items()
        if record.usable and key in index.peak_by_file
    }

    images = [
        {
            "id": image["id"],
            "file_name": image["file_name"],
            "width": image.get("width"),
            "height": image.get("height"),
            "license": LICENSES[0]["id"],
            "source_video": image.get("source_video"),
            "frame_number": image.get("frame_number"),
        }
        for image in coco["images"]
    ]

    annotations: list[dict] = []
    rows: list[dict[str, object]] = []
    for source in coco["annotations"]:
        file_name = index.images[source["image_id"]]["file_name"]
        correction = corrections.get(file_name)
        is_peak = source.get("frame_role") == "peak"
        record = verdict_by_peak.get(file_name) if is_peak else None
        auto = au_auto_verdicts(source.get("au_analysis", {}))
        from_model = _model_verdicts(model, index, source) if is_peak else {}
        annotation = {
            "id": source["id"],
            "image_id": source["image_id"],
            "category_id": 1,
            "bbox": source.get("bbox"),
            "area": source.get("area"),
            "iscrowd": 0,
            "keypoints": list(correction.keypoints)
            if correction is not None
            else source.get("keypoints"),
            "num_keypoints": source.get("num_keypoints"),
            "track_id": source["track_id"],
            "frame_role": source.get("frame_role"),
            "neutral_frame_id": source.get("neutral_frame_id"),
            "breed": (record.breed if record else None) or source.get("breed"),
            "emotion": (record.emotion if record else None) or source.get("emotion"),
            "au_auto_verdict": auto,
            "label_source": "auto_model" if from_model else "auto_rules",
        }
        if record is not None:
            annotation["au_verdicts"] = record.au_verdicts
            annotation["annotator"] = record.annotator
            annotation["roles_swapped"] = record.roles_swapped
            annotation["label_source"] = "human_verified"
        if from_model:
            annotation["au_model_verdict"] = from_model
        annotations.append(annotation)

        if is_peak:
            rows.append(
                _peak_row(index.images[source["image_id"]], source, auto, record, from_model)
            )

    full = {
        "info": {
            "description": "Dog FACS Dataset — pełny zbiór 9k (auto + podzbiór człowieka)",
            "url": "https://github.com/eternaki/group-project",
            "version": "1.0",
            "year": datetime.now(timezone.utc).year,
            "contributor": "Politechnika Gdańska WETI",
            "date_created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "image_root": "data/dataset_final/work/frames",
        },
        "licenses": LICENSES,
        "categories": [
            {
                "id": 1,
                "name": "dog",
                "supercategory": "animal",
                "keypoints": KEYPOINT_NAMES,
                "skeleton": [list(pair) for pair in SKELETON_CONNECTIONS],
            }
        ],
        "emotion_categories": [
            {"id": index_, "name": name} for index_, name in enumerate(EMOTION_CLASSES)
        ],
        "action_units": list(ACTION_UNIT_NAMES),
        "images": images,
        "annotations": annotations,
    }
    return full, rows


def _model_verdicts(model: Optional[AUGeometryModel], index, peak: dict) -> dict[str, str]:
    """
    Liczy werdykt modelu geometrycznego dla jednej pary.

    Model dostaje role TAKIE, JAKIE DAŁ PIPELINE — przy nowym materiale nikt nie
    powie, że klatki są zamienione. Dokładnie tak został zmierzony (precyzja
    26.8%), więc etykieta w zbiorze odpowiada podanej liczbie.

    Args:
        model: Wyuczony model albo None
        index: Indeks kuracji
        peak: Anotacja klatki szczytowej

    Returns:
        Werdykt na każde AU, które model umie ocenić; pusty, gdy modelu nie ma
    """
    if model is None:
        return {}
    neutral = _neutral_of(index, peak)
    if neutral is None:
        return {}
    return model.predict(pair_features(peak.get("keypoints"), neutral.get("keypoints")))


def _peak_row(
    image: dict, source: dict, auto: dict, record, from_model: Optional[dict] = None
) -> dict[str, object]:
    """
    Buduje wiersz CSV dla klatki szczytowej: człowiek, potem model, potem reguły.

    Kolejność nie jest dowolna — to porządek zmierzonej wiarygodności. Werdykt
    człowieka jest etykietą. Model ma precyzję 26.8%, reguły 5.0%, więc tam,
    gdzie model się wypowiada, jego zdanie zastępuje regułę.

    Args:
        image: Wpis obrazu z kuracji
        source: Anotacja piku
        auto: Automatyczne werdykty AU tego piku
        record: Werdykt człowieka albo None
        from_model: Werdykt modelu geometrycznego albo None

    Returns:
        Wiersz jako słownik kolumn
    """
    human_verdicts = record.au_verdicts if record is not None else None
    if record is not None:
        zrodlo = "human_verified"
    elif from_model:
        zrodlo = "auto_model"
    else:
        zrodlo = "auto_rules"
    row: dict[str, object] = {
        "pair_key": image["file_name"],
        "source_video": image.get("source_video"),
        "emotion": (record.emotion if record else None) or source.get("emotion"),
        "label_source": zrodlo,
        "annotator": record.annotator if record is not None else "",
    }
    verdicts = human_verdicts or from_model or auto
    for au in ACTION_UNIT_NAMES:
        row[au] = VERDICT_TO_CSV.get(verdicts.get(au, "not_observable"), "")
    return row


README_MARKER: str = "## Pełny zbiór 9k"


def _write_full_json(path: Path, coco: dict) -> None:
    """
    Zapisuje COCO pełnego zbioru możliwie ciasno.

    Dwa powody, oba zmierzone. Precyzja: same współrzędne niosą po 15 cyfr
    znaczących, z których realne są dwie — `shrink` ścina to bez straty treści
    (68.6 MB -> 29.8 MB). Sposób zapisu: `Path.write_text` na tym
    Pythonie/Windows wywraca się na `OSError: [Errno 22]` w okolicach 70 MB,
    a zwykłe `open()` tej samej treści przechodzi — przy 9.5k par byliśmy
    1.4 MB od tej granicy.

    Args:
        path: Ścieżka pliku
        coco: Złożony COCO całego zbioru
    """
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(coco, ensure_ascii=False, separators=(",", ":")))


def write_outputs(full: dict, rows: list[dict], output_dir: Path) -> None:
    """
    Zapisuje COCO, CSV, licencję i dopisuje sekcję do README.

    Args:
        full: Złożony COCO całego zbioru
        rows: Wiersze CSV pików
        output_dir: Katalog `release`
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    slim = {**full, "annotations": shrink(full["annotations"])}
    _write_full_json(output_dir / "annotations_full.json", slim)
    with (output_dir / "au_full_labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "LICENSE").write_text(CC_BY_NC_TEXT, encoding="utf-8")
    _append_readme(output_dir / "README.md", full, rows)


def _append_readme(path: Path, full: dict, rows: list[dict]) -> None:
    """
    Dopisuje sekcję o pełnym zbiorze do README (idempotentnie).

    Args:
        path: Ścieżka README
        full: Złożony COCO
        rows: Wiersze CSV pików
    """
    human = sum(1 for row in rows if row["label_source"] == "human_verified")
    section = "\n".join(
        [
            README_MARKER,
            "",
            "Obok złotego podzbioru (`annotations.json` — kadry mordy zweryfikowane",
            "przez człowieka) leży CAŁY materiał:",
            "",
            "```",
            f"annotations_full.json   COCO {len(full['images'])} klatek, {len(full['annotations'])} anotacji",
            f"au_full_labels.csv      {len(rows)} pików: werdykt człowieka albo auto (label_source)",
            "LICENSE                 CC BY-NC 4.0",
            "```",
            "",
            f"- Klatek: **{len(full['images'])}**, pików: **{len(rows)}** "
            f"(w tym **{human}** z werdyktem człowieka, reszta etykieta automatyczna).",
            "- **Obrazem są PEŁNE klatki** z `data/dataset_final/work/frames/` (już w repo),",
            "  a `file_name` wskazuje je względem tego katalogu. Punkty są w układzie pełnej",
            "  klatki. Pomiar reguł (`au_analysis`) jest w `work/curated.json`.",
            "",
            "### Skąd bierze się etykieta AU i ile jest warta",
            "",
            "Kolumna `label_source` mówi, kto orzekł. Zmierzone na parach człowieka,",
            "sprawdzianem krzyżowym z podziałem po nagraniach:",
            "",
            "| źródło | pole w COCO | precyzja | pokrycie |",
            "|--------|-------------|----------|----------|",
            "| `human_verified` | `au_verdicts` | etykieta odniesienia | — |",
            "| `auto_model` | `au_model_verdict` | 26.8% | 22.5% |",
            "| `auto_rules` | `au_auto_verdict` | 5.0% | 40.5% |",
            "",
            "- **Do treningu bierz `au_verdicts` tam, gdzie jest, a dalej `au_model_verdict`.**",
            "  Reguły (`au_auto_verdict`) zostają w pliku wyłącznie dla porównania.",
            "- **Żadna etykieta automatyczna nie jest prawdą.** Przy precyzji 27% trzy",
            "  aktywacje na cztery są zmyślone. Model zapala średnio 0.50 AU na kadr,",
            "  człowiek 0.42, reguły 5.65 — to jedyny sens, w jakim model „się zgadza\".",
            "- **Sufit jest nisko i to nie wina modelu.** Na parach ocenionych niezależnie",
            "  przez dwie osoby zgoda na aktywacjach AU wynosi 7.4% (kappa 0.132), więc",
            "  samo zjawisko jest słabo powtarzalne. Odtworzenie liczb:",
            "  `python -m scripts.annotation.eval_au_rules` i `train_au_model`.",
            "",
        ]
    )
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    if README_MARKER in existing:
        existing = existing[: existing.index(README_MARKER)].rstrip() + "\n"
    path.write_text(existing.rstrip() + "\n\n" + section, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parsuje argumenty wiersza poleceń."""
    parser = argparse.ArgumentParser(description="Zlozenie pelnego zbioru 9k Dog FACS")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Nazwa zbioru w data/")
    return parser.parse_args()


def main() -> None:
    """Punkt wejścia."""
    args = parse_args()
    dataset_dir = REPO_ROOT / "data" / args.dataset
    coco = json.loads(_resolve_data(dataset_dir, "curated.json").read_text(encoding="utf-8"))
    labels, _ = resolve_labels(args.dataset)

    model = load_model()
    if model is None:
        logger.warning("Brak %s — etykieta automatyczna zostanie z regul", DEFAULT_WEIGHTS.name)
    else:
        logger.info("Model AU: %d jednostek z %s", len(model.per_unit), DEFAULT_WEIGHTS.name)
    full, rows = build_full_coco(coco, labels, model)
    write_outputs(full, rows, dataset_dir / "release")

    sources = Counter(row["label_source"] for row in rows)
    logger.info("Klatek (obrazów)           : %d", len(full["images"]))
    logger.info("Anotacji                   : %d", len(full["annotations"]))
    logger.info("Pików w CSV                : %d", len(rows))
    logger.info("  w tym werdykt człowieka  : %d", sources.get("human_verified", 0))
    logger.info("  etykieta modelu          : %d", sources.get("auto_model", 0))
    logger.info("  etykieta regul           : %d", sources.get("auto_rules", 0))
    logger.info("Gotowe: %s", dataset_dir / "release")


if __name__ == "__main__":
    main()
