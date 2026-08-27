#!/usr/bin/env python3
"""
Złożenie finalnego zbioru z pracy człowieka — jedną komendą, w każdej chwili.

    python -m scripts.annotation.build_final_dataset

Anotator widzi tylko jedno: „zapisz i dalej". Jego decyzja ląduje wtedy w
dzienniku `data/labels/<zbior>/<kto>.jsonl` i tam zostaje — ale sam dziennik
nie jest zbiorem: nie ma w nim ani obrazów, ani pomiarów pipeline'u, ani formy
COCO. Ten skrypt scala trzy rzeczy w jeden katalog gotowy do oddania:

    dziennik decyzji  +  pomiary kuracji  +  klatki  ->  data/dataset_final/

Wolno go uruchamiać na dowolnym etapie pracy: składa to, co zweryfikowano DO
TEJ PORY, i nadpisuje poprzedni wynik. Dzięki temu „katalog do wysłania"
istnieje zawsze, a nie dopiero po ostatniej klatce.

Obrazy zapisujemy jako KADRY MORDY, nie pełne klatki. Zbiór opisuje 46 punktów
twarzy i 21 AU — piksele tułowia nie niosą żadnej etykiety, a kosztują
trzynastokrotnie więcej miejsca (700 MB wobec 52 MB na 2222 klatkach). Oryginał
nie ginie: `source_video` i `frame_number` wskazują nagranie w `data/drive_dogs/`,
a `source_bbox` trzyma położenie kadru w pełnej klatce.
"""

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# Dziennik etykiet mieszka w backendzie, bo to on go zapisuje. Skrypt czyta ten
# sam moduł zamiast powielać format — inaczej dwie implementacje jednego pliku
# rozjechałyby się przy pierwszej zmianie pola.
sys.path.insert(0, str(REPO_ROOT / "apps" / "webapp" / "backend"))

from label_store import LabelRecord, read_all  # noqa: E402

from packages.data.schemas import (  # noqa: E402
    EMOTION_CLASSES,
    KEYPOINT_NAMES,
    NUM_KEYPOINTS,
    SKELETON_CONNECTIONS,
)
from packages.models.delta_action_units import ACTION_UNIT_NAMES  # noqa: E402
from scripts.annotation.cropping import (  # noqa: E402
    bbox_from_keypoints,
    crop_and_scale,
    face_box,
    file_size,
    read_image,
    remap_keypoints,
    remove_tree,
    write_jpeg,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Ile dołożyć wokół rozpiętości punktów. Ćwierć szerokości mordy z każdej strony
# zostawia ucho i podgardle, po których człowiek poznaje kadr — sam obrys punktów
# ucina je i zdjęcie przestaje być czytelne przy weryfikacji.
FACE_MARGIN: float = 0.25

# Dłuższy bok kadru mordy. 512 px to granica, powyżej której rozmiar zbioru rośnie
# szybciej niż jego użyteczność: mediana rozpiętości punktów to 276 px, więc powyżej
# 512 skalowalibyśmy w górę szum interpolacji, nie szczegół.
MAX_CROP_SIDE: int = 512

JPEG_QUALITY: int = 90

DEFAULT_DATASET: str = "dataset_final"

# Gotowy zbiór leży W ŚRODKU zbioru roboczego, a nie obok: „finalny" ma być jeden
# katalog, żeby nie trzeba było pamiętać, który z dwóch podobnie nazwanych jest
# tym do oddania. Materiał roboczy obok niego jest ignorowany przez gita,
# `release/` — nie.
DEFAULT_OUTPUT: str = "data/dataset_final/release"

IMAGES_DIRNAME: str = "images"
ANNOTATIONS_NAME: str = "annotations.json"
CSV_NAME: str = "au_labels.csv"
README_NAME: str = "README.md"

# Werdykt człowieka w CSV. `not_observable` NIE jest zerem — to brak wiedzy,
# a nie brak ruchu. Pusta komórka zmusza trening do pominięcia próbki zamiast
# uczenia się na zmyślonym „nieaktywne".
VERDICT_TO_CSV: dict[str, str] = {"active": "1", "inactive": "0", "not_observable": ""}

LICENSES: list[dict] = [
    {"id": 1, "name": "CC BY-NC 4.0", "url": "https://creativecommons.org/licenses/by-nc/4.0/"}
]


@dataclass
class BuildStats:
    """Licznik tego, co weszło do zbioru i co odpadło."""

    pairs_labeled: int = 0
    pairs_written: int = 0
    skipped_unusable: int = 0
    skipped_no_frame: int = 0
    skipped_no_face: int = 0
    images_written: int = 0
    bytes_written: int = 0
    disagreements: int = 0


def resolve_labels(dataset: str) -> tuple[dict[str, LabelRecord], int]:
    """
    Zwraca obowiązujący werdykt każdej pary i liczbę par spornych.

    Scalanie po polach (jak w `latest_by_pair`), ale dodatkowo liczy pary, które
    oceniło więcej niż jedną osobą różniąc się werdyktem AU. Takie pary są
    materiałem na zgodność między anotatorami i nie wolno ich cicho uśrednić.

    Args:
        dataset: Nazwa zbioru

    Returns:
        Para (mapa `pair_key` → werdykt, liczba par spornych)
    """
    by_pair: dict[str, list[LabelRecord]] = {}
    for record in read_all(dataset):
        by_pair.setdefault(record.pair_key, []).append(record)

    resolved: dict[str, LabelRecord] = {}
    disagreements = 0
    for pair_key, records in by_pair.items():
        resolved[pair_key] = _merge_records(records)
        if _has_disagreement(records):
            disagreements += 1
    return resolved, disagreements


def _merge_records(records: list[LabelRecord]) -> LabelRecord:
    """
    Składa kolejne decyzje o jednej parze w stan obowiązujący.

    Args:
        records: Rekordy tej samej pary, rosnąco po czasie

    Returns:
        Rekord obowiązujący
    """
    merged = records[0]
    for record in records[1:]:
        for field_name in ("keypoints", "breed", "emotion", "keypoints_ok"):
            if getattr(record, field_name) is None:
                setattr(record, field_name, getattr(merged, field_name))
        merged = record
    return merged


def _has_disagreement(records: list[LabelRecord]) -> bool:
    """
    Mówi, czy różne osoby dały tej parze różne werdykty AU.

    Args:
        records: Rekordy tej samej pary

    Returns:
        True, gdy co najmniej dwie osoby oceniły to samo AU inaczej
    """
    by_annotator: dict[str, dict[str, str]] = {}
    for record in records:
        by_annotator[record.annotator] = record.au_verdicts
    if len(by_annotator) < 2:
        return False
    verdicts = list(by_annotator.values())
    return any(
        first.get(au) != second.get(au)
        for first, second in zip(verdicts, verdicts[1:])
        for au in set(first) | set(second)
    )


def _au_labels(record: LabelRecord) -> dict[str, str]:
    """
    Zamienia werdykty człowieka na etykiety binarne z luką na „nie wiadomo".

    Args:
        record: Werdykt obowiązujący dla pary

    Returns:
        Mapa kod AU → "1" | "0" | ""
    """
    return {
        au: VERDICT_TO_CSV.get(record.au_verdicts.get(au, "not_observable"), "")
        for au in ACTION_UNIT_NAMES
    }


@dataclass
class CuratedIndex:
    """Kuracja rozłożona na mapy, po których szuka się pary w jednym kroku."""

    images: dict[int, dict]
    peak_by_file: dict[str, dict]
    by_image_track: dict[tuple[int, int], dict]


def index_curated(coco: dict) -> CuratedIndex:
    """
    Buduje mapy dostępu do kuracji.

    Kuracja powiela anotację klatki neutralnej — jedna neutralna obsługuje kilka
    peaków tego samego psa — więc mapy z natury nadpisują duplikaty tą samą
    treścią i to jest w porządku.

    Args:
        coco: Wczytany `curated.json`

    Returns:
        Indeks kuracji
    """
    images = {image["id"]: image for image in coco["images"]}
    peak_by_file: dict[str, dict] = {}
    by_image_track: dict[tuple[int, int], dict] = {}
    for annotation in coco["annotations"]:
        by_image_track[(annotation["image_id"], annotation["track_id"])] = annotation
        if annotation.get("frame_role") == "peak":
            peak_by_file[images[annotation["image_id"]]["file_name"]] = annotation
    return CuratedIndex(images=images, peak_by_file=peak_by_file, by_image_track=by_image_track)


def _neutral_of(index: CuratedIndex, peak: dict) -> Optional[dict]:
    """
    Znajduje anotację klatki neutralnej tego samego psa.

    `neutral_frame_id` wskazuje OBRAZ, nie anotację, więc psa dobiera się po
    `track_id` — bez tego przy kilku psach w kadrze baza AU wzięłaby się od
    sąsiada.

    Args:
        index: Indeks kuracji
        peak: Anotacja klatki szczytowej

    Returns:
        Anotacja neutralna albo None
    """
    neutral_image_id = peak.get("neutral_frame_id")
    if neutral_image_id is None:
        return None
    return index.by_image_track.get((neutral_image_id, peak["track_id"]))


def write_crop(
    annotation: dict,
    image_entry: dict,
    frames_root: Path,
    images_root: Path,
    keypoints: list[float],
) -> Optional[tuple[dict, list[float], int]]:
    """
    Zapisuje kadr mordy i przelicza punkty do jego układu.

    Args:
        annotation: Anotacja klatki
        image_entry: Wpis obrazu z kuracji
        frames_root: Katalog pełnych klatek
        images_root: Katalog wynikowy obrazów
        keypoints: Punkty obowiązujące (po poprawkach człowieka)

    Returns:
        Trójka (wpis obrazu, punkty w układzie kadru, rozmiar pliku) albo None,
        gdy klatki nie ma na dysku lub nie da się wyznaczyć kadru mordy
    """
    source = frames_root / image_entry["file_name"]
    image = read_image(source)
    if image is None:
        return None

    box = face_box(keypoints, image.shape[1], image.shape[0], FACE_MARGIN)
    if box is None:
        return None

    crop, scale = crop_and_scale(image, box, MAX_CROP_SIDE)
    target = images_root / image_entry["file_name"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if not write_jpeg(target, crop, JPEG_QUALITY):
        return None

    entry = {
        "file_name": image_entry["file_name"],
        "width": int(crop.shape[1]),
        "height": int(crop.shape[0]),
        "license": LICENSES[0]["id"],
        "source_video": image_entry.get("source_video"),
        "frame_number": image_entry.get("frame_number"),
        # Położenie kadru w pełnej klatce — bez tego powrót do oryginału jest
        # niemożliwy, a z tym wystarczy nagranie z `data/drive_dogs/`.
        "source_bbox": [box.x0, box.y0, box.width, box.height],
        "source_scale": scale,
        "source_size": [image_entry.get("width"), image_entry.get("height")],
    }
    return entry, remap_keypoints(keypoints, box, scale), file_size(target)


def build_annotation(
    source: dict,
    keypoints: list[float],
    ids: tuple[int, int, Optional[int]],
    record: Optional[LabelRecord],
) -> dict:
    """
    Składa anotację COCO z pomiaru pipeline'u i werdyktu człowieka.

    Werdykt człowieka (`au_verdicts`) trafia obok pomiaru reguł (`au_analysis`),
    nigdy zamiast niego: reguły są pomiarem geometrii i zostają jako materiał
    porównawczy, a etykietą zbioru jest tylko ocena człowieka.

    Args:
        source: Anotacja z kuracji
        keypoints: Punkty w układzie kadru
        ids: Trójka (id anotacji, id obrazu, id obrazu neutralnego)
        record: Werdykt człowieka albo None dla klatki neutralnej

    Returns:
        Anotacja COCO
    """
    annotation_id, image_id, neutral_image_id = ids
    bbox = bbox_from_keypoints(keypoints)
    annotation = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": bbox,
        "area": bbox[2] * bbox[3],
        "iscrowd": 0,
        "keypoints": keypoints,
        "num_keypoints": int(sum(1 for i in range(NUM_KEYPOINTS) if keypoints[i * 3 + 2] > 0)),
        "track_id": source["track_id"],
        "frame_role": source.get("frame_role"),
        "neutral_frame_id": neutral_image_id,
        "source_body_bbox": source.get("bbox"),
        "breed": source.get("breed"),
        "emotion": source.get("emotion"),
        "au_analysis": source.get("au_analysis", {}),
        "au_noise": source.get("au_noise"),
        "au_sample_count": source.get("au_sample_count"),
        "label_source": "auto_rules",
    }
    if record is not None:
        annotation.update(
            {
                "au_verdicts": record.au_verdicts,
                "breed": record.breed or source.get("breed"),
                "emotion": record.emotion or source.get("emotion"),
                "keypoints_ok": record.keypoints_ok,
                "annotator": record.annotator,
                "verified_at": record.timestamp,
                "label_source": "human_verified",
            }
        )
    return annotation


def _effective_keypoints(source: dict, record: Optional[LabelRecord]) -> list[float]:
    """
    Wybiera punkty obowiązujące: poprawione ręcznie, a w ich braku z pipeline'u.

    Args:
        source: Anotacja z kuracji
        record: Werdykt człowieka albo None

    Returns:
        Płaska lista 138 wartości
    """
    if record is not None and record.keypoints:
        return list(record.keypoints)
    return list(source["keypoints"])


def _csv_row(
    pair_key: str,
    record: LabelRecord,
    keypoints: list[float],
    annotation: dict,
) -> dict[str, object]:
    """
    Buduje wiersz CSV pod trening sieci AU (Sprint 16).

    Wejściem sieci jest kształt po Prokrustesie, a nie surowe piksele: rozmiar
    i obrót głowy nie są cechą emocji, a bez normalizacji zdominowałyby wejście.
    Liczymy go z punktów OBOWIĄZUJĄCYCH, więc ręczna poprawka trafia do treningu.

    Args:
        pair_key: Ścieżka klatki szczytowej
        record: Werdykt człowieka
        keypoints: Punkty w układzie kadru
        annotation: Gotowa anotacja COCO

    Returns:
        Wiersz jako słownik kolumn
    """
    from packages.models.shape_normalization import procrustes_align
    from packages.pipeline.quality_gate import load_mean_shape

    aligned = procrustes_align(np.asarray(keypoints, dtype=float), load_mean_shape())
    row: dict[str, object] = {
        "pair_key": pair_key,
        "annotator": record.annotator,
        "breed": annotation["breed"],
        "emotion": annotation["emotion"],
    }
    for index in range(NUM_KEYPOINTS):
        row[f"kp{index}_x"] = round(float(aligned[index * 3]), 6)
        row[f"kp{index}_y"] = round(float(aligned[index * 3 + 1]), 6)
        row[f"kp{index}_v"] = round(float(aligned[index * 3 + 2]), 4)
    row.update(_au_labels(record))
    return row


class FinalDatasetBuilder:
    """
    Składa katalog `dataset_final` z kuracji, dziennika etykiet i klatek.

    Trzyma numerację COCO i mapę już zapisanych obrazów: jedna klatka neutralna
    obsługuje kilka peaków, więc bez tej mapy trafiłaby do zbioru kilka razy pod
    różnymi `image_id` i psuła statystyki.
    """

    def __init__(
        self,
        dataset: str,
        output: Path,
        include_unusable: bool = False,
        data_root: Optional[Path] = None,
    ) -> None:
        """
        Args:
            dataset: Nazwa zbioru w katalogu danych
            output: Katalog wynikowy
            include_unusable: Czy dołączać pary odrzucone przez człowieka
            data_root: Katalog danych; None znaczy `data/` w repozytorium
        """
        self.dataset = dataset
        self.output = output
        self.include_unusable = include_unusable
        self.dataset_dir = (data_root or REPO_ROOT / "data") / dataset
        self.frames_root = self.dataset_dir / "frames"
        self.images_root = output / IMAGES_DIRNAME
        self.stats = BuildStats()
        self.rows: list[dict[str, object]] = []
        self.coco: dict = {}
        self.index: Optional[CuratedIndex] = None
        self._labels: dict[str, LabelRecord] = {}
        self._image_id_by_file: dict[str, int] = {}
        self._keypoints_in_crop: dict[int, list[float]] = {}
        self._annotation_by_key: dict[tuple[int, int, str], dict] = {}

    def run(self) -> BuildStats:
        """
        Wykonuje całe złożenie.

        Returns:
            Statystyki przebiegu

        Raises:
            SystemExit: Gdy nie ma pliku kuracji
        """
        curated_path = self.dataset_dir / "curated.json"
        if not curated_path.is_file():
            raise SystemExit(f"Brak kuracji {curated_path} — uruchom curate_for_review")

        self.index = index_curated(json.loads(curated_path.read_text(encoding="utf-8")))
        self._labels, self.stats.disagreements = resolve_labels(self.dataset)

        # Parą jest tylko to, co człowiek OCENIŁ jako parę. Dziennik zawiera też
        # rekordy pod ścieżką klatki neutralnej — powstają, gdy poprawiano na niej
        # punkty — i wzięcie ich za pary dałoby zbiór z etykietami spoczynku.
        pairs = {
            key: record
            for key, record in self._labels.items()
            if key in self.index.peak_by_file
        }
        self.stats.pairs_labeled = len(pairs)

        if self.output.exists():
            remove_tree(self.output)
        self.images_root.mkdir(parents=True)
        self.coco = self._empty_coco()

        for pair_key, record in sorted(pairs.items()):
            if not record.usable and not self.include_unusable:
                self.stats.skipped_unusable += 1
                continue
            self._emit_pair(pair_key, record)

        self._write_outputs()
        return self.stats

    def _empty_coco(self) -> dict:
        """Tworzy szkielet pliku COCO z pełnymi metadanymi."""
        return {
            "info": {
                "description": "Dog FACS Dataset — kadry mordy z weryfikacją człowieka",
                "url": "https://github.com/eternaki/group-project",
                "version": "1.0",
                "year": datetime.now(timezone.utc).year,
                "contributor": "Politechnika Gdańska WETI",
                "date_created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
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
                {"id": index, "name": name} for index, name in enumerate(EMOTION_CLASSES)
            ],
            "action_units": list(ACTION_UNIT_NAMES),
            "images": [],
            "annotations": [],
        }

    def _emit_pair(self, pair_key: str, record: LabelRecord) -> None:
        """
        Zapisuje obie klatki pary i ich anotacje.

        Args:
            pair_key: Ścieżka klatki szczytowej
            record: Werdykt obowiązujący
        """
        assert self.index is not None
        peak_source = self.index.peak_by_file.get(pair_key)
        if peak_source is None:
            self.stats.skipped_no_frame += 1
            return
        neutral_source = _neutral_of(self.index, peak_source)
        if neutral_source is None:
            self.stats.skipped_no_frame += 1
            return

        # Człowiek mógł stwierdzić, że pipeline pomylił role. Wtedy wyrazem jest
        # klatka nazwana neutralną i to do NIEJ należy werdykt — inaczej etykieta
        # opisywałaby spoczynek.
        expression, baseline = (
            (neutral_source, peak_source) if record.roles_swapped else (peak_source, neutral_source)
        )

        baseline_ann = self._emit_frame(baseline, None, None, "neutral")
        if baseline_ann is None:
            self.stats.skipped_no_face += 1
            return
        expression_ann = self._emit_frame(
            expression, record, baseline_ann["image_id"], "peak"
        )
        if expression_ann is None:
            self.stats.skipped_no_face += 1
            return

        self.rows.append(
            _csv_row(pair_key, record, expression_ann["keypoints"], expression_ann)
        )
        self.stats.pairs_written += 1

    def _correction_for(self, file_name: str) -> Optional[LabelRecord]:
        """
        Znajduje ręczną poprawkę punktów dla KONKRETNEJ klatki.

        Poprawka punktów zapisuje się pod ścieżką klatki, którą poprawiano — także
        wtedy, gdy była to klatka neutralna. Szukanie jej po kluczu pary zgubiłoby
        każdą poprawkę bazy AU, czyli tę, która psuje najwięcej: błędna neutralna
        przesuwa wszystkie 21 AU tego psa naraz.

        Args:
            file_name: Ścieżka klatki względem katalogu klatek

        Returns:
            Rekord z poprawionymi punktami albo None
        """
        record = self._labels.get(file_name)
        return record if record is not None and record.keypoints else None

    def _emit_frame(
        self,
        source: dict,
        verdict: Optional[LabelRecord],
        neutral_image_id: Optional[int],
        role: str,
    ) -> Optional[dict]:
        """
        Zapisuje kadr mordy jednej klatki i dopisuje jej anotację.

        Args:
            source: Anotacja z kuracji
            verdict: Werdykt pary, gdy to klatka wyrazu; None dla bazy
            neutral_image_id: Identyfikator obrazu bazowego albo None
            role: Rola klatki w parze po uwzględnieniu zamiany ról

        Returns:
            Anotacja COCO albo None, gdy klatki nie da się zapisać
        """
        assert self.index is not None
        image_entry = self.index.images[source["image_id"]]
        file_name = image_entry["file_name"]
        keypoints = _effective_keypoints(source, self._correction_for(file_name))

        image_id = self._write_image(source, image_entry, keypoints)
        if image_id is None:
            return None

        key = (image_id, source["track_id"], role)
        existing = self._annotation_by_key.get(key)
        if existing is not None:
            return existing

        annotation = build_annotation(
            {**source, "frame_role": role},
            self._keypoints_in_crop[image_id],
            (len(self.coco["annotations"]) + 1, image_id, neutral_image_id),
            verdict,
        )
        self.coco["annotations"].append(annotation)
        self._annotation_by_key[key] = annotation
        return annotation

    def _write_image(
        self, source: dict, image_entry: dict, keypoints: list[float]
    ) -> Optional[int]:
        """
        Zapisuje kadr mordy raz na klatkę i zwraca jej identyfikator.

        Jedna klatka neutralna obsługuje kilka peaków, więc bez tej pamięci
        trafiłaby do zbioru kilka razy pod różnymi `image_id`.

        Args:
            source: Anotacja z kuracji
            image_entry: Wpis obrazu z kuracji
            keypoints: Punkty obowiązujące w układzie pełnej klatki

        Returns:
            Identyfikator obrazu albo None, gdy zapis się nie powiódł
        """
        file_name = image_entry["file_name"]
        cached = self._image_id_by_file.get(file_name)
        if cached is not None:
            return cached

        result = write_crop(source, image_entry, self.frames_root, self.images_root, keypoints)
        if result is None:
            return None
        entry, moved, size = result

        image_id = len(self.coco["images"]) + 1
        entry["id"] = image_id
        self.coco["images"].append(entry)
        self._image_id_by_file[file_name] = image_id
        self._keypoints_in_crop[image_id] = moved
        self.stats.images_written += 1
        self.stats.bytes_written += size
        return image_id

    def _write_outputs(self) -> None:
        """Zapisuje COCO, CSV i README."""
        (self.output / ANNOTATIONS_NAME).write_text(
            json.dumps(self.coco, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_csv(self.rows, self.output / CSV_NAME)
        write_readme(self.output / README_NAME, self.stats, self.dataset, self.coco)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    """
    Zapisuje tabelę pod trening sieci AU.

    Args:
        rows: Wiersze z `_csv_row`
        path: Ścieżka pliku CSV
    """
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path: Path, stats: BuildStats, dataset: str, coco: dict) -> None:
    """
    Zapisuje opis zbioru — co w nim jest i czego w nim NIE ma.

    Args:
        path: Ścieżka README
        stats: Statystyki przebiegu
        dataset: Nazwa zbioru źródłowego
        coco: Złożony plik COCO
    """
    emotions = Counter(
        annotation.get("emotion")
        for annotation in coco["annotations"]
        if annotation.get("label_source") == "human_verified"
    )
    active = Counter(
        au
        for annotation in coco["annotations"]
        for au, verdict in annotation.get("au_verdicts", {}).items()
        if verdict == "active"
    )
    lines = [
        "# Dog FACS Dataset — zbiór finalny",
        "",
        f"Złożono: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Źródło: `data/{dataset}/curated.json` + `data/labels/{dataset}/*.jsonl`",
        "",
        "## Zawartość",
        "",
        f"- Par zweryfikowanych przez człowieka: **{stats.pairs_written}**",
        f"- Obrazów (kadry mordy): **{stats.images_written}**",
        f"- Rozmiar obrazów: **{stats.bytes_written / 1024 / 1024:.1f} MB**",
        f"- Par spornych (różni anotatorzy, różny werdykt): **{stats.disagreements}**",
        "",
        "```",
        f"{IMAGES_DIRNAME}/          kadry mordy, JPEG q{JPEG_QUALITY}, dłuższy bok <= {MAX_CROP_SIDE} px",
        f"{ANNOTATIONS_NAME}   COCO: 46 keypoints, 21 AU (reguły + werdykt człowieka)",
        f"{CSV_NAME}       tabela pod trening sieci AU (Sprint 16)",
        "```",
        "",
        "## Czego tu nie ma",
        "",
        "- **Pełnych klatek.** Obrazem jest kadr mordy — zbiór opisuje twarz, a tułów",
        "  kosztowałby trzynaście razy więcej miejsca bez jednej dodatkowej etykiety.",
        "  Powrót do oryginału: `source_video` + `frame_number` wskazują nagranie",
        "  w `data/drive_dogs/`, a `source_bbox` położenie kadru w pełnej klatce.",
        "- **Nagrań źródłowych.** COCO opisuje obrazy, nie wideo.",
        "",
        "## Jak czytać etykiety AU",
        "",
        "`au_verdicts` to ocena CZŁOWIEKA i tylko ona jest etykietą. Jest trójstanowa:",
        "`active` / `inactive` / `not_observable` — ostatnie znaczy **brak wiedzy**",
        "(np. ucho poza kadrem), a nie brak ruchu. W CSV odpowiada mu pusta komórka;",
        "potraktowanie jej jako zera nauczyłoby sieć wymyślonych negatywów.",
        "",
        "`au_analysis` to pomiar reguł geometrycznych — materiał porównawczy, NIE etykieta.",
        "Zmierzony szum tych reguł przewyższa próg aktywacji na większości par trek–AU.",
        "",
        "## Rozkłady",
        "",
        "Emocje (klatki zweryfikowane): "
        + (", ".join(f"{name} {count}" for name, count in emotions.most_common()) or "brak"),
        "",
        "AU oznaczone jako aktywne: "
        + (", ".join(f"{name} {count}" for name, count in active.most_common()) or "brak"),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parsuje argumenty wiersza poleceń."""
    parser = argparse.ArgumentParser(description="Zlozenie finalnego zbioru Dog FACS")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Nazwa zbioru w data/")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Katalog wynikowy")
    parser.add_argument(
        "--include-unusable",
        action="store_true",
        help="Dolacz pary odrzucone przez czlowieka (domyslnie pomijane)",
    )
    return parser.parse_args()


def report(stats: BuildStats, output: Path) -> None:
    """
    Wypisuje podsumowanie przebiegu.

    Args:
        stats: Statystyki złożenia
        output: Katalog wynikowy
    """
    logger.info("Par ocenionych przez czlowieka : %d", stats.pairs_labeled)
    logger.info("Par w zbiorze                  : %d", stats.pairs_written)
    logger.info("Obrazow (kadry mordy)          : %d", stats.images_written)
    logger.info("Rozmiar obrazow                : %.1f MB", stats.bytes_written / 1024 / 1024)
    if stats.skipped_unusable:
        logger.info("Pominiete (czlowiek odrzucil)  : %d", stats.skipped_unusable)
    if stats.skipped_no_frame:
        logger.info("Pominiete (brak pary w kuracji): %d", stats.skipped_no_frame)
    if stats.skipped_no_face:
        logger.info("Pominiete (brak kadru mordy)   : %d", stats.skipped_no_face)
    if stats.disagreements:
        logger.info("Pary sporne (rozni anotatorzy) : %d", stats.disagreements)
    logger.info("Gotowe: %s", output)


def main() -> None:
    """Punkt wejścia."""
    args = parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = REPO_ROOT / output
    builder = FinalDatasetBuilder(args.dataset, output, args.include_unusable)
    report(builder.run(), output)


if __name__ == "__main__":
    main()
