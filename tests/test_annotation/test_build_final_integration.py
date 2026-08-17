"""
Złożenie zbioru od dziennika po katalog — na sztucznym, ale kompletnym zbiorze.

Najdroższa praca anotatora to przeciąganie 46 punktów, a najłatwiej ją zgubić na
klatce NEUTRALNEJ: poprawka zapisuje się pod ścieżką tej klatki, nie pod kluczem
pary, więc szukanie jej po kluczu pary nic nie znajdzie i baza AU zostanie ta
sprzed poprawki. Ten test pilnuje właśnie tego przejścia.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.annotation.build_final_dataset import FinalDatasetBuilder

NUM_KEYPOINTS = 46
FRAME_SIZE = (480, 640)

PEAK_FILE = "DOGS/nagranie/nagranie_t0_000100.jpg"
NEUTRAL_FILE = "DOGS/nagranie/nagranie_t0_000000.jpg"


def _keypoints(center_x: float, center_y: float) -> list[float]:
    """
    Buduje pewne keypoints rozłożone wokół zadanego środka.

    Args:
        center_x: Środek w poziomie
        center_y: Środek w pionie

    Returns:
        Płaska lista 138 wartości
    """
    flat: list[float] = []
    for index in range(NUM_KEYPOINTS):
        angle = 2 * np.pi * index / NUM_KEYPOINTS
        flat += [center_x + 40 * np.cos(angle), center_y + 30 * np.sin(angle), 0.9]
    return flat


def _annotation(annotation_id: int, image_id: int, role: str, neutral_id: int) -> dict:
    """
    Buduje anotację kuracji.

    Args:
        annotation_id: Identyfikator anotacji
        image_id: Identyfikator obrazu
        role: `peak` albo `neutral`
        neutral_id: Identyfikator obrazu bazowego

    Returns:
        Anotacja w postaci, jaką daje kuracja
    """
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": [100.0, 100.0, 200.0, 200.0],
        "area": 40000.0,
        "iscrowd": 0,
        "keypoints": _keypoints(300.0, 240.0),
        "num_keypoints": NUM_KEYPOINTS,
        "track_id": 0,
        "frame_role": role,
        "neutral_frame_id": neutral_id,
        "breed": "pug",
        "emotion": "neutral",
        "au_analysis": {"AU101": {"ratio": 1.0, "is_active": False, "confidence": 0.8}},
        "au_noise": {"AU101": 0.1},
        "au_sample_count": 3,
    }


@pytest.fixture
def dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Tworzy kompletny zbiór roboczy: klatki, kurację i dziennik etykiet.

    Args:
        tmp_path: Katalog tymczasowy testu
        monkeypatch: Podmiana zmiennych środowiskowych dziennika

    Returns:
        Katalog danych
    """
    data_root = tmp_path / "data"
    frames = data_root / "zbior" / "frames"
    for name in (PEAK_FILE, NEUTRAL_FILE):
        path = frames / name
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.full((*FRAME_SIZE, 3), 128, dtype=np.uint8))

    curated = {
        "info": {},
        "licenses": [],
        "categories": [{"id": 1, "name": "dog", "keypoints": [], "skeleton": []}],
        "images": [
            {"id": 1, "file_name": NEUTRAL_FILE, "width": 640, "height": 480,
             "source_video": "nagranie", "frame_number": 0},
            {"id": 2, "file_name": PEAK_FILE, "width": 640, "height": 480,
             "source_video": "nagranie", "frame_number": 100},
        ],
        "annotations": [
            _annotation(1, 1, "neutral", 1),
            _annotation(2, 2, "peak", 1),
        ],
    }
    (data_root / "zbior" / "curated.json").write_text(
        json.dumps(curated), encoding="utf-8"
    )

    labels = data_root / "labels" / "zbior"
    labels.mkdir(parents=True)
    monkeypatch.setenv("DOGFACS_LABELS_ROOT", str(data_root / "labels"))
    return data_root


def _write_labels(data_root: Path, records: list[dict]) -> None:
    """
    Dopisuje rekordy do dziennika etykiet.

    Args:
        data_root: Katalog danych
        records: Rekordy w postaci słowników
    """
    path = data_root / "labels" / "zbior" / "anton.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _record(pair_key: str, **overrides: object) -> dict:
    """
    Buduje rekord dziennika z sensownymi wartościami domyślnymi.

    Args:
        pair_key: Ścieżka klatki, której dotyczy decyzja
        overrides: Pola do nadpisania

    Returns:
        Rekord gotowy do zapisu
    """
    record = {
        "pair_key": pair_key,
        "annotator": "anton",
        "timestamp": "2026-08-17T10:00:00+00:00",
        "au_verdicts": {"AU101": "active", "AU143": "not_observable"},
        "usable": True,
        "keypoints_ok": True,
        "breed": "mops",
        "emotion": "relaxed",
        "roles_swapped": False,
        "keypoints": None,
    }
    record.update(overrides)
    return record


def _build(dataset_root: Path, output: Path) -> dict:
    """
    Uruchamia złożenie i zwraca powstałe COCO.

    Args:
        dataset_root: Katalog danych
        output: Katalog wynikowy

    Returns:
        Wczytany plik `annotations.json`
    """
    FinalDatasetBuilder("zbior", output, data_root=dataset_root).run()
    return json.loads((output / "annotations.json").read_text(encoding="utf-8"))


def _by_role(coco: dict, role: str) -> dict:
    """Zwraca anotację o zadanej roli."""
    return next(a for a in coco["annotations"] if a["frame_role"] == role)


class TestPracaCzlowiekaTrafiaDoZbioru:
    """Werdykty i poprawki muszą przeżyć drogę z dziennika do katalogu."""

    def test_werdykt_au_staje_sie_etykieta_klatki_szczytowej(self, dataset: Path,
                                                             tmp_path: Path) -> None:
        """Bez tego zbiór nie niesie żadnej informacji od człowieka."""
        _write_labels(dataset, [_record(PEAK_FILE)])
        coco = _build(dataset, tmp_path / "out")

        peak = _by_role(coco, "peak")
        assert peak["au_verdicts"]["AU101"] == "active"
        assert peak["label_source"] == "human_verified"
        assert peak["breed"] == "mops"
        assert peak["emotion"] == "relaxed"

    def test_klatka_neutralna_nie_dostaje_etykiety_czlowieka(self, dataset: Path,
                                                             tmp_path: Path) -> None:
        """Werdykt opisuje wyraz, więc na bazie oznaczałby ruch, którego nie oceniano."""
        _write_labels(dataset, [_record(PEAK_FILE)])
        coco = _build(dataset, tmp_path / "out")

        neutral = _by_role(coco, "neutral")
        assert "au_verdicts" not in neutral
        assert neutral["label_source"] == "auto_rules"

    def test_poprawka_punktow_klatki_neutralnej_nie_ginie(self, dataset: Path,
                                                          tmp_path: Path) -> None:
        """
        Poprawka bazy zapisuje się pod ścieżką KLATKI NEUTRALNEJ, nie pary.

        Szukanie jej po kluczu pary nic nie znajduje, a błędna baza przesuwa
        wszystkie 21 AU tego psa naraz — dlatego to najdroższa cicha strata.
        """
        moved = _keypoints(200.0, 150.0)
        _write_labels(dataset, [
            _record(PEAK_FILE),
            _record(NEUTRAL_FILE, keypoints=moved, au_verdicts={}),
        ])
        coco = _build(dataset, tmp_path / "out")

        neutral_image = next(
            image for image in coco["images"] if image["file_name"] == NEUTRAL_FILE
        )
        # Kadr liczy się z punktów, więc poprawka MUSI przesunąć wycinek.
        assert neutral_image["source_bbox"][0] < 200
        assert neutral_image["source_bbox"][1] < 150

    def test_zamiana_rol_przenosi_werdykt_na_druga_klatke(self, dataset: Path,
                                                          tmp_path: Path) -> None:
        """Gdy pipeline pomylił role, etykieta ma opisywać wyraz, a nie spoczynek."""
        _write_labels(dataset, [_record(PEAK_FILE, roles_swapped=True)])
        coco = _build(dataset, tmp_path / "out")

        peak = _by_role(coco, "peak")
        image = next(i for i in coco["images"] if i["id"] == peak["image_id"])
        assert image["file_name"] == NEUTRAL_FILE

    def test_para_odrzucona_przez_czlowieka_nie_wchodzi(self, dataset: Path,
                                                        tmp_path: Path) -> None:
        """`usable=False` znaczy „kadr się nie nadaje" — to weto, nie sugestia."""
        _write_labels(dataset, [_record(PEAK_FILE, usable=False)])
        coco = _build(dataset, tmp_path / "out")

        assert coco["annotations"] == []

    def test_rekord_spod_klatki_neutralnej_nie_tworzy_pary(self, dataset: Path,
                                                           tmp_path: Path) -> None:
        """
        Sama poprawka punktów bazy nie jest oceną pary.

        Wzięta za parę dałaby zbiór, w którym etykietą jest spoczynek.
        """
        _write_labels(dataset, [_record(NEUTRAL_FILE, keypoints=_keypoints(200.0, 150.0))])
        coco = _build(dataset, tmp_path / "out")

        assert coco["annotations"] == []


class TestZgodnoscObrazuZOpisem:
    """Punkty w JSON muszą opisywać obraz leżący obok nich."""

    def test_punkty_leza_w_obrebie_zapisanego_obrazu(self, dataset: Path,
                                                     tmp_path: Path) -> None:
        """Punkt poza obrazem znaczy, że zbiór wskazuje poza własne zdjęcie."""
        _write_labels(dataset, [_record(PEAK_FILE)])
        output = tmp_path / "out"
        coco = _build(dataset, output)

        for annotation in coco["annotations"]:
            image = next(i for i in coco["images"] if i["id"] == annotation["image_id"])
            points = np.asarray(annotation["keypoints"], dtype=float).reshape(-1, 3)
            visible = points[points[:, 2] > 0.3]
            assert (visible[:, 0] >= 0).all()
            assert (visible[:, 1] >= 0).all()
            assert (visible[:, 0] <= image["width"]).all()
            assert (visible[:, 1] <= image["height"]).all()

    def test_wymiary_w_json_zgadzaja_sie_z_plikiem(self, dataset: Path,
                                                   tmp_path: Path) -> None:
        """Brak `width`/`height` albo ich rozjazd wywraca narzędzia COCO."""
        _write_labels(dataset, [_record(PEAK_FILE)])
        output = tmp_path / "out"
        coco = _build(dataset, output)

        for image in coco["images"]:
            written = cv2.imread(str(output / "images" / image["file_name"]))
            assert written is not None
            assert (written.shape[1], written.shape[0]) == (image["width"], image["height"])

    def test_klatka_neutralna_zapisuje_sie_raz(self, dataset: Path, tmp_path: Path) -> None:
        """Jedna baza obsługuje kilka peaków — powielona psułaby statystyki."""
        _write_labels(dataset, [_record(PEAK_FILE)])
        coco = _build(dataset, tmp_path / "out")

        names = [image["file_name"] for image in coco["images"]]
        assert len(names) == len(set(names))
