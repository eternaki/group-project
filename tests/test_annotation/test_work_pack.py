"""
Paczka robocza musi nieść TĘ SAMĄ kolejkę co materiał surowy.

Paczka jest jedyną postacią zbioru, jaka dociera do zespołu przez `git clone`,
więc każda para zgubiona przy jej składaniu to para, której nikt poza autorem
zbioru nigdy nie zobaczy. Strata jest cicha: karta zbioru liczy pary wprost
z pliku COCO, a kolejka powstaje z parowania (klatka neutralna, klatka
szczytowa) — więc niekompletna paczka pokazuje pełną liczbę i daje połowę.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.annotation.build_work_pack import build_pack

NUM_KEYPOINTS = 46
FRAME_SIZE = (480, 640)

# Jedna klatka neutralna obsługuje dwa peaki — układ, na którym paczka gubiła pary
NEUTRAL_FILE = "DOGS/nagranie/nagranie_t0_000000.jpg"
PEAK_FILES = (
    "DOGS/nagranie/nagranie_t0_000100.jpg",
    "DOGS/nagranie/nagranie_t0_000200.jpg",
)


def _keypoints() -> list[float]:
    """Buduje pewne keypoints WEWNĄTRZ boksu psa — tak jak morda w realnych danych."""
    flat: list[float] = []
    for index in range(NUM_KEYPOINTS):
        angle = 2 * np.pi * index / NUM_KEYPOINTS
        flat += [200.0 + 40 * np.cos(angle), 200.0 + 30 * np.sin(angle), 0.9]
    return flat


def _annotation(annotation_id: int, image_id: int, role: str, order: int) -> dict:
    """
    Buduje anotację w postaci, jaką daje kuracja.

    Args:
        annotation_id: Identyfikator anotacji
        image_id: Identyfikator obrazu
        role: `peak` albo `neutral`
        order: Pozycja w kolejce weryfikacji

    Returns:
        Anotacja COCO
    """
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": [100.0, 100.0, 200.0, 200.0],
        "area": 40000.0,
        "iscrowd": 0,
        "keypoints": _keypoints(),
        "num_keypoints": NUM_KEYPOINTS,
        "track_id": 0,
        "frame_role": role,
        "neutral_frame_id": 1,
        "review_order": order,
        "procrustes_keypoints": [0.0] * (NUM_KEYPOINTS * 3),
    }


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    """
    Tworzy zbiór, w którym JEDNA klatka neutralna obsługuje DWA peaki.

    Kuracja powiela wtedy wpis neutralny — raz przy każdym peaku — i to właśnie
    te powtórzenia paczka kiedyś odsiewała.

    Args:
        tmp_path: Katalog tymczasowy testu

    Returns:
        Katalog zbioru roboczego
    """
    dataset_dir = tmp_path / "zbior"
    frames = dataset_dir / "frames"
    for name in (NEUTRAL_FILE, *PEAK_FILES):
        path = frames / name
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.full((*FRAME_SIZE, 3), 128, dtype=np.uint8))

    images = [
        {"id": 1, "file_name": NEUTRAL_FILE, "width": 640, "height": 480,
         "source_video": "nagranie", "frame_number": 0},
        {"id": 2, "file_name": PEAK_FILES[0], "width": 640, "height": 480,
         "source_video": "nagranie", "frame_number": 100},
        {"id": 3, "file_name": PEAK_FILES[1], "width": 640, "height": 480,
         "source_video": "nagranie", "frame_number": 200},
    ]
    annotations = [
        _annotation(1, 1, "neutral", 0),
        _annotation(2, 2, "peak", 0),
        # Ta sama klatka neutralna JESZCZE RAZ — przy drugim peaku
        _annotation(1, 1, "neutral", 1),
        _annotation(3, 3, "peak", 1),
    ]
    (dataset_dir / "curated.json").write_text(
        json.dumps({"info": {}, "licenses": [], "categories": [{"id": 1, "name": "dog"}],
                    "images": images, "annotations": annotations}),
        encoding="utf-8",
    )
    return dataset_dir


def _packed(dataset_dir: Path, output: Path) -> dict:
    """Składa paczkę i zwraca jej plik COCO."""
    build_pack(dataset_dir, output)
    return json.loads((output / "curated.json").read_text(encoding="utf-8"))


class TestKolejkaPrzezywaSpakowanie:
    """Liczba par w paczce musi zgadzać się z materiałem surowym."""

    def test_kazdy_peak_ma_wlasny_wpis_neutralny(self, dataset: Path, tmp_path: Path) -> None:
        """
        Bez własnego wpisu neutralnego para nie ma z czego powstać i znika z kolejki.

        Zmierzone na realnym zbiorze przed poprawką: karta pokazywała 263 pary,
        a anotator dostawał 114.
        """
        packed = _packed(dataset, tmp_path / "work")

        roles = [a["frame_role"] for a in packed["annotations"]]
        assert roles.count("peak") == 2
        assert roles.count("neutral") == 2, (
            "wpis neutralny musi towarzyszyc KAZDEMU peakowi, tez gdy klatka jest ta sama"
        )

    def test_klatka_neutralna_zapisuje_sie_raz(self, dataset: Path, tmp_path: Path) -> None:
        """Powtarzają się wpisy w JSON, nie pliki — inaczej paczka puchnie bez potrzeby."""
        output = tmp_path / "work"
        packed = _packed(dataset, output)

        names = [image["file_name"] for image in packed["images"]]
        assert len(names) == len(set(names)) == 3
        assert len(list((output / "frames").rglob("*.jpg"))) == 3

    def test_punkty_i_boks_ida_do_ukladu_wycinka(self, dataset: Path, tmp_path: Path) -> None:
        """Punkty poza obrazem znaczyłyby, że paczka wskazuje poza własne zdjęcia."""
        output = tmp_path / "work"
        packed = _packed(dataset, output)
        images = {image["id"]: image for image in packed["images"]}

        for annotation in packed["annotations"]:
            image = images[annotation["image_id"]]
            points = np.asarray(annotation["keypoints"], dtype=float).reshape(-1, 3)
            visible = points[points[:, 2] > 0.3]
            assert (visible[:, 0] >= 0).all() and (visible[:, 0] <= image["width"]).all()
            assert (visible[:, 1] >= 0).all() and (visible[:, 1] <= image["height"]).all()

    def test_kolejnosc_pracy_przezywa(self, dataset: Path, tmp_path: Path) -> None:
        """Bez `review_order` karta zbioru nie policzy par, a kolejka straci porządek."""
        packed = _packed(dataset, tmp_path / "work")

        orders = {a["review_order"] for a in packed["annotations"]}
        assert orders == {0, 1}

    def test_nieczytane_pole_wypada(self, dataset: Path, tmp_path: Path) -> None:
        """`procrustes_keypoints` waży 6.5 MB na zbiór i nikt go nie czyta."""
        packed = _packed(dataset, tmp_path / "work")

        assert all("procrustes_keypoints" not in a for a in packed["annotations"])
