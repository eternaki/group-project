"""
Odwrócenie spakowania musi wracać dokładnie tam, skąd wyszło.

Kolejka wydana anotatorom jest paczką — jej współrzędne są w układzie wycinka.
Podana jako baza do ponownej kuracji trafia do pakowania po raz drugi i wycinek
ląduje w lewym górnym rogu pełnej klatki: anotator dostaje kadr podłogi
z punktami na drzwiach. Zmierzone 29.08.2026 na wszystkich odziedziczonych
parach.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.annotation.cropping import (  # noqa: E402
    CropBox,
    crop_and_scale,
    remap_bbox,
    remap_keypoints,
    write_jpeg,
)
from scripts.annotation.queue_unpack import is_packed, unpack_queue  # noqa: E402


def _klatka(tmp_path: Path, nazwa: str, szerokosc: int, wysokosc: int) -> Path:
    """Zapisuje pełną klatkę o zadanych wymiarach."""
    sciezka = tmp_path / nazwa
    write_jpeg(sciezka, np.full((wysokosc, szerokosc, 3), 120, dtype=np.uint8))
    return sciezka


class TestOdwroceniePakowania:
    """Współrzędne muszą wrócić do układu pełnej klatki."""

    def test_punkty_wracaja_na_swoje_miejsce(self, tmp_path: Path) -> None:
        _klatka(tmp_path, "pies.jpg", 1280, 720)
        box = CropBox(x0=300, y0=150, x1=900, y1=600)
        obraz = np.full((720, 1280, 3), 7, dtype=np.uint8)
        _, skala = crop_and_scale(obraz, box, 512)

        # `remap_keypoints` wymaga kompletu 46 punktów — dwa pierwsze niosą test
        oryginalne = [700.0, 300.0, 1.0, 800.0, 400.0, 0.9] + [600.0, 350.0, 0.8] * 44
        spakowane = remap_keypoints(oryginalne, box, skala)
        kolejka = {
            "images": [
                {
                    "id": 1,
                    "file_name": "pies.jpg",
                    "width": int(box.width * skala),
                    "height": int(box.height * skala),
                    "source_bbox": [box.x0, box.y0, box.width, box.height],
                    "source_scale": skala,
                }
            ],
            "annotations": [{"id": 1, "image_id": 1, "keypoints": spakowane}],
        }

        wynik = unpack_queue(kolejka, tmp_path)

        wrocone = wynik["annotations"][0]["keypoints"]
        assert wrocone[:2] == [700.0, 300.0] or np.allclose(wrocone[:2], [700.0, 300.0], atol=1.0)
        assert np.allclose(wrocone[3:5], [800.0, 400.0], atol=1.0)

    def test_boks_wraca_na_swoje_miejsce(self, tmp_path: Path) -> None:
        _klatka(tmp_path, "pies.jpg", 1280, 720)
        box = CropBox(x0=300, y0=150, x1=900, y1=600)
        _, skala = crop_and_scale(np.full((720, 1280, 3), 7, dtype=np.uint8), box, 512)
        spakowany = remap_bbox([400.0, 200.0, 300.0, 250.0], box, skala)
        kolejka = {
            "images": [
                {
                    "id": 1,
                    "file_name": "pies.jpg",
                    "width": 512,
                    "height": 384,
                    "source_bbox": [box.x0, box.y0, box.width, box.height],
                    "source_scale": skala,
                }
            ],
            "annotations": [{"id": 1, "image_id": 1, "bbox": spakowany}],
        }

        wynik = unpack_queue(kolejka, tmp_path)

        assert np.allclose(wynik["annotations"][0]["bbox"], [400.0, 200.0, 300.0, 250.0], atol=1.0)

    def test_wymiary_wracaja_do_pelnej_klatki(self, tmp_path: Path) -> None:
        _klatka(tmp_path, "pies.jpg", 1280, 720)
        kolejka = {
            "images": [
                {
                    "id": 1,
                    "file_name": "pies.jpg",
                    "width": 512,
                    "height": 384,
                    "source_bbox": [300, 150, 600, 450],
                    "source_scale": 0.85,
                }
            ],
            "annotations": [],
        }

        obraz = unpack_queue(kolejka, tmp_path)["images"][0]

        assert (obraz["width"], obraz["height"]) == (1280, 720)
        assert "source_bbox" not in obraz, "ślad pakowania musi zniknąć"

    def test_obraz_bez_sladu_pakowania_zostaje_nietkniety(self, tmp_path: Path) -> None:
        _klatka(tmp_path, "pies.jpg", 1280, 720)
        kolejka = {
            "images": [{"id": 1, "file_name": "pies.jpg", "width": 1280, "height": 720}],
            "annotations": [{"id": 1, "image_id": 1, "keypoints": [10.0, 20.0, 1.0]}],
        }

        wynik = unpack_queue(kolejka, tmp_path)

        assert wynik["annotations"][0]["keypoints"] == [10.0, 20.0, 1.0]

    def test_brak_pelnej_klatki_nie_zmysla_wymiarow(self, tmp_path: Path) -> None:
        """Bez klatki na dysku nie wiadomo, do czego wracać — zostawiamy jak jest."""
        kolejka = {
            "images": [
                {
                    "id": 1,
                    "file_name": "nie-ma.jpg",
                    "width": 512,
                    "height": 384,
                    "source_bbox": [300, 150, 600, 450],
                    "source_scale": 0.85,
                }
            ],
            "annotations": [{"id": 1, "image_id": 1, "keypoints": [10.0, 20.0, 1.0]}],
        }

        wynik = unpack_queue(kolejka, tmp_path)

        assert wynik["images"][0]["width"] == 512
        assert wynik["annotations"][0]["keypoints"] == [10.0, 20.0, 1.0]

    def test_rozpoznaje_spakowana_kolejke(self) -> None:
        assert is_packed({"images": [{"source_bbox": [0, 0, 10, 10]}]}) is True
        assert is_packed({"images": [{"width": 100}]}) is False
