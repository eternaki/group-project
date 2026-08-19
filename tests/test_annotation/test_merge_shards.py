"""
Testy scalania części masowej anotacji.

Scalanie przenumerowuje identyfikatory, bo każda część liczy je od zera. Razem
z `image_id` trzeba przemapować `neutral_frame_id` — wskazuje on OBRAZ, więc
pozostawiony bez zmian po cichu związałby anotację z klatką neutralną innego
psa z innej części. Taka usterka nie wysypuje niczego, tylko cicho psuje każdy
pomiar AU w zbiorze, dlatego ma własne testy.
"""

import json
from pathlib import Path

import pytest

from scripts.annotation.run_batch_parallel import merge_shards, shard_output_dir


def _shard(offset: int, video: str) -> dict:
    """
    Buduje zbiór COCO jednej części z parą neutral+peak.

    Args:
        offset: Przesunięcie identyfikatorów (części liczą je niezależnie)
        video: Nazwa nagrania, po której poznamy pochodzenie anotacji

    Returns:
        Słownik COCO
    """
    neutral_id, peak_id = offset + 1, offset + 2
    return {
        "info": {"description": "czesc"},
        "categories": [{"id": 1, "name": "dog"}],
        "images": [
            {"id": neutral_id, "file_name": f"{video}/neutral.jpg"},
            {"id": peak_id, "file_name": f"{video}/peak.jpg"},
        ],
        "annotations": [
            {
                "id": neutral_id,
                "image_id": neutral_id,
                "frame_role": "neutral",
                "neutral_frame_id": neutral_id,
                "video": video,
            },
            {
                "id": peak_id,
                "image_id": peak_id,
                "frame_role": "peak",
                "neutral_frame_id": neutral_id,
                "video": video,
            },
        ],
    }


@pytest.fixture
def shard_paths(tmp_path: Path) -> list[Path]:
    """Dwie części o CELOWO kolidujących identyfikatorach."""
    paths = []
    for shard, video in enumerate(("VIDEO_A", "VIDEO_B")):
        directory = shard_output_dir(tmp_path, shard)
        directory.mkdir(parents=True)
        path = directory / "annotations.json"
        # Obie części zaczynają od id=1 — bez przenumerowania nadpiszą się
        path.write_text(json.dumps(_shard(0, video)), encoding="utf-8")
        paths.append(path)
    return paths


class TestMergeShards:
    """Scalanie części z przenumerowaniem identyfikatorów."""

    def test_nic_nie_ginie(self, shard_paths: list[Path]) -> None:
        merged = merge_shards(shard_paths)
        assert len(merged["images"]) == 4
        assert len(merged["annotations"]) == 4

    def test_identyfikatory_obrazow_sa_unikalne(self, shard_paths: list[Path]) -> None:
        merged = merge_shards(shard_paths)
        ids = [image["id"] for image in merged["images"]]
        assert len(set(ids)) == len(ids)

    def test_identyfikatory_anotacji_sa_unikalne(self, shard_paths: list[Path]) -> None:
        merged = merge_shards(shard_paths)
        ids = [annotation["id"] for annotation in merged["annotations"]]
        assert len(set(ids)) == len(ids)

    def test_anotacja_wskazuje_wlasny_obraz(self, shard_paths: list[Path]) -> None:
        merged = merge_shards(shard_paths)
        images = {image["id"]: image for image in merged["images"]}
        for annotation in merged["annotations"]:
            expected_role = "neutral" if annotation["frame_role"] == "neutral" else "peak"
            assert expected_role in images[annotation["image_id"]]["file_name"]

    def test_klatka_neutralna_zostaje_we_wlasnej_czesci(
        self, shard_paths: list[Path]
    ) -> None:
        """Sedno testu: peak nie może wskazywać neutralnej z drugiego nagrania."""
        merged = merge_shards(shard_paths)
        images = {image["id"]: image for image in merged["images"]}
        for annotation in merged["annotations"]:
            own_video = annotation["video"]
            neutral_file = images[annotation["neutral_frame_id"]]["file_name"]
            assert neutral_file.startswith(f"{own_video}/")
            assert neutral_file.endswith("neutral.jpg")

    def test_metadane_przechodza(self, shard_paths: list[Path]) -> None:
        merged = merge_shards(shard_paths)
        assert merged["categories"] == [{"id": 1, "name": "dog"}]

    def test_brak_czesci_daje_czytelny_blad(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Brak wyniku"):
            merge_shards([tmp_path / "nie_ma.json"])

    def test_wiszaca_referencja_nie_wskazuje_cudzej_klatki(self, tmp_path: Path) -> None:
        """Lepszy None niż wskazanie klatki neutralnej obcego psa."""
        directory = shard_output_dir(tmp_path, 0)
        directory.mkdir(parents=True)
        broken = _shard(0, "VIDEO_A")
        broken["annotations"][1]["neutral_frame_id"] = 999
        path = directory / "annotations.json"
        path.write_text(json.dumps(broken), encoding="utf-8")

        merged = merge_shards([path])
        assert merged["annotations"][1]["neutral_frame_id"] is None
