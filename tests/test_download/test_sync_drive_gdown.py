"""
Testy dociągania folderu Drive.

Sprawdzany jest wyłącznie sposób reagowania na ODMOWY, bo to jedyne miejsce,
w którym skrypt podejmuje decyzję. Samego pobierania nie ruszamy — to cudza
biblioteka i sieć.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.download import sync_drive_gdown as sync_module
from scripts.download.sync_drive_gdown import QUOTA_FAILURE_STREAK, sync


def _listing(count: int) -> list[SimpleNamespace]:
    """Udaje listing folderu: `count` plików, których nie ma na dysku."""
    return [
        SimpleNamespace(id=f"ID{i}", path=f"neutral/plik_{i}.mp4") for i in range(count)
    ]


@pytest.fixture
def folder(monkeypatch: pytest.MonkeyPatch):
    """Podstawia listing folderu, żeby test nie dotykał sieci."""

    def _install(count: int) -> None:
        monkeypatch.setattr(
            sync_module.gdown,
            "download_folder",
            lambda **_: _listing(count),
        )

    return _install


class TestLimitPobran:
    """Seria odmów to limit Drive, a nie awaria pojedynczych plików."""

    def test_seria_odmow_przerywa_przebieg(
        self, folder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        folder(200)
        proby: list[str] = []

        def _odmowa(file_id: str, target: Path) -> int:
            proby.append(file_id)
            return 0

        monkeypatch.setattr(sync_module, "_download_one", _odmowa)
        stats = sync("FOLDER", tmp_path)

        assert stats.stopped_on_quota is True
        # Przerywamy NA progu, a nie po przemieleniu całej listy
        assert len(proby) == QUOTA_FAILURE_STREAK

    def test_pojedyncze_odmowy_nie_przerywaja(
        self, folder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Plik potrafi odmówić raz i pobrać się przy następnym — to nie limit."""
        folder(40)
        licznik = {"n": 0}

        def _co_drugi(file_id: str, target: Path) -> int:
            licznik["n"] += 1
            return 0 if licznik["n"] % 2 else 1024

        monkeypatch.setattr(sync_module, "_download_one", _co_drugi)
        stats = sync("FOLDER", tmp_path)

        assert stats.stopped_on_quota is False
        assert stats.downloaded == 20
        assert stats.failed == 20

    def test_pelny_sukces_nie_zglasza_limitu(
        self, folder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        folder(5)
        monkeypatch.setattr(sync_module, "_download_one", lambda *_: 2048)
        stats = sync("FOLDER", tmp_path)

        assert stats.stopped_on_quota is False
        assert stats.downloaded == 5
        assert stats.bytes_downloaded == 5 * 2048
