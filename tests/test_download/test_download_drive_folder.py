"""
Pobieranie musi PRZERWAĆ przebieg, gdy obie drogi są wyczerpane.

Odmowa jednego pliku nic nie znaczy. Seria znaczy, że klucz nie ma dostępu do
tych plików, a `gdown` wypalił limit dzienny. W obiegu uruchamianym co
kilkanaście godzin bez nadzoru mielenie takiej listy zjada godziny na samych
odmowach — zanim dojdzie do przerabiania tego, co już leży na dysku.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import scripts.download.download_drive_folder as pobieranie  # noqa: E402
from scripts.download.download_drive_folder import DriveFile, sync  # noqa: E402


def _udaj_dysk(monkeypatch, ile: int, udane: set[int]) -> list[str]:
    """
    Podstawia listowanie i pobieranie; zwraca listę nazw, o które poproszono.

    Args:
        monkeypatch: Podmieniacz pytest
        ile: Ile nagrań udaje Dysk
        udane: Indeksy nagrań, które mają się pobrać

    Returns:
        Lista nazw, dla których wywołano pobieranie
    """
    proszone: list[str] = []
    videos = [DriveFile(file_id=f"id{i}", name=f"film{i}.mp4", relative_dir="") for i in range(ile)]
    monkeypatch.setattr(pobieranie, "read_key", lambda: "klucz")
    monkeypatch.setattr(pobieranie, "list_videos", lambda *_: videos)
    monkeypatch.setattr(pobieranie, "existing_names", lambda _: set())

    def pobierz(video: DriveFile, destination: Path, key: str) -> int:
        proszone.append(video.name)
        return 100 if int(video.name[4:-4]) in udane else 0

    monkeypatch.setattr(pobieranie, "download_file", pobierz)
    return proszone


class TestPrzerwaniePoSeriiOdmow:
    """Seria odmów kończy przebieg, pojedyncze nie."""

    def test_seria_odmow_przerywa(self, monkeypatch, tmp_path: Path) -> None:
        proszone = _udaj_dysk(monkeypatch, ile=200, udane=set())

        stats = sync("folder", tmp_path)

        limit = pobieranie.FAILURE_STREAK_LIMIT
        assert len(proszone) == limit, f"mielono dalej mimo {limit} odmow z rzedu"
        assert stats.downloaded == 0

    def test_pojedyncze_odmowy_nie_przerywaja(self, monkeypatch, tmp_path: Path) -> None:
        """Co drugi plik odmawia — seria nigdy nie narasta, lecimy do konca."""
        udane = {i for i in range(40) if i % 2 == 0}
        proszone = _udaj_dysk(monkeypatch, ile=40, udane=udane)

        stats = sync("folder", tmp_path)

        assert len(proszone) == 40, "pojedyncze odmowy nie moga przerwac przebiegu"
        assert stats.downloaded == len(udane)
