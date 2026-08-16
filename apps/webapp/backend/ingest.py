"""
Dosypywanie nagrań: przyjęcie plików i praca w tle obok anotacji.

Nagrania trafiają do JEDNEGO wspólnego katalogu — dzieli się nie materiał, tylko
przydział, a ten liczy się ze skrótu nazwy nagrania (patrz `annotators`). Dzięki
temu dorzucenie plików nie rusza przydziału tych, które już są w pracy.

Przetwarzanie idzie w ODDZIELNYM PROCESIE, nie w backendzie. Jedno nagranie to
około siedemdziesięciu sekund na tym sprzęcie; puszczone w procesie serwera
zablokowałoby anotację na cały przebieg — czyli dokładnie to, czego ta funkcja
ma unikać. Osobny proces przeżywa też restart backendu i zamknięcie przeglądarki.
"""

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Wspólny katalog nagrań całego zespołu
VIDEOS_ROOT_ENV: str = "DOGFACS_VIDEOS_ROOT"
DEFAULT_VIDEOS_ROOT: str = "data/videos"

# Podkatalog, w którym batch spodziewa się nagrań (jego nazwa staje się etykietą
# źródła w ścieżkach klatek)
VIDEOS_SUBDIR: str = "DOGS"

# Gdzie ląduje zbiór budowany z dosypanych nagrań
INBOX_DATASET_ENV: str = "DOGFACS_INBOX_DATASET"
DEFAULT_INBOX_DATASET: str = "data/dataset_manual"

# Rozszerzenia, które w ogóle przyjmujemy
ALLOWED_SUFFIXES: frozenset[str] = frozenset({".mp4", ".webm", ".mkv", ".avi", ".mov"})

# Kod, jaki Windows zwraca dla procesu, który jeszcze pracuje
_STILL_ACTIVE: int = 259

# Znaki niedozwolone w nazwie pliku. Nazwa nagrania wchodzi do ścieżek klatek
# i do skrótu wyznaczającego właściciela, więc musi być przewidywalna.
_UNSAFE_CHARS: str = '<>:"/\\|?*'
_SAFE_REPLACEMENT: str = "_"


class IngestError(ValueError):
    """Wyjątek, gdy pliku nie da się przyjąć do obróbki."""


@dataclass(frozen=True)
class IngestStatus:
    """
    Stan dosypywania.

    Attributes:
        videos_total: Ile nagrań leży we wspólnym katalogu
        videos_processed: Ile z nich pipeline już przerobił
        running: Czy proces przetwarzania właśnie pracuje
        pairs_ready: Ile par czeka już na anotatora w zbiorze z dosypki
    """

    videos_total: int
    videos_processed: int
    running: bool
    pairs_ready: int


def videos_root() -> Path:
    """Zwraca wspólny katalog nagrań (czytany przy każdym wywołaniu)."""
    return Path(os.environ.get(VIDEOS_ROOT_ENV, DEFAULT_VIDEOS_ROOT)).resolve()


def inbox_dataset() -> Path:
    """Zwraca katalog zbioru budowanego z dosypanych nagrań."""
    return Path(os.environ.get(INBOX_DATASET_ENV, DEFAULT_INBOX_DATASET)).resolve()


def safe_video_name(raw_name: str) -> str:
    """
    Sprowadza nazwę pliku do postaci bezpiecznej i przewidywalnej.

    Nazwa nagrania wchodzi do ścieżek klatek ORAZ do skrótu wyznaczającego
    właściciela, więc jej zmiana po fakcie przeniosłaby nagranie do innej osoby.
    Dlatego czyścimy ją raz, przy przyjęciu.

    Args:
        raw_name: Nazwa pliku podana przez przeglądarkę

    Returns:
        Nazwa bez znaków niedozwolonych i bez części katalogowej

    Raises:
        IngestError: Gdy rozszerzenie nie jest obsługiwane albo nazwa jest pusta
    """
    name = Path(raw_name).name
    cleaned = "".join(_SAFE_REPLACEMENT if char in _UNSAFE_CHARS else char for char in name)
    cleaned = cleaned.strip().strip(".")
    if not cleaned:
        raise IngestError("Pusta nazwa pliku")
    if Path(cleaned).suffix.lower() not in ALLOWED_SUFFIXES:
        raise IngestError(
            f"Nieobsługiwany format: {Path(cleaned).suffix}. "
            f"Przyjmujemy: {', '.join(sorted(ALLOWED_SUFFIXES))}"
        )
    return cleaned


def save_video(raw_name: str, content: bytes) -> Path:
    """
    Zapisuje przyjęte nagranie do wspólnego katalogu.

    Args:
        raw_name: Nazwa pliku z przeglądarki
        content: Zawartość pliku

    Returns:
        Ścieżka zapisanego pliku

    Raises:
        IngestError: Gdy nazwa albo format są nie do przyjęcia
    """
    target_dir = videos_root() / VIDEOS_SUBDIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / safe_video_name(raw_name)
    path.write_bytes(content)
    return path


def list_videos() -> list[Path]:
    """
    Wylicza nagrania leżące we wspólnym katalogu.

    Returns:
        Ścieżki nagrań; pusta lista, gdy katalogu jeszcze nie ma
    """
    root = videos_root()
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES
    )


def _progress_path() -> Path:
    """Ścieżka pliku postępu przetwarzania dosypki."""
    return inbox_dataset() / "progress.json"


def processed_count() -> int:
    """
    Liczy nagrania, które pipeline już przerobił.

    Returns:
        Liczba przetworzonych nagrań; 0, gdy przetwarzania jeszcze nie było
    """
    path = _progress_path()
    if not path.is_file():
        return 0
    try:
        with open(path, encoding="utf-8") as handle:
            return len(json.load(handle).get("processed_videos", []))
    except (json.JSONDecodeError, OSError):
        return 0


def pairs_ready() -> int:
    """
    Liczy pary gotowe do anotacji w zbiorze z dosypki.

    Returns:
        Liczba par; 0, gdy kuracji jeszcze nie było
    """
    path = inbox_dataset() / "curated.json"
    if not path.is_file():
        return 0
    try:
        with open(path, encoding="utf-8") as handle:
            coco = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return 0
    return len(
        {
            annotation.get("review_order")
            for annotation in coco.get("annotations", [])
            if annotation.get("review_order") is not None
        }
    )


def _lock_path() -> Path:
    """Ścieżka pliku blokady — trzyma PID pracującego procesu."""
    return inbox_dataset() / "worker.pid"


def is_running() -> bool:
    """
    Mówi, czy przetwarzanie właśnie pracuje.

    Sprawdzamy, czy proces o zapisanym PID nadal żyje: sam plik blokady zostaje
    po nagłym zamknięciu (wyłączenie komputera, zabicie procesu) i bez tej
    weryfikacji zablokowałby dosypywanie na zawsze.

    Returns:
        True, gdy proces przetwarzania działa
    """
    path = _lock_path()
    if not path.is_file():
        return False
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return False
    return _pid_alive(pid)


def _pid_alive(pid: int) -> bool:
    """
    Sprawdza, czy proces o danym PID żyje.

    Na Windowsie pytamy system wprost przez `OpenProcess`, a nie przez
    `tasklist`: backend bywa uruchomiony jako proces odłączony z przekierowanymi
    strumieniami, a wtedy `subprocess` potrafi zwrócić puste wyjście i sprawdzenie
    wywracało się na `None`.

    Args:
        pid: Identyfikator procesu

    Returns:
        True, gdy proces istnieje
    """
    if pid <= 0:
        return False

    if os.name == "nt":
        import ctypes

        # PROCESS_QUERY_LIMITED_INFORMATION — najmniejsze prawo, jakie wystarcza
        access = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(access, False, pid)
        if not handle:
            return False
        exit_code = ctypes.c_ulong()
        alive = bool(
            ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        ) and exit_code.value == _STILL_ACTIVE
        ctypes.windll.kernel32.CloseHandle(handle)
        return alive

    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def start_processing(repo_root: Path, python_executable: Optional[str] = None) -> int:
    """
    Uruchamia przetwarzanie dosypanych nagrań w tle.

    Args:
        repo_root: Korzeń repozytorium (katalog roboczy procesu)
        python_executable: Interpreter; domyślnie ten, który uruchomił backend

    Returns:
        PID uruchomionego procesu

    Raises:
        IngestError: Gdy przetwarzanie już trwa albo nie ma czego przetwarzać
    """
    if is_running():
        raise IngestError("Przetwarzanie już trwa")
    if not list_videos():
        raise IngestError("Brak nagrań do przetworzenia")

    dataset = inbox_dataset()
    dataset.mkdir(parents=True, exist_ok=True)

    command = [
        python_executable or sys.executable,
        "-m",
        "scripts.annotation.process_inbox",
        "--videos",
        str(videos_root()),
        "--dataset",
        str(dataset),
    ]
    # Bez odłączenia proces ginie razem z backendem, a przebieg trwa godzinami
    creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    process = subprocess.Popen(
        command,
        cwd=repo_root,
        stdout=open(dataset / "worker.log", "a", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        creationflags=creation_flags,
    )
    _lock_path().write_text(str(process.pid), encoding="utf-8")
    return process.pid


def status() -> IngestStatus:
    """
    Zwraca stan dosypywania.

    Returns:
        `IngestStatus` z licznikami i informacją, czy praca trwa
    """
    return IngestStatus(
        videos_total=len(list_videos()),
        videos_processed=processed_count(),
        running=is_running(),
        pairs_ready=pairs_ready(),
    )
