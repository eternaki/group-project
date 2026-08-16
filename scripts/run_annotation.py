#!/usr/bin/env python3
"""
Uruchomienie stanowiska weryfikacji jedną komendą.

    python scripts/run_annotation.py

Robi wszystko, co inaczej trzeba by pamiętać: kuruje zbiór, jeśli go jeszcze
nie ma, ustawia zmienne środowiskowe wskazujące backendowi katalog danych,
podnosi API i frontend, a na Ctrl+C gasi oba. Sens jest taki, żeby osoba
anotująca nie konfigurowała niczego — bo każdy krok konfiguracji to godzina
mniej rozmowy z danymi.

Wymaga: `.venv` z zainstalowanym pakietem (`pip install -e .`), `npm` w PATH.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Optional

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = REPO_ROOT / "data"
FRONTEND_DIR: Path = REPO_ROOT / "apps" / "webapp" / "frontend"

# Nazwy plików, po których poznajemy zbiór wsadowy i jego wersję po kuracji
ANNOTATIONS_NAME: str = "annotations.json"
CURATED_NAME: str = "curated.json"

BACKEND_HOST: str = "127.0.0.1"
BACKEND_PORT: int = 8000
FRONTEND_URL: str = "http://localhost:5173"

# Ile sekund czekamy, aż backend wstanie, zanim podniesiemy frontend. Odwrotna
# kolejność działa, ale anotator widzi wtedy pustą stronę i myśli, że nie działa.
BACKEND_WARMUP_S: float = 3.0

# Ile czekamy, aż Vite wstanie, zanim otworzymy przeglądarkę. Otwarta za wcześnie
# pokazuje błąd połączenia i anotator uznaje, że narzędzie nie działa.
FRONTEND_WARMUP_S: float = 4.0

# Zbiór zapisany niedawno jest najpewniej wciąż dopisywany przez batch. Kuracja
# dałaby wtedy urywek materiału, a sesja zbudowana z tego urywka nie zobaczyłaby
# już par dopisanych później — anotator pracowałby na przestarzałej kolejce,
# nie wiedząc o tym.
FRESH_WRITE_WINDOW_S: float = 600.0


def _fail(message: str) -> None:
    """Kończy pracę z czytelnym komunikatem."""
    print(f"\n[BLAD] {message}", file=sys.stderr)
    sys.exit(1)


def venv_python() -> str:
    """
    Zwraca interpreter, którym da się uruchomić pipeline.

    `sys.executable` wystarcza tylko wtedy, gdy skrypt odpalono z aktywowanego
    środowiska. Odpalony z „gołej" konsoli brałby systemowego Pythona, który nie
    ma ani cv2, ani torcha — i cała praca kończyła się `ModuleNotFoundError`
    zamiast się zacząć. Dlatego najpierw szukamy `.venv` w repozytorium.

    Returns:
        Ścieżka do interpretera
    """
    candidates = [
        REPO_ROOT / ".venv" / "Scripts" / "python.exe",  # Windows
        REPO_ROOT / ".venv" / "bin" / "python",  # Linux/macOS
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def check_prerequisites() -> None:
    """
    Sprawdza, czy da się w ogóle wystartować.

    Raises:
        SystemExit: Gdy nie ma katalogu danych albo frontendu
    """
    if not DATA_DIR.is_dir():
        _fail(f"Nie ma katalogu danych {DATA_DIR}")
    if not (FRONTEND_DIR / "package.json").is_file():
        _fail(f"Nie ma frontendu w {FRONTEND_DIR}")


def find_datasets() -> list[Path]:
    """
    Znajduje zbiory wsadowe w katalogu danych.

    Anotator nie wskazuje niczego ręcznie — narzędzie samo widzi, co jest
    do zrobienia.

    Returns:
        Katalogi zbiorów zawierające `annotations.json`, najświeższe pierwsze
    """
    found = [
        path.parent
        for path in DATA_DIR.glob(f"*/{ANNOTATIONS_NAME}")
        if (path.parent / "frames").is_dir()
    ]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def _is_being_written(dataset_dir: Path) -> bool:
    """
    Zgaduje, czy batch wciąż dopisuje ten zbiór.

    Nie wystarczy patrzeć na `annotations.json`: batch zapisuje go co sto
    anotacji, więc przerwy między zapisami bywają dłuższe niż okno i zbiór
    w trakcie pracy wygląda na skończony. Katalog klatek zmienia się przy
    każdym nagraniu, czyli mniej więcej co minutę — i to jest wiarygodny
    sygnał życia.

    Args:
        dataset_dir: Katalog zbioru

    Returns:
        True, gdy coś w zbiorze zmieniało się niedawno
    """
    candidates = [dataset_dir / ANNOTATIONS_NAME]
    frames = dataset_dir / "frames"
    if frames.is_dir():
        # Klatki leżą w `frames/<zbior>/<nagranie>/`, więc sam `frames/` zmienia
        # się raz na cały przebieg. Katalog nagrania powstaje przy KAŻDYM wideo
        # i to on jest sygnałem życia — dlatego schodzimy dwa poziomy niżej.
        candidates.append(frames)
        candidates.extend(child for child in frames.iterdir() if child.is_dir())

    newest = max(
        (path.stat().st_mtime for path in candidates if path.exists()),
        default=0.0,
    )
    return time.time() - newest < FRESH_WRITE_WINDOW_S


def ensure_curated(limit: Optional[int]) -> list[Path]:
    """
    Przygotowuje do weryfikacji KAŻDY zbiór, który jeszcze nie ma kuracji.

    Args:
        limit: Najwyżej tyle par w wyniku; None znaczy wszystkie

    Returns:
        Zbiory gotowe do weryfikacji (pliki po kuracji)

    Raises:
        SystemExit: Gdy nie ma ani jednego zbioru do pracy
    """
    datasets = find_datasets()
    if not datasets:
        _fail(
            f"W {DATA_DIR} nie ma zadnego zbioru (katalogu z {ANNOTATIONS_NAME} i frames/).\n"
            "Zbior generuje: python -m scripts.annotation.batch_annotate"
        )

    ready: list[Path] = []
    for dataset_dir in datasets:
        curated = dataset_dir / CURATED_NAME
        if curated.is_file():
            print(f"[OK] {dataset_dir.name}: gotowy do weryfikacji")
            ready.append(curated)
            continue

        if _is_being_written(dataset_dir):
            print(
                f"[..] {dataset_dir.name}: pomijam — zbior jest wlasnie dopisywany "
                "przez batch. Uruchom ponownie po zakonczeniu przegonu."
            )
            continue

        print(f"[..] {dataset_dir.name}: kuruje (bramka jakosci + kolejnosc pod anotatora)")
        command = [
            venv_python(),
            "-m",
            "scripts.annotation.curate_for_review",
            "--dataset",
            str(dataset_dir / ANNOTATIONS_NAME),
            "--out",
            str(curated),
        ]
        if limit is not None:
            command += ["--limit", str(limit)]
        result = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if result.returncode == 0 and curated.is_file():
            ready.append(curated)
        else:
            print(f"[UWAGA] {dataset_dir.name}: kuracja sie nie powiodla, pomijam")

    if not ready:
        _fail("Zaden zbior nie przeszedl kuracji")
    return ready


def build_environment() -> dict[str, str]:
    """
    Buduje środowisko dla backendu.

    Backend musi wiedzieć, skąd wolno importować zbiory i skąd serwować klatki
    — jedno i drugie leży poza repozytorium i u każdego może stać gdzie indziej.

    Returns:
        Kopia środowiska z ustawionymi zmiennymi projektu
    """
    environment = dict(os.environ)
    environment["DOGFACS_IMPORT_ROOT"] = str(DATA_DIR)
    # Cały katalog danych, nie pojedynczy zbiór: anotator ma móc przełączać się
    # między zbiorami bez restartu serwera.
    environment["DOGFACS_DATASET_FRAMES"] = str(DATA_DIR)
    # Kto anotuje — trafia do nazwy pliku etykiet, żeby dwie osoby nigdy nie
    # pisały do jednego pliku i git scalał ich pracę bez konfliktu.
    environment.setdefault("DOGFACS_ANNOTATOR", os.environ.get("USERNAME", "anonim"))
    environment["PYTHONUTF8"] = "1"
    return environment


def start_backend(environment: dict[str, str]) -> subprocess.Popen:
    """
    Podnosi API.

    Args:
        environment: Środowisko z ustawionymi ścieżkami projektu

    Returns:
        Uruchomiony proces uvicorna
    """
    print(f"[..] Backend na http://{BACKEND_HOST}:{BACKEND_PORT}")
    return subprocess.Popen(
        [
            venv_python(),
            "-m",
            "uvicorn",
            "apps.webapp.backend.main:app",
            "--host",
            BACKEND_HOST,
            "--port",
            str(BACKEND_PORT),
        ],
        cwd=REPO_ROOT,
        env=environment,
    )


def start_frontend() -> subprocess.Popen:
    """
    Podnosi frontend (Vite dev server).

    Returns:
        Uruchomiony proces npm
    """
    print(f"[..] Frontend na {FRONTEND_URL}")
    # shell=True na Windowsie, bo `npm` to skrypt .cmd, a nie plik wykonywalny
    return subprocess.Popen(
        "npm run dev",
        cwd=FRONTEND_DIR,
        shell=True,
    )


def _terminate(process: Optional[subprocess.Popen], name: str) -> None:
    """Gasi proces potomny, jeśli jeszcze żyje."""
    if process is None or process.poll() is not None:
        return
    print(f"[..] Zatrzymuje {name}")
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def parse_args() -> argparse.Namespace:
    """Parsuje argumenty wiersza poleceń."""
    parser = argparse.ArgumentParser(description="Stanowisko weryfikacji AU jedna komenda")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Najwyzej tyle par w kuracji (przy pierwszym uruchomieniu)",
    )
    parser.add_argument(
        "--backend-only",
        action="store_true",
        help="Podnies samo API, bez frontendu",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Nie otwieraj przegladarki automatycznie",
    )
    return parser.parse_args()


def main() -> None:
    """Punkt wejścia: przygotowuje zbiór i podnosi oba serwery."""
    args = parse_args()
    check_prerequisites()
    ready = ensure_curated(args.limit)

    backend = start_backend(build_environment())
    frontend: Optional[subprocess.Popen] = None
    try:
        if not args.backend_only:
            time.sleep(BACKEND_WARMUP_S)
            frontend = start_frontend()
            time.sleep(FRONTEND_WARMUP_S)
            if not args.no_browser:
                webbrowser.open(FRONTEND_URL)
        print(f"\n[OK] Gotowe: {len(ready)} zbior(ow) do weryfikacji, {FRONTEND_URL}")
        print("     Ctrl+C konczy prace obu serwerow.\n")
        backend.wait()
    except KeyboardInterrupt:
        print("\n[..] Przerwano")
    finally:
        _terminate(frontend, "frontend")
        _terminate(backend, "backend")


if __name__ == "__main__":
    # Bez tego Ctrl+C na Windowsie bywa przechwytywany przez proces potomny,
    # a skrypt nadrzędny zostaje z osieroconymi serwerami.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
