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
from pathlib import Path
from typing import Optional

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = REPO_ROOT / "data"
DATASET_DIR: Path = DATA_DIR / "dataset_v2"
CURATED_PATH: Path = DATASET_DIR / "curated.json"
FRAMES_DIR: Path = DATASET_DIR / "frames"
FRONTEND_DIR: Path = REPO_ROOT / "apps" / "webapp" / "frontend"

BACKEND_HOST: str = "127.0.0.1"
BACKEND_PORT: int = 8000
FRONTEND_URL: str = "http://localhost:5173"

# Ile sekund czekamy, aż backend wstanie, zanim podniesiemy frontend. Odwrotna
# kolejność działa, ale anotator widzi wtedy pustą stronę i myśli, że nie działa.
BACKEND_WARMUP_S: float = 3.0


def _fail(message: str) -> None:
    """Kończy pracę z czytelnym komunikatem."""
    print(f"\n[BLAD] {message}", file=sys.stderr)
    sys.exit(1)


def check_prerequisites() -> None:
    """
    Sprawdza, czy da się w ogóle wystartować.

    Raises:
        SystemExit: Gdy brakuje klatek zbioru albo katalogu frontendu
    """
    if not FRAMES_DIR.is_dir():
        _fail(
            f"Nie ma klatek zbioru w {FRAMES_DIR}.\n"
            "Zbior generuje scripts/annotation/batch_annotate.py"
        )
    if not (FRONTEND_DIR / "package.json").is_file():
        _fail(f"Nie ma frontendu w {FRONTEND_DIR}")


def ensure_curated(limit: Optional[int]) -> None:
    """
    Kuruje zbiór, jeśli jeszcze nie ma pliku po kuracji.

    Args:
        limit: Najwyżej tyle par w wyniku; None znaczy wszystkie
    """
    if CURATED_PATH.is_file():
        print(f"[OK] Zbior po kuracji juz jest: {CURATED_PATH}")
        return

    print("[..] Kuruje zbior (bramka jakosci + kolejnosc pod anotatora)...")
    command = [
        sys.executable,
        "-m",
        "scripts.annotation.curate_for_review",
        "--out",
        str(CURATED_PATH),
    ]
    if limit is not None:
        command += ["--limit", str(limit)]
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode != 0 or not CURATED_PATH.is_file():
        _fail("Kuracja zbioru nie powiodla sie")


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
    environment["DOGFACS_DATASET_FRAMES"] = str(FRAMES_DIR)
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
            sys.executable,
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
    return parser.parse_args()


def main() -> None:
    """Punkt wejścia: przygotowuje zbiór i podnosi oba serwery."""
    args = parse_args()
    check_prerequisites()
    ensure_curated(args.limit)

    backend = start_backend(build_environment())
    frontend: Optional[subprocess.Popen] = None
    try:
        if not args.backend_only:
            time.sleep(BACKEND_WARMUP_S)
            frontend = start_frontend()
        print(f"\n[OK] Gotowe. Otworz {FRONTEND_URL} i wybierz zakladke 'Weryfikacja AU'.")
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
