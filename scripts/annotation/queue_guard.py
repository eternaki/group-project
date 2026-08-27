"""
Sprawdzian, bez którego nie wolno opublikować kolejki.

Werdykt anotatora przypina się do `pair_key`, czyli do ŚCIEŻKI KLATKI. Gdy para
wypadnie z kolejki, dziennik w `data/labels/` zostaje NIETKNIĘTY — po prostu
przestaje się z czymkolwiek wiązać. Nic nie krzyczy, nic nie pada, a cudza
praca przepada.

Zmierzone 27.08.2026: przebudowa kolejki od zera wyrzuciła 1533 kadry i razem
z nimi 603 z 605 werdyktów dwóch osób. Wyszło to na jaw wyłącznie dlatego, że
ktoś porównał liczby RĘCZNIE przed scaleniem do `main`.

Ten moduł zamienia tamto ręczne porównanie w warunek, który musi przejść, zanim
kolejka pojedzie do ludzi — szczególnie gdy publikuje ją automat bez nadzoru.
"""

import json
from pathlib import Path

# Klucze zaczynające się tak pochodzą ze stanowiska uruchomionego na danych
# testowych (`/static/test1234/...`) i nigdy nie miały pary w zbiorze.
TEST_KEY_PREFIX: str = "/static/"


def queue_frames(queue_path: Path) -> set[str]:
    """
    Czyta ścieżki klatek obecnych w kolejce.

    Args:
        queue_path: Plik COCO kolejki (`curated.json`)

    Returns:
        Ścieżki klatek — te same wartości, którymi posługuje się `pair_key`
    """
    with open(queue_path, encoding="utf-8") as handle:
        coco = json.load(handle)
    return {image["file_name"] for image in coco.get("images", [])}


def verdict_keys(labels_dir: Path) -> set[str]:
    """
    Zbiera klucze par, o których ktokolwiek już się wypowiedział.

    Uszkodzonych wierszy nie pomijamy po cichu w nieskończoność — dziennik jest
    dopisywany wierszami, więc ostatni wiersz bywa ucięty przerwanym zapisem
    i to jedyny przypadek, który tu odpuszczamy.

    Args:
        labels_dir: Katalog z dziennikami (`data/labels/`)

    Returns:
        Klucze par bez wpisów testowych
    """
    keys: set[str] = set()
    if not labels_dir.is_dir():
        return keys
    for journal in labels_dir.rglob("*.jsonl"):
        for line in journal.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = record.get("pair_key", "")
            if key and not key.startswith(TEST_KEY_PREFIX):
                keys.add(key)
    return keys


def orphaned_verdicts(queue_path: Path, labels_dir: Path) -> set[str]:
    """
    Wskazuje werdykty, które w tej kolejce nie mają już swojej pary.

    Args:
        queue_path: Plik COCO kolejki
        labels_dir: Katalog z dziennikami

    Returns:
        Klucze werdyktów bez pokrycia w kolejce
    """
    return verdict_keys(labels_dir) - queue_frames(queue_path)


def assert_no_new_orphans(
    queue_path: Path, labels_dir: Path, allowed: int
) -> set[str]:
    """
    Przerywa publikację, gdy kolejka gubi więcej werdyktów, niż gubiła dotąd.

    Progu nie ustawiamy na zero, bo część werdyktów osierociała wcześniej
    (przebieg z 21.08) i tego się już nie odwróci. Pilnujemy PRZYROSTU:
    nowa kolejka nie ma prawa zgubić ani jednego werdyktu więcej.

    Args:
        queue_path: Plik COCO kolejki do opublikowania
        labels_dir: Katalog z dziennikami
        allowed: Ile werdyktów wolno nie odnaleźć (stan zastany)

    Returns:
        Osierocone klucze, gdy mieszczą się w progu

    Raises:
        SystemExit: Gdy kolejka gubi więcej, niż wolno
    """
    orphans = orphaned_verdicts(queue_path, labels_dir)
    if len(orphans) > allowed:
        sample = "\n".join(f"    {key}" for key in sorted(orphans)[:5])
        raise SystemExit(
            f"PRZERWANE: kolejka gubi {len(orphans)} werdyktow, wolno {allowed}.\n"
            f"  Publikacja zabralaby czyjas prace. Uzyj kuracji z --keep na "
            f"kolejce juz wydanej anotatorom.\n{sample}"
        )
    return orphans
