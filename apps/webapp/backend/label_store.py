"""
Etykiety człowieka jako pliki JSONL — jedyna postać wyników, która jeździ w gicie.

Sesja anotacji leży poza repozytorium i jest lokalna dla maszyny, więc praca
anotatora nie miała jak trafić do kolegów. Tutaj każdy werdykt dopisuje się
linią do pliku `data/labels/<zbior>/<kto>.jsonl`, a taki plik:

* **da się scalić bez konfliktów** — git łączy DOPISANE linie automatycznie,
  podczas gdy jeden wspólny JSON kończyłby konfliktem przy każdym pchnięciu;
* **zachowuje historię, nie tylko stan** — ta sama para oceniona dwa razy zostaje
  dwiema liniami. To nie śmieć: rozbieżność między anotatorami jest materiałem
  do policzenia zgodności (kappa Cohena), a nadpisanie skasowałoby ją bezpowrotnie;
* **jest czytelny wprost** — w razie awarii narzędzia etykiety odczyta `grep`.

Parę identyfikuje ŚCIEŻKA KLATKI SZCZYTOWEJ, a nie `image_id`. Identyfikatory
COCO nadaje kuracja i zmieniają się przy każdym jej powtórzeniu; ścieżka niesie
nagranie, trek i numer klatki, więc przeżywa i ponowną kurację, i przeniesienie
zbioru na inną maszynę.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Katalog etykiet — w repozytorium, w przeciwieństwie do klatek i sesji
LABELS_ROOT_ENV: str = "DOGFACS_LABELS_ROOT"
DEFAULT_LABELS_ROOT: str = "data/labels"

# Kto anotuje. Trafia do nazwy pliku, więc dwie osoby nigdy nie piszą do jednego.
ANNOTATOR_ENV: str = "DOGFACS_ANNOTATOR"
DEFAULT_ANNOTATOR: str = "anonim"

# Znaki niedozwolone w nazwie pliku — podmieniamy, żeby imię z systemu
# („Anton Shkrebela", „jan/kowalski") nie wysadziło zapisu
_SAFE_REPLACEMENT: str = "_"
_UNSAFE_CHARS: str = '<>:"/\\|?* '


@dataclass
class LabelRecord:
    """
    Jedna decyzja człowieka o jednej parze.

    Attributes:
        pair_key: Ścieżka klatki szczytowej — stabilny identyfikator pary
        annotator: Kto oceniał
        timestamp: Kiedy (ISO 8601, UTC)
        au_verdicts: {nazwa AU: active | inactive | not_observable}
        usable: Czy kadr nadaje się do kodowania AU
        keypoints_ok: Czy punkty leżą na mordzie; None = nieoceniono
        breed: Rasa po poprawce człowieka
        emotion: Emocja po poprawce człowieka
        roles_swapped: Czy role klatek są odwrotne (szczytowa jest spoczynkowa)
        keypoints: Punkty poprawione ręcznie (138 wartości) albo None, gdy ta
            decyzja ich nie dotyczyła. Przeciąganie 46 punktów to najdroższa
            praca anotatora — bez zapisania jej TUTAJ ginęła przy każdym
            przebudowaniu sesji, czyli przy każdym otwarciu narzędzia.
    """

    pair_key: str
    annotator: str
    timestamp: str
    au_verdicts: dict[str, str] = field(default_factory=dict)
    usable: bool = True
    keypoints_ok: Optional[bool] = None
    breed: Optional[str] = None
    emotion: Optional[str] = None
    roles_swapped: bool = False
    keypoints: Optional[list[float]] = None

    def to_json_line(self) -> str:
        """Serializuje rekord do jednej linii JSONL."""
        payload = {
            "pair_key": self.pair_key,
            "annotator": self.annotator,
            "timestamp": self.timestamp,
            "au_verdicts": self.au_verdicts,
            "usable": self.usable,
            "keypoints_ok": self.keypoints_ok,
            "breed": self.breed,
            "emotion": self.emotion,
            "roles_swapped": self.roles_swapped,
            "keypoints": self.keypoints,
        }
        return json.dumps(payload, ensure_ascii=False)


def labels_root() -> Path:
    """Zwraca katalog etykiet (odczytywany przy każdym wywołaniu — jak w testach)."""
    return Path(os.environ.get(LABELS_ROOT_ENV, DEFAULT_LABELS_ROOT))


def current_annotator() -> str:
    """
    Zwraca nazwę bieżącego anotatora, bezpieczną jako nazwa pliku.

    Returns:
        Nazwa z `DOGFACS_ANNOTATOR`, a gdy jej nie ma — nazwa konta systemowego
    """
    raw = os.environ.get(ANNOTATOR_ENV) or os.environ.get("USERNAME") or DEFAULT_ANNOTATOR
    safe = "".join(_SAFE_REPLACEMENT if char in _UNSAFE_CHARS else char for char in raw)
    return safe.strip(_SAFE_REPLACEMENT) or DEFAULT_ANNOTATOR


def labels_path(dataset: str, annotator: Optional[str] = None) -> Path:
    """
    Buduje ścieżkę pliku etykiet.

    Args:
        dataset: Nazwa zbioru (katalog, w którym leży kuracja)
        annotator: Kto anotuje; domyślnie `current_annotator()`

    Returns:
        Ścieżka pliku JSONL tego anotatora dla tego zbioru
    """
    return labels_root() / dataset / f"{annotator or current_annotator()}.jsonl"


def append_label(dataset: str, record: LabelRecord) -> Path:
    """
    Dopisuje decyzję do pliku etykiet.

    Args:
        dataset: Nazwa zbioru
        record: Decyzja człowieka

    Returns:
        Ścieżka pliku, do którego dopisano
    """
    path = labels_path(dataset, record.annotator)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(record.to_json_line() + "\n")
    return path


def build_record(
    pair_key: str,
    au_verdicts: dict[str, str],
    usable: bool,
    keypoints_ok: Optional[bool],
    breed: Optional[str],
    emotion: Optional[str],
    roles_swapped: bool = False,
    annotator: Optional[str] = None,
    keypoints: Optional[list[float]] = None,
) -> LabelRecord:
    """
    Składa rekord etykiety z bieżącym anotatorem i znacznikiem czasu.

    Args:
        pair_key: Ścieżka klatki szczytowej
        au_verdicts: Werdykty AU
        usable: Czy kadr się nadaje
        keypoints_ok: Ocena punktów; None = nieoceniono
        breed: Rasa po poprawce
        emotion: Emocja po poprawce
        roles_swapped: Czy role klatek są odwrotne
        annotator: Kto ocenia; None znaczy „weź z konta systemowego". Przy
            jednym komputerze dzielonym przez zespół nazwa konta wpisałaby
            werdykty wszystkich do pliku właściciela maszyny, więc interfejs
            zawsze podaje wybraną osobę jawnie.

    Returns:
        Gotowy `LabelRecord`
    """
    return LabelRecord(
        pair_key=pair_key,
        annotator=annotator or current_annotator(),
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        au_verdicts=dict(au_verdicts),
        usable=usable,
        keypoints_ok=keypoints_ok,
        breed=breed,
        emotion=emotion,
        roles_swapped=roles_swapped,
        keypoints=keypoints,
    )


def read_all(dataset: str) -> list[LabelRecord]:
    """
    Wczytuje etykiety WSZYSTKICH anotatorów danego zbioru.

    Linie uszkodzone (np. przerwany zapis) pomijamy zamiast wywracać wczytywanie:
    utrata jednej decyzji jest do przeżycia, utrata całej pracy zespołu nie.

    Args:
        dataset: Nazwa zbioru

    Returns:
        Rekordy w kolejności czasu; puste, gdy katalogu jeszcze nie ma
    """
    directory = labels_root() / dataset
    if not directory.is_dir():
        return []

    records: list[LabelRecord] = []
    for path in sorted(directory.glob("*.jsonl")):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(LabelRecord(**json.loads(line)))
                except (json.JSONDecodeError, TypeError):
                    continue
    return sorted(records, key=lambda record: record.timestamp)


# Pola, w których None znaczy „ta decyzja ich nie dotyczyła", a nie „wyczyść".
# Nadpisanie ich wartością None gubiłoby wcześniejszą pracę: zapis werdyktów AU
# nie niesie keypoints, więc skasowałby ręczne poprawki punktów.
_MERGEABLE_FIELDS: tuple[str, ...] = ("keypoints", "breed", "emotion", "keypoints_ok")


def latest_by_pair(dataset: str) -> dict[str, LabelRecord]:
    """
    Zwraca obowiązujący stan każdej pary, SCALONY z kolejnych decyzji.

    Scalanie, a nie zwykłe nadpisanie ostatnim rekordem: zapis werdyktów AU nie
    niesie poprawionych keypoints, więc nadpisanie skasowałoby przeciągnięte
    ręcznie punkty — najdroższą pracę anotatora. Pole ustawione na None znaczy
    „ta decyzja go nie dotyczyła", więc zostaje poprzednia wartość.

    Historia zostaje w pliku nietknięta — jest potrzebna do policzenia zgodności
    między osobami.

    Args:
        dataset: Nazwa zbioru

    Returns:
        Mapa `pair_key` → stan obowiązujący
    """
    latest: dict[str, LabelRecord] = {}
    for record in read_all(dataset):
        previous = latest.get(record.pair_key)
        if previous is not None:
            for field_name in _MERGEABLE_FIELDS:
                if getattr(record, field_name) is None:
                    setattr(record, field_name, getattr(previous, field_name))
        latest[record.pair_key] = record
    return latest
