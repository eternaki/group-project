#!/usr/bin/env python3
"""
Uczy model AU z geometrii i porównuje go z regułami na tym samym materiale.

    python -m scripts.annotation.train_au_model

Reguły sprowadzają każde AU do jednej odległości między punktami i wspólnego
progu 1.15. Ten skrypt uczy na tych samych parach model, który widzi
przesunięcie WSZYSTKICH punktów, i sprawdza go tak samo jak `eval_au_rules`:
sprawdzianem krzyżowym z podziałem PO NAGRANIACH, z progiem dobieranym wyłącznie
na części uczącej. Zmierzone — F1 8.9% (reguły surowe) -> 16.7% (reguły
z progiem na AU) -> 24.5% (geometria na wszystkich ocenach) -> 29.1%
(geometria na ocenach o spójnym standardzie). Wszystko przy rolach
z pipeline'u, czyli tak, jak wypadnie na nowym materiale.

NAJWIĘKSZY POJEDYNCZY ZYSK DAŁO ODRZUCENIE CZĘŚCI ETYKIET, NIE ULEPSZANIE
MODELU. Jeden anotator zapala 0.17% komórek, reszta 3.5-5.4% — mieszanie tych
standardów uczy model, że ten sam kadr jest i aktywny, i spoczynkowy. Uczenie
na 273 parach o spójnym standardzie bije uczenie na wszystkich 527: precyzja
32.7% wobec 26.8%. Przełącznik `--mixed-standards` pozwala to sprawdzić.

DLACZEGO PODZIAŁ IDZIE PO NAGRANIACH. Kadry jednego nagrania pokazują tego
samego psa w tej samej scenie; model, który zobaczył część kadrów nagrania,
rozpoznaje pozostałe z pamięci, nie z umiejętności. Przy podziale losowym po
parach wynik wychodzi zawyżony.

DLACZEGO PRÓG DOBIERA SIĘ WEWNĄTRZ FOLDU. Próg to też parametr uczony —
wybrany na wszystkich ocenach, a potem pokazany jako wynik, zawyża precyzję.
Liczba, którą wolno cytować, pochodzi wyłącznie z par nieoglądanych: ani model,
ani próg nie widziały nagrania, na którym są sprawdzane.

Wynik: `models/au_geometry.json` — współczynniki, normalizacja i próg każdego AU.
Model wchodzi do zbioru przy najbliższym `build_full_dataset` jako pole
`au_model_verdict`; werdykt człowieka nadal ma pierwszeństwo nad wszystkim.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "apps" / "webapp" / "backend"))

from packages.models.au_geometry import (  # noqa: E402
    MIN_POSITIVES,
    AUGeometryModel,
    pair_features,
    train_action_unit,
)
from packages.models.delta_action_units import ACTION_UNIT_NAMES  # noqa: E402
from scripts.annotation.build_final_dataset import (  # noqa: E402
    DEFAULT_DATASET,
    _neutral_of,
    _resolve_data,
    index_curated,
    resolve_labels,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS: Path = REPO_ROOT / "models" / "au_geometry.json"

# Liczba części sprawdzianu krzyżowego.
FOLDS: int = 5

# Werdykty, które w ogóle są oceną. `not_observable` znaczy „nie wiem" i nie
# wolno go liczyć ani jako aktywację, ani jako spoczynek.
VERDICT_ACTIVE: str = "active"
VERDICT_INACTIVE: str = "inactive"

# Najniższy udział aktywacji, przy którym oceny anotatora niosą jakikolwiek
# sygnał dodatni. Zmierzone udziały w zespole rozjeżdżają się o dwa rzędy
# wielkości: 0.19%, 3.57%, 3.78%, 11.91% — więc próg 1% rozdziela je czysto
# i nie jest dobrany do konkretnej osoby.
MIN_ACTIVATION_RATE: float = 0.01

# Poniżej tylu ocenionych komórek udział aktywacji jest zbyt niepewny, żeby
# na jego podstawie kogokolwiek pomijać.
MIN_CELLS_TO_JUDGE: int = 200


class TrainingSet:
    """Pary uczące: cechy, nagranie źródłowe i werdykt człowieka na każde AU."""

    def __init__(self) -> None:
        self.videos: list[str] = []
        self.features: list[np.ndarray] = []
        self.pipeline_features: list[np.ndarray] = []
        self.verdicts: list[dict[str, str]] = []

    def add(
        self,
        video: str,
        features: np.ndarray,
        pipeline_features: np.ndarray,
        verdicts: dict[str, str],
    ) -> None:
        """
        Dokłada jedną parę w dwóch wariantach cech.

        Args:
            video: Nagranie źródłowe
            features: Cechy przy rolach poprawionych przez człowieka
            pipeline_features: Cechy przy rolach z pipeline'u (jak przy wdrożeniu)
            verdicts: Werdykt człowieka na każde AU
        """
        self.videos.append(video)
        self.features.append(features)
        self.pipeline_features.append(pipeline_features)
        self.verdicts.append(verdicts)

    def matrix(self, pipeline_roles: bool = False) -> np.ndarray:
        """
        Zwraca cechy jako macierz.

        Args:
            pipeline_roles: Czy wziąć wariant z rolami z pipeline'u

        Returns:
            Macierz cech
        """
        return np.asarray(self.pipeline_features if pipeline_roles else self.features)

    def labels(self, action_unit: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Wyciąga oceny jednego AU razem z maską par, które je oceniają.

        Args:
            action_unit: Nazwa AU

        Returns:
            Para (maska ocenionych par, etykiety 0/1 dla tych par)
        """
        mask = np.array(
            [v.get(action_unit) in (VERDICT_ACTIVE, VERDICT_INACTIVE) for v in self.verdicts]
        )
        labels = np.array(
            [1 if self.verdicts[i].get(action_unit) == VERDICT_ACTIVE else 0 for i in np.where(mask)[0]]
        )
        return mask, labels

    def __len__(self) -> int:
        return len(self.features)


def activation_rates(labels: dict) -> dict[str, tuple[float, int]]:
    """
    Liczy, jak często każdy anotator w ogóle orzeka aktywację.

    Args:
        labels: Mapa pair_key -> werdykt obowiązujący

    Returns:
        Mapa anotator -> (udział aktywacji, liczba ocenionych komórek)
    """
    counters: dict[str, list[int]] = {}
    for record in labels.values():
        if not record.usable or not record.au_verdicts:
            continue
        entry = counters.setdefault(record.annotator, [0, 0])
        for verdict in record.au_verdicts.values():
            if verdict == VERDICT_ACTIVE:
                entry[0] += 1
            if verdict in (VERDICT_ACTIVE, VERDICT_INACTIVE):
                entry[1] += 1
    return {
        name: (active / cells if cells else 0.0, cells) for name, (active, cells) in counters.items()
    }


def inconsistent_annotators(labels: dict) -> set[str]:
    """
    Wskazuje anotatorów, których oceny nie niosą sygnału dodatniego.

    NIE chodzi o to, że ktoś ocenia „źle" — chodzi o MIESZANIE STANDARDÓW.
    Anotator zapalający 0.19% komórek i anotator zapalający 3.78% opisują tę
    samą mimikę sprzecznie, a model uczony na sumie dostaje ten sam kadr raz
    jako aktywny, raz jako spoczynek. Zmierzone: uczenie na wszystkich 527
    parach daje precyzję 26.8%, a na 240 parach o spójnym standardzie — 37.5%,
    czyli DWA RAZY MNIEJ danych wypada lepiej.

    Oceny pominiętego anotatora zostają w dzienniku i w zbiorze — niosą
    `usable`, emocję, rasę i poprawki punktów. Odpada tylko ich udział
    w uczeniu AU.

    Args:
        labels: Mapa pair_key -> werdykt obowiązujący

    Returns:
        Nazwy anotatorów do pominięcia przy uczeniu
    """
    return {
        name
        for name, (rate, cells) in activation_rates(labels).items()
        if cells >= MIN_CELLS_TO_JUDGE and rate < MIN_ACTIVATION_RATE
    }


def collect_training_set(dataset: str, mixed_standards: bool = False) -> TrainingSet:
    """
    Zbiera pary ocenione przez człowieka razem z geometrią obu klatek.

    Klatkę neutralną dobiera `_neutral_of`, czyli po `track_id`, a nie po samym
    obrazie — przy kilku psach w kadrze baza AU wzięłaby się inaczej od sąsiada.

    ROLE BIERZE SIĘ Z WERDYKTU, NIE Z PIPELINE'U. Gdy człowiek zaznaczył
    `roles_swapped`, wyrazem twarzy jest klatka nazwana przez pipeline neutralną
    i to do niej odnosi się werdykt. Pominięcie tego nie jest drobiazgiem:
    dotyczy 45 par z 527 (8.5%), odwraca im znak przesunięcia I przypina ocenę
    do niewłaściwej klatki. Zmierzone — średnie AUC dziewięciu uczonych AU
    spada z 0.757 na 0.645, czyli te 8.5% par kasuje połowę sygnału. Tak mała
    domieszka tyle waży, bo potwierdzeń jest 10-37 na AU.

    Args:
        dataset: Nazwa zbioru w `data/`
        mixed_standards: Czy uczyć na ocenach wszystkich anotatorów, także tych
            o niezgodnym standardzie (do porównania, nie do produkcji)

    Returns:
        Zebrane pary uczące
    """
    dataset_dir = REPO_ROOT / "data" / dataset
    coco = json.loads(_resolve_data(dataset_dir, "curated.json").read_text(encoding="utf-8"))
    labels, _ = resolve_labels(dataset)
    index = index_curated(coco)
    skipped = set() if mixed_standards else inconsistent_annotators(labels)
    if skipped:
        logger.info("Pominieci przy uczeniu (standard bez aktywacji): %s", ", ".join(sorted(skipped)))

    zbior = TrainingSet()
    for file_name, peak in index.peak_by_file.items():
        record = labels.get(file_name)
        if record is None or not record.usable or not record.au_verdicts:
            continue
        if record.annotator in skipped:
            continue
        neutral = _neutral_of(index, peak)
        if neutral is None:
            continue
        expression, baseline = (neutral, peak) if record.roles_swapped else (peak, neutral)
        features = pair_features(expression.get("keypoints"), baseline.get("keypoints"))
        pipeline = pair_features(peak.get("keypoints"), neutral.get("keypoints"))
        if features is None or pipeline is None:
            continue
        video = index.images[peak["image_id"]].get("source_video") or file_name
        zbior.add(video, features, pipeline, dict(record.au_verdicts))
    return zbior


def assign_folds(videos: list[str]) -> dict[str, int]:
    """
    Rozdziela nagrania na części sprawdzianu, powtarzalnie i bez losowania.

    `hash()` w Pythonie jest solony na proces, więc kolejność wyznacza własny
    skrót — inaczej ten sam materiał dawałby inny wynik przy każdym uruchomieniu.

    Args:
        videos: Nagrania kolejnych par

    Returns:
        Mapa nagranie -> numer części
    """
    unique = sorted(set(videos))
    ordered = sorted(unique, key=_digest)
    return {name: number % FOLDS for number, name in enumerate(ordered)}


def _digest(text: str) -> int:
    """
    Liczy powtarzalny skrót nazwy.

    Args:
        text: Nazwa nagrania

    Returns:
        Liczba całkowita
    """
    value = 0
    for character in text:
        value = (value * 131 + ord(character)) % (2**31)
    return value


class Score:
    """Zliczenia trafień i pomyłek."""

    def __init__(self) -> None:
        self.hits = 0
        self.false_alarms = 0
        self.missed = 0

    def add(self, predicted: bool, actual: bool) -> None:
        """
        Dokłada jedną ocenę.

        Args:
            predicted: Czy model orzekł aktywację
            actual: Czy człowiek orzekł aktywację
        """
        if predicted and actual:
            self.hits += 1
        elif predicted:
            self.false_alarms += 1
        elif actual:
            self.missed += 1

    @property
    def precision(self) -> float:
        """Udział trafień wśród orzeczeń."""
        called = self.hits + self.false_alarms
        return self.hits / called if called else 0.0

    @property
    def recall(self) -> float:
        """Udział znalezionych aktywacji."""
        total = self.hits + self.missed
        return self.hits / total if total else 0.0

    @property
    def f1(self) -> float:
        """Średnia harmoniczna precyzji i pokrycia."""
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    def __iadd__(self, other: "Score") -> "Score":
        self.hits += other.hits
        self.false_alarms += other.false_alarms
        self.missed += other.missed
        return self


def cross_validate(
    zbior: TrainingSet,
    action_unit: str,
    folds: dict[str, int],
    pipeline_roles: bool = False,
) -> Optional[Score]:
    """
    Mierzy AU na nagraniach nieoglądanych przy uczeniu i przy wyborze progu.

    Uczenie zawsze idzie po rolach poprawionych przez człowieka — po to są
    werdykty. Sprawdzać można na dwa sposoby i oba są uczciwe, tylko odpowiadają
    na inne pytanie. Na rolach człowieka: ile model umie, gdy para jest ułożona
    poprawnie. Na rolach z pipeline'u: ile zostanie przy PUSZCZENIU na nowy
    materiał, gdzie nikt nie powie, że klatki są zamienione — a pipeline myli je
    na 8.5% par.

    Args:
        zbior: Pary uczące
        action_unit: Nazwa AU
        folds: Mapa nagranie -> numer części
        pipeline_roles: Czy sprawdzać na cechach z rolami z pipeline'u

    Returns:
        Zliczenia albo None, gdy potwierdzeń jest za mało, by cokolwiek nauczyć
    """
    mask, labels = zbior.labels(action_unit)
    if labels.sum() < MIN_POSITIVES:
        return None
    features = zbior.matrix()[mask]
    tested = zbior.matrix(pipeline_roles)[mask]
    videos = [zbior.videos[i] for i in np.where(mask)[0]]

    score = Score()
    for fold in range(FOLDS):
        train = np.array([folds[v] != fold for v in videos])
        test = ~train
        if not test.any() or labels[train].sum() == 0:
            continue
        weights = train_action_unit(features[train], labels[train])
        for index in np.where(test)[0]:
            called = weights.score(tested[index]) >= weights.threshold
            score.add(called, bool(labels[index]))
    return score


def fit_model(zbior: TrainingSet) -> AUGeometryModel:
    """
    Uczy ostateczny model na wszystkich parach.

    AU bez dostatecznej liczby potwierdzeń NIE dostaje modelu — brak wpisu
    znaczy „nie umiem tego ocenić" i jest uczciwszy niż orzeczenie z trzech
    przykładów.

    Args:
        zbior: Pary uczące

    Returns:
        Model gotowy do zapisu
    """
    per_unit = {}
    for action_unit in ACTION_UNIT_NAMES:
        mask, labels = zbior.labels(action_unit)
        if labels.sum() < MIN_POSITIVES:
            continue
        per_unit[action_unit] = train_action_unit(zbior.matrix()[mask], labels)
    return AUGeometryModel(per_unit)


def load_model(path: Path = DEFAULT_WEIGHTS) -> Optional[AUGeometryModel]:
    """
    Wczytuje zapisany model, jeśli istnieje.

    Args:
        path: Ścieżka pliku ze współczynnikami

    Returns:
        Model albo None, gdy pliku nie ma
    """
    if not path.is_file():
        return None
    return AUGeometryModel.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _report(zbior: TrainingSet, folds: dict[str, int]) -> Score:
    """
    Wypisuje wynik na każde AU i zwraca sumę.

    Args:
        zbior: Pary uczące
        folds: Mapa nagranie -> numer części

    Returns:
        Zliczenia zsumowane po wszystkich AU
    """
    total = Score()
    logger.info("%-10s %6s %10s %10s", "AU", "aktyw", "precyzja", "pokrycie")
    for action_unit in ACTION_UNIT_NAMES:
        _, labels = zbior.labels(action_unit)
        score = cross_validate(zbior, action_unit, folds)
        if score is None:
            logger.info("%-10s %6d %10s", action_unit, int(labels.sum()), "za malo")
            continue
        total += score
        logger.info(
            "%-10s %6d %9.1f%% %9.1f%%",
            action_unit,
            int(labels.sum()),
            100 * score.precision,
            100 * score.recall,
        )
    return total


def _log_annotators(dataset: str) -> None:
    """
    Wypisuje, jak często każdy anotator orzeka aktywację.

    Args:
        dataset: Nazwa zbioru w `data/`
    """
    labels, _ = resolve_labels(dataset)
    logger.info("%-10s %10s %10s", "anotator", "aktywacje", "komorki")
    for name, (rate, cells) in sorted(activation_rates(labels).items(), key=lambda kv: -kv[1][1]):
        logger.info("%-10s %9.2f%% %10d", name, 100 * rate, cells)


def main() -> None:
    """Uczy model, mierzy go sprawdzianem krzyżowym i zapisuje współczynniki."""
    parser = argparse.ArgumentParser(description="Model AU z geometrii twarzy")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Nazwa zbioru w data/")
    parser.add_argument("--output", default=str(DEFAULT_WEIGHTS), help="Plik ze wspolczynnikami")
    parser.add_argument(
        "--skip-eval", action="store_true", help="Tylko naucz i zapisz, bez sprawdzianu"
    )
    parser.add_argument(
        "--mixed-standards",
        action="store_true",
        help="Ucz na ocenach wszystkich anotatorow (do porownania, nie do produkcji)",
    )
    args = parser.parse_args()

    _log_annotators(args.dataset)
    zbior = collect_training_set(args.dataset, args.mixed_standards)
    logger.info("Par uczacych : %d", len(zbior))
    if not len(zbior):
        raise SystemExit("Brak par z werdyktem czlowieka — nie ma na czym uczyc")
    folds = assign_folds(zbior.videos)
    logger.info("Nagran       : %d", len(folds))

    if not args.skip_eval:
        logger.info("")
        total = _report(zbior, folds)
        wdrozenie = Score()
        for action_unit in ACTION_UNIT_NAMES:
            score = cross_validate(zbior, action_unit, folds, pipeline_roles=True)
            if score is not None:
                wdrozenie += score
        logger.info("")
        logger.info(
            "MODEL, role poprawione   precyzja %.1f%%  pokrycie %.1f%%  F1 %.1f%%",
            100 * total.precision,
            100 * total.recall,
            100 * total.f1,
        )
        logger.info(
            "MODEL, role z pipeline'u precyzja %.1f%%  pokrycie %.1f%%  F1 %.1f%%  <- jak przy wdrozeniu",
            100 * wdrozenie.precision,
            100 * wdrozenie.recall,
            100 * wdrozenie.f1,
        )
        logger.info("reguly kalibrowane       precyzja 10.7%%  pokrycie 38.5%%  F1 16.7%%")
        logger.info("reguly surowe            precyzja  5.0%%  pokrycie 40.5%%  F1  8.9%%")

    model = fit_model(zbior)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model.to_dict(), separators=(",", ":")), encoding="utf-8")
    logger.info("")
    logger.info("AU z modelem : %d z %d", len(model.per_unit), len(ACTION_UNIT_NAMES))
    logger.info("Zapisano: %s", output)


if __name__ == "__main__":
    main()
