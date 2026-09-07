#!/usr/bin/env python3
"""
Uklada kandydatow tak, zeby podejrzane kadry trafily na poczatek listy.

Model NIE PODEJMUJE decyzji — zmienia wylacznie KOLEJNOSC pokazywania. Powod
jest zmierzony: na odlozonej polowie decyzji czlowieka AUC wynosi 0.80, ale
precyzja nawet na 25 najpewniejszych typach to 64%, czyli co trzeci typ jest
chybiony. Automat odrzucajacy kadry wyrzucilby setki dobrych par. Automat
sortujacy kosztuje najwyzej jedno zbedne klikniecie.

Czego model sie uczy. Z decyzji czlowieka wynika, ze o odrzuceniu NIE decyduje
szerokosc mordy (odsetek odrzucen w pasmach 80-120, 120-200 i 200+ px wynosi
22%, 22% i 17% — plasko), tylko jakosc punktow: przy udziale niepewnych
punktow ponizej 0.10 odrzucane jest 4% kadrow, powyzej 0.40 juz 39%.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from packages.pipeline.quality_gate import QualityThresholds, assess_frame
from scripts.annotation.curate_for_review import FRAME_ROLE_PEAK, keypoints_outside_bbox

DEFAULT_DIR: str = "data/kandydaci"
DEFAULT_DATASET: str = "data/dataset_final/annotations.json"

# Ile krokow spadku gradientu i z jakim krokiem — regresja logistyczna na
# czterech cechach zbiega spokojnie, nie ma po co dokladac biblioteki
STEPS: int = 4000
LEARNING_RATE: float = 0.05


def features(annotation: dict) -> list[float]:
    """
    Wyciaga cechy kadru, po ktorych czlowiek rozpoznaje zly kadr.

    Args:
        annotation: Anotacja klatki szczytowej z surowego COCO

    Returns:
        Lista cech w stalej kolejnosci
    """
    loose = QualityThresholds(min_face_width=0.0, max_asymmetry=99.0, max_weak_ratio=1.0)
    quality = assess_frame(annotation.get("keypoints"), loose)
    points = np.asarray(annotation["keypoints"], dtype=float).reshape(-1, 3)
    return [
        float(getattr(quality, "weak_ratio", 0.0)),
        float(getattr(quality, "asymmetry", 0.0)),
        float(points[:, 2].mean()),
        float(quality.face_width) / 100.0,
        float(keypoints_outside_bbox(annotation)),
    ]


def fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Uczy regresje logistyczna z wyrownaniem klas.

    Wyrownanie jest konieczne: odrzucen jest 20%, wiec model bez wag nauczylby
    sie mowic "zostaw" na wszystko i mial 80% trafnosci nie robiac nic.

    Args:
        x: Cechy ze staloa kolumna na koncu
        y: Etykiety, 1 znaczy ODRZUCIC

    Returns:
        Wagi modelu
    """
    weights = np.zeros(x.shape[1])
    positive = max(int((y == 1).sum()), 1)
    sample_weight = np.where(y == 1, (y == 0).sum() / positive, 1.0)
    for _ in range(STEPS):
        prediction = 1 / (1 + np.exp(-x @ weights))
        weights -= LEARNING_RATE * (x.T @ (sample_weight * (prediction - y))) / len(y)
    return weights


def _normalise(raw: np.ndarray, center: np.ndarray, spread: np.ndarray) -> np.ndarray:
    """
    Skaluje cechy i dokleja kolumne staloa.

    Args:
        raw: Surowe cechy
        center: Srednie z danych uczacych
        spread: Odchylenia z danych uczacych

    Returns:
        Macierz gotowa dla modelu
    """
    scaled = (raw - center) / (spread + 1e-9)
    return np.hstack([scaled, np.ones((len(scaled), 1))])


def order_by_risk(
    candidates: list[dict],
    decisions: dict[str, bool],
    annotations: dict[str, dict],
) -> list[dict]:
    """
    Uklada kandydatow: nieprzejrzane wedlug ryzyka, przejrzane na koniec.

    Args:
        candidates: Lista kandydatow z `candidates.json`
        decisions: Decyzje czlowieka {sciezka peaku: czy odrzucony}
        annotations: Anotacje klatek szczytowych po sciezce

    Returns:
        Nowa lista w kolejnosci do pokazania
    """
    trained = [k for k in candidates if k["peak"] in decisions and k["peak"] in annotations]
    if len(trained) < 30:
        return candidates

    x_raw = np.array([features(annotations[k["peak"]]) for k in trained], dtype=float)
    y = np.array([1 if decisions[k["peak"]] else 0 for k in trained])
    center, spread = x_raw.mean(0), x_raw.std(0)
    weights = fit(_normalise(x_raw, center, spread), y)

    fresh = [k for k in candidates if k["peak"] not in decisions]
    seen = [k for k in candidates if k["peak"] in decisions]
    if fresh:
        x_fresh = np.array(
            [
                features(annotations[k["peak"]])
                if k["peak"] in annotations
                else [0.0] * x_raw.shape[1]
                for k in fresh
            ],
            dtype=float,
        )
        risk = 1 / (1 + np.exp(-_normalise(x_fresh, center, spread) @ weights))
        for candidate, value in zip(fresh, risk):
            candidate["risk"] = round(float(value), 3)
        fresh.sort(key=lambda k: -k["risk"])
    return fresh + seen


def load_annotations(dataset: Path) -> dict[str, dict]:
    """
    Indeksuje anotacje klatek szczytowych po sciezce pliku.

    Args:
        dataset: Surowy COCO

    Returns:
        Anotacje pod sciezkami klatek
    """
    coco = json.loads(dataset.read_text(encoding="utf-8"))
    names = {image["id"]: image["file_name"] for image in coco["images"]}
    found: dict[str, dict] = {}
    for annotation in coco["annotations"]:
        if annotation.get("frame_role") != FRAME_ROLE_PEAK:
            continue
        found.setdefault(names[annotation["image_id"]], annotation)
    return found


def read_decisions(directory: Path) -> dict[str, bool]:
    """
    Odczytuje decyzje czlowieka z zapisanego wyboru.

    Args:
        directory: Katalog kandydatow

    Returns:
        {sciezka peaku: czy ODRZUCONY}; pusty slownik, gdy nic nie zapisano
    """
    saved: Optional[Path] = None
    for name in ("wybrane.json", "postep.json"):
        path = directory / name
        if path.is_file() and path.stat().st_size > 2:
            saved = path
            break
    if saved is None:
        return {}
    keep = {entry["peak"] for entry in json.loads(saved.read_text(encoding="utf-8"))}
    everything = json.loads((directory / "candidates.json").read_text(encoding="utf-8"))
    reviewed_to = max(
        (i for i, k in enumerate(everything) if k["peak"] not in keep), default=-1
    )
    return {k["peak"]: k["peak"] not in keep for k in everything[: reviewed_to + 1]}


def main() -> None:
    """Punkt wejscia CLI."""
    parser = argparse.ArgumentParser(description="Sortuje kandydatow wedlug ryzyka")
    parser.add_argument("--dir", default=DEFAULT_DIR)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    args = parser.parse_args()

    directory = Path(args.dir)
    candidates = json.loads((directory / "candidates.json").read_text(encoding="utf-8"))
    decisions = read_decisions(directory)
    print(f"decyzji czlowieka: {len(decisions)} (odrzuconych {sum(decisions.values())})")

    ordered = order_by_risk(candidates, decisions, load_annotations(Path(args.dataset)))
    (directory / "candidates.json").write_text(
        json.dumps(ordered, ensure_ascii=False), encoding="utf-8"
    )
    fresh = [k for k in ordered if "risk" in k]
    print(f"ulozono {len(ordered)}: nieprzejrzanych {len(fresh)}, przejrzanych {len(ordered)-len(fresh)}")
    if fresh:
        print(f"ryzyko pierwszego {fresh[0]['risk']:.2f}, ostatniego {fresh[-1]['risk']:.2f}")


if __name__ == "__main__":
    main()
