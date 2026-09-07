"""
Model AU uczony z pełnej geometrii twarzy, zamiast ręcznie dobranych proporcji.

DLACZEGO NIE REGUŁY. `DeltaActionUnitsExtractor` sprowadza każde AU do JEDNEJ
odległości między punktami, podzielonej przez rozstaw oczu i porównanej ze
wspólnym progiem 1.15. Zmierzone na 527 parach ocenionych przez człowieka:
precyzja 5.0% przy pokryciu 40.5%. Osobny próg na każde AU podnosi ją do 10.7%,
ale sufit jest w cesze, nie w progu — pojedyncza odległość gubi informację,
którą punkty niosą. Ten sam materiał i ta sama walidacja (podział po
NAGRANIACH, próg dobierany wyłącznie na części uczącej), ale cechą jest
PRZESUNIĘCIE WSZYSTKICH 46 punktów względem klatki bazowej:

    reguły surowe               precyzja  5.0%   pokrycie 40.5%   F1  8.9%
    reguły z progiem na AU      precyzja 10.7%   pokrycie 38.5%   F1 16.7%
    model na wszystkich ocenach precyzja 26.8%   pokrycie 22.5%   F1 24.5%
    model, spójny standard      precyzja 39.5%   pokrycie 36.1%   F1 37.7%
    ten sam, role z pipeline'u  precyzja 32.7%   pokrycie 26.3%   F1 29.1%

Dwa ostatnie wiersze wymagają komentarza. „Spójny standard" znaczy: uczone
wyłącznie na ocenach anotatorów, którzy w ogóle orzekają aktywacje — jeden
członek zespołu zapala 0.17% komórek wobec 3.78% u reszty, więc jego oceny
uczą modelu, że mimika nie istnieje. Pominięcie ich zostawia 273 pary zamiast
527 i mimo to podnosi precyzję z 26.8% na 39.5%: DWA RAZY MNIEJ danych wypada
o połowę lepiej. Ostatni wiersz to ta sama liczba przy rolach z pipeline'u,
czyli tak, jak wypadnie na nowym materiale, gdzie nikt nie powie modelowi, że
klatki są zamienione — i to jego należy cytować mówiąc o zbiorze 9k.

Model odtwarza więc JEDEN standard oceniania, nie „prawdę o psach". Przy
precyzji 33% dwie aktywacje na trzy są zmyślone.

CO TO ZNACZY DLA PROJEKTU: wąskim gardłem nie są keypoints, tylko reguły.
Punkty niosą sygnał, którego geometryczne wzory nie wyciągają.

CZEGO TEN MODEL NIE ZAŁATWIA. Precyzja 33% to nadal nie jest etykieta do
oddania jako prawda — to lepsza PRE-etykieta. Sufit jest niżej, niż się wydaje:
na 39 parach ocenionych niezależnie przez dwie osoby zgoda na aktywacjach AU
wynosi 7.4% (kappa 0.132), więc model dobija do powtarzalności samego zjawiska.
Przed kolejną rundą ulepszania modelu opłaca się naprawić etykietę.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# Udział punktów, które muszą być widoczne, żeby klatka nadawała się do pomiaru.
# Poniżej tego progu brakujące punkty wypełniłyby cechę zerami i model uczyłby
# się rozpoznawać braki detektora zamiast mimiki.
MIN_VISIBLE_SHARE: float = 0.7

# Przelicznik mediany odchyleń bezwzględnych na skalę porównywalną z rozpiętością
# twarzy. Wartość dobrana tak, żeby znormalizowane punkty mieściły się mniej
# więcej w [-1, 1] — sama wartość nie zmienia wyniku modelu, bo cechy i tak
# przechodzą standaryzację, ale trzyma je w zakresie czytelnym przy podglądzie.
ROBUST_SCALE_FACTOR: float = 4.0

# Siła regularyzacji L2. Przy 10-35 przykładach pozytywnych na 184 cechy bez
# regularyzacji model zapamiętuje pojedyncze kadry.
DEFAULT_L2: float = 1.0

# Kroki i tempo spadku gradientu. Regresja logistyczna na tak małym zbiorze
# zbiega w kilkuset krokach; pełny solver nie jest wart zależności od sklearn.
STEPS: int = 400
LEARNING_RATE: float = 0.5

# Zabezpieczenie przed przepełnieniem `exp` przy dużych wartościach liniowych.
LOGIT_CLIP: float = 30.0

# Minimalna liczba potwierdzeń człowieka, przy której da się cokolwiek nauczyć.
# Poniżej model dostaje etykietę „nie wiadomo" zamiast zmyślonej odpowiedzi.
MIN_POSITIVES: int = 10


@dataclass(frozen=True)
class AUWeights:
    """Wyuczone współczynniki jednego AU wraz z normalizacją i progiem."""

    weights: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    threshold: float

    def score(self, features: np.ndarray) -> float:
        """
        Liczy surową ocenę aktywacji dla jednej pary.

        Args:
            features: Wektor cech pary

        Returns:
            Wartość liniowa — im wyższa, tym pewniejsza aktywacja
        """
        standardized = (features - self.mean) / self.std
        return float(np.append(standardized, 1.0) @ self.weights)


def normalized_points(keypoints: Sequence[float]) -> Optional[np.ndarray]:
    """
    Sprowadza punkty do własnego układu twarzy: środek w zerze, skala jednostkowa.

    Skalę i środek liczymy Z SAMYCH PUNKTÓW, a nie z boksu, i to jest celowe.
    Boks znaczy w tym projekcie dwie różne rzeczy: w kuracji `bbox` to CAŁY PIES,
    a w zbiorze do oddania — kadr mordy po przycięciu. Model uczony na jednym,
    a puszczony na drugim, dostałby cechy w innej skali i po cichu zgłupiał.
    Miarą jest mediana odchyleń od mediany punktów: odporna na pojedyncze
    punkty uciekające przy otwarciu pyska, w przeciwieństwie do rozpiętości.

    Args:
        keypoints: Płaska lista (x, y, widoczność) × N

    Returns:
        Tablica (N, 2) albo None, gdy zbyt wiele punktów jest niewidocznych
    """
    if not keypoints:
        return None
    points = np.asarray(keypoints, dtype=float).reshape(-1, 3)
    if np.mean(points[:, 2] > 0) < MIN_VISIBLE_SHARE:
        return None
    coordinates = points[:, :2]
    center = np.median(coordinates, axis=0)
    scale = float(np.median(np.abs(coordinates - center)) * ROBUST_SCALE_FACTOR)
    if scale <= 0:
        return None
    return (coordinates - center) / scale


def pair_features(
    expression_keypoints: Sequence[float], baseline_keypoints: Sequence[float]
) -> Optional[np.ndarray]:
    """
    Buduje wektor cech pary: przesunięcie każdego punktu plus sama poza wyrazu.

    Przesunięcie niesie mimikę (AU są z definicji różnicą względem klatki
    bazowej), a poza wyrazu pozwala modelowi odróżnić ruch mimiczny od obrotu
    głowy, który sam z siebie zmienia wszystkie odległości.

    Kolejność argumentów jest znacząca: pierwszy to klatka Z WYRAZEM, której
    dotyczy werdykt. Gdy człowiek zaznaczył `roles_swapped`, jest nią klatka
    nazwana przez pipeline neutralną — podanie ich odwrotnie odwraca znak
    całego wektora.

    Args:
        expression_keypoints: Punkty klatki z wyrazem twarzy
        baseline_keypoints: Punkty klatki bazowej (spoczynek)

    Returns:
        Wektor cech albo None, gdy którejś klatki nie da się znormalizować
    """
    expression = normalized_points(expression_keypoints)
    baseline = normalized_points(baseline_keypoints)
    if expression is None or baseline is None or expression.shape != baseline.shape:
        return None
    return np.concatenate([(expression - baseline).ravel(), expression.ravel()])


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """
    Liczy sigmoidę odporną na przepełnienie.

    Args:
        values: Wartości liniowe

    Returns:
        Prawdopodobieństwa
    """
    return 1.0 / (1.0 + np.exp(-np.clip(values, -LOGIT_CLIP, LOGIT_CLIP)))


def fit_logistic(features: np.ndarray, labels: np.ndarray, l2: float = DEFAULT_L2) -> np.ndarray:
    """
    Uczy regresję logistyczną z wyrównaniem klas.

    Wyrównanie jest konieczne, nie kosmetyczne: aktywacje stanowią ~2% komórek,
    więc model bez wag odpowiadałby zawsze „spoczynek" i miałby 98% trafności
    przy zerowym pożytku.

    Args:
        features: Macierz cech (przykłady × cechy), już zestandaryzowana
        labels: Etykiety 0/1
        l2: Siła regularyzacji

    Returns:
        Wektor współczynników z wyrazem wolnym na końcu
    """
    design = np.hstack([features, np.ones((len(features), 1))])
    coefficients = np.zeros(design.shape[1])
    positives = max(1, int(labels.sum()))
    negatives = max(1, int((1 - labels).sum()))
    weights = np.where(labels == 1, len(labels) / (2 * positives), len(labels) / (2 * negatives))
    for _ in range(STEPS):
        error = _sigmoid(design @ coefficients) - labels
        penalty = l2 * np.append(coefficients[:-1], 0.0) / len(labels)
        coefficients -= LEARNING_RATE * (design.T @ (weights * error) / len(labels) + penalty)
    return coefficients


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Wybiera próg maksymalizujący F1 na podanych ocenach.

    Args:
        scores: Oceny modelu
        labels: Etykiety 0/1

    Returns:
        Próg; przy braku aktywacji zwraca nieskończoność, czyli „nigdy nie orzekaj"
    """
    if labels.sum() == 0:
        return float("inf")
    chosen, best = float("inf"), -1.0
    for candidate in np.unique(scores):
        predicted = scores >= candidate
        hits = float((predicted & (labels == 1)).sum())
        f1 = 2 * hits / max(1.0, float(predicted.sum()) + float(labels.sum()))
        if f1 > best:
            chosen, best = float(candidate), f1
    return chosen


def train_action_unit(features: np.ndarray, labels: np.ndarray, l2: float = DEFAULT_L2) -> AUWeights:
    """
    Uczy model jednego AU razem z normalizacją i progiem decyzyjnym.

    Args:
        features: Macierz cech (przykłady × cechy)
        labels: Etykiety 0/1
        l2: Siła regularyzacji

    Returns:
        Komplet współczynników gotowy do orzekania
    """
    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-6
    standardized = (features - mean) / std
    coefficients = fit_logistic(standardized, labels, l2)
    scores = np.hstack([standardized, np.ones((len(standardized), 1))]) @ coefficients
    return AUWeights(
        weights=coefficients, mean=mean, std=std, threshold=best_threshold(scores, labels)
    )


# Etykieta trójstanowa — te same nazwy, co w werdykcie człowieka, żeby dało się
# je porównać wprost i żeby „nie wiadomo" nie udawało spoczynku.
VERDICT_ACTIVE: str = "active"
VERDICT_INACTIVE: str = "inactive"
VERDICT_NOT_OBSERVABLE: str = "not_observable"


class AUGeometryModel:
    """Komplet modeli AU: po jednym na każdą jednostkę, którą dało się nauczyć."""

    def __init__(self, per_unit: dict[str, AUWeights]) -> None:
        """
        Args:
            per_unit: Współczynniki w rozbiciu na nazwę AU
        """
        self.per_unit = per_unit

    def predict(self, features: Optional[np.ndarray]) -> dict[str, str]:
        """
        Orzeka trójstanowo o wszystkich AU, które model umie ocenić.

        Brak cech (nie dało się znormalizować którejś klatki) znaczy „nie
        wiadomo" dla WSZYSTKICH AU — nie spoczynek. AU bez wyuczonego modelu
        w wyniku nie występuje wcale, żeby nie udawać wiedzy, której nie ma.

        Args:
            features: Wektor cech pary albo None

        Returns:
            Słownik AU -> werdykt
        """
        if features is None:
            return {au: VERDICT_NOT_OBSERVABLE for au in self.per_unit}
        return {
            au: VERDICT_ACTIVE if weights.score(features) >= weights.threshold else VERDICT_INACTIVE
            for au, weights in self.per_unit.items()
        }

    def to_dict(self) -> dict:
        """
        Zapisuje model do postaci nadającej się do JSON-a.

        Returns:
            Słownik z listami liczb zamiast tablic
        """
        return {
            au: {
                "weights": weights.weights.tolist(),
                "mean": weights.mean.tolist(),
                "std": weights.std.tolist(),
                "threshold": weights.threshold,
            }
            for au, weights in self.per_unit.items()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AUGeometryModel":
        """
        Odtwarza model z postaci zapisanej w JSON-ie.

        Args:
            data: Słownik z `to_dict`

        Returns:
            Gotowy model
        """
        return cls(
            {
                au: AUWeights(
                    weights=np.asarray(zapis["weights"], dtype=float),
                    mean=np.asarray(zapis["mean"], dtype=float),
                    std=np.asarray(zapis["std"], dtype=float),
                    threshold=float(zapis["threshold"]),
                )
                for au, zapis in data.items()
            }
        )
