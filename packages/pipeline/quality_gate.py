"""
Bramka jakości kadru — czy z tej klatki wolno w ogóle mierzyć AU.

Audyt wizualny (`docs/sprints/14-batch-annotation/AUDYT.md`) pokazał, że pipeline
przypisuje pełne etykiety kadrom, na których psa widać z profilu albo keypoints
leżą na grzbiecie zamiast na mordzie. Takie kadry nie tracą informacji — one
PRODUKUJĄ fałszywe AU, bo każdy pomiar dzieli się przez rozstaw oczu, a ten
skraca się z obrotem głowy.

Moduł mierzy dwie rzeczy, które łapią różne usterki:

* **asymetria połówek mordy** — łapie profil i obrót. Geometryczna, niezależna
  od tego, co detektor sądzi o swojej pewności.
* **udział słabych keypoints** — łapie okluzje i rozmycie.

Sam licznik pewnych keypoints NIE WYSTARCZA: audyt znalazł kadry z punktami na
łopatce biegnącego psa, na których model raportował 0/46 słabych punktów.
Dlatego bramka wymaga obu warunków naraz.

Bramka działa na PARZE (peak, neutral), a nie na pojedynczej klatce, bo delta AU
liczy się względem klatki neutralnej — zepsuta neutralna psuje każdy pomiar
w całym treku.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from packages.data.schemas import KP, NUM_KEYPOINTS

# Keypoints przychodzą raz jako lista COCO, raz jako tablica z treku — bramka
# przyjmuje jedno i drugie, żeby nie mnożyć konwersji po stronie wywołań.
KeypointsInput = Union[Sequence[float], np.ndarray]

# Poniżej tej pewności keypoint uznajemy za niewiarygodny. Próg zgodny
# z `_count_visible_keypoints` w eksporcie webappu byłby za luźny (0.3) —
# do decyzji o wpuszczeniu kadru do zbioru bierzemy ostrzejszy.
WEAK_KEYPOINT_CONFIDENCE: float = 0.5

# Progi domyślne. Zmierzone na `data/dataset_v2` (4145 anotacji): przepuszczają
# ~55% par, a średnia liczba aktywnych AU spada z 5.65 do 4.56 — odsiew zabiera
# przede wszystkim aktywacje wyprodukowane obrotem głowy.
DEFAULT_MAX_ASYMMETRY: float = 0.20
DEFAULT_MAX_WEAK_RATIO: float = 0.25

# Morda węższa niż tyle pikseli jest nie do zweryfikowania przez człowieka —
# anotator nie rozpozna na niej ani szpary powiek, ani ułożenia warg.
DEFAULT_MIN_FACE_WIDTH_PX: float = 40.0

# Pary punktów symetrycznych względem osi mordy. Przy obrocie głowy jedna
# połówka zbliża się do osi, druga oddala — i to jest sygnał, którego pewność
# detektora nie niesie.
_SYMMETRIC_PAIRS: tuple[tuple[int, int], ...] = (
    (KP.LEFT_EAR_BASE_FRONT, KP.RIGHT_EAR_BASE_FRONT),
    (KP.LEFT_BROW_INNER, KP.RIGHT_BROW_INNER),
    (KP.LEFT_EYE_INNER, KP.RIGHT_EYE_INNER),
    (KP.LEFT_EYE_OUTER, KP.RIGHT_EYE_OUTER),
    (KP.MUZZLE_LEFT, KP.MUZZLE_RIGHT),
    (KP.LEFT_CHEEK_UPPER, KP.RIGHT_CHEEK_UPPER),
    (KP.NOSE_LEFT_WING, KP.NOSE_RIGHT_WING),
    (KP.MOUTH_LEFT_CORNER, KP.MOUTH_RIGHT_CORNER),
)

# Punkty leżące na osi mordy. Uśredniamy kilka zamiast brać sam nos, bo
# pojedynczy źle postawiony punkt przesunąłby oś i zafałszował całą miarę.
_MIDLINE_POINTS: tuple[int, ...] = (
    KP.MUZZLE_TOP,
    KP.NOSE_BRIDGE,
    KP.NOSE_TIP,
    KP.UPPER_LIP_CENTER,
    KP.LOWER_LIP_CENTER,
    KP.CHIN,
)

# Zabezpieczenie przed dzieleniem przez zero przy zdegenerowanych keypoints
_EPSILON: float = 1e-6

# Powody odrzucenia — po polsku, bo trafiają wprost do interfejsu anotatora
REASON_ASYMMETRY: str = "profil lub obrót głowy"
REASON_WEAK_KEYPOINTS: str = "za dużo niepewnych keypoints"
REASON_SMALL_FACE: str = "morda za mała do weryfikacji"
REASON_NO_KEYPOINTS: str = "brak keypoints"


@dataclass(frozen=True)
class QualityThresholds:
    """
    Progi bramki jakości.

    Attributes:
        max_asymmetry: Maksymalna mediana asymetrii połówek mordy
        max_weak_ratio: Maksymalny udział keypoints poniżej progu pewności
        min_face_width: Minimalna szerokość mordy w pikselach
    """

    max_asymmetry: float = DEFAULT_MAX_ASYMMETRY
    max_weak_ratio: float = DEFAULT_MAX_WEAK_RATIO
    min_face_width: float = DEFAULT_MIN_FACE_WIDTH_PX


@dataclass(frozen=True)
class FrameQuality:
    """
    Ocena jednej klatki.

    Attributes:
        asymmetry: Mediana asymetrii par symetrycznych (0 = idealny front)
        weak_ratio: Udział keypoints o pewności poniżej `WEAK_KEYPOINT_CONFIDENCE`
        face_width: Szerokość mordy w pikselach (rozstaw policzków)
        is_usable: Czy klatka przechodzi wszystkie progi
        reasons: Powody odrzucenia; pusta krotka, gdy klatka przeszła
    """

    asymmetry: float
    weak_ratio: float
    face_width: float
    is_usable: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PairQuality:
    """
    Ocena pary klatek (szczytowa, neutralna).

    Attributes:
        peak: Ocena klatki szczytowej
        neutral: Ocena klatki neutralnej
        is_usable: Czy OBIE klatki przechodzą bramkę
        reasons: Powody odrzucenia z przedrostkiem wskazującym klatkę
    """

    peak: FrameQuality
    neutral: FrameQuality
    is_usable: bool
    reasons: tuple[str, ...]


def split_keypoints(keypoints: KeypointsInput) -> tuple[np.ndarray, np.ndarray]:
    """
    Rozdziela płaską listę COCO na współrzędne i pewności.

    Args:
        keypoints: 138 wartości w układzie [x0, y0, v0, x1, y1, v1, ...]

    Returns:
        Para (współrzędne o kształcie (46, 2), pewności o kształcie (46,))

    Raises:
        ValueError: Gdy lista nie ma 3 × NUM_KEYPOINTS wartości
    """
    expected = NUM_KEYPOINTS * 3
    if len(keypoints) != expected:
        raise ValueError(
            f"keypoints musi mieć {expected} wartości, otrzymano {len(keypoints)}"
        )
    flat = np.asarray(keypoints, dtype=float).reshape(NUM_KEYPOINTS, 3)
    return flat[:, :2], flat[:, 2]


def _midline_normal(coords: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Wyznacza oś mordy jako PROSTĄ i zwraca jej normalną.

    Oś liczona jako punkt, a nie prosta, byłaby bezużyteczna: odległość punktu
    od środka mordy jest zdominowana przez położenie w pionie, więc asymetria
    lewo-prawo tonęłaby w niej. Odległość od prostej mierzy dokładnie to, o co
    chodzi — jak daleko w bok leży punkt.

    Kierunek prostej bierzemy z rozkładu SVD punktów osiowych, więc oś obraca
    się razem z przechyłem głowy i przechył nie udaje asymetrii.

    Args:
        coords: Współrzędne keypoints o kształcie (46, 2)

    Returns:
        Para (punkt na osi, wektor normalny do osi) albo None, gdy punkty
        osiowe są zdegenerowane (zlepione w jedno miejsce)
    """
    midline = coords[list(_MIDLINE_POINTS)]
    center = midline.mean(axis=0)
    centered = midline - center
    if float(np.abs(centered).max()) < _EPSILON:
        return None
    direction = np.linalg.svd(centered, full_matrices=False)[2][0]
    normal = np.array([-direction[1], direction[0]], dtype=float)
    return center, normal


def face_asymmetry(coords: np.ndarray) -> float:
    """
    Mierzy asymetrię połówek mordy względem jej osi.

    Dla każdej pary symetrycznej liczy odległość obu punktów od osi mordy
    (prostopadle do niej) i porównuje je znormalizowaną różnicą. Przy ujęciu
    na wprost obie odległości są równe, przy obrocie — rozjeżdżają się.

    Args:
        coords: Współrzędne keypoints o kształcie (46, 2)

    Returns:
        Mediana asymetrii par w zakresie [0, 1]; 0 znaczy idealną symetrię,
        1 zwracamy też wtedy, gdy osi nie da się wyznaczyć
    """
    axis = _midline_normal(coords)
    if axis is None:
        return 1.0
    center, normal = axis

    scores = []
    for left, right in _SYMMETRIC_PAIRS:
        left_dist = abs(float((coords[left] - center) @ normal))
        right_dist = abs(float((coords[right] - center) @ normal))
        total = left_dist + right_dist
        if total > _EPSILON:
            scores.append(abs(left_dist - right_dist) / total)
    if not scores:
        return 1.0
    return float(np.median(scores))


def weak_keypoint_ratio(
    confidences: np.ndarray,
    threshold: float = WEAK_KEYPOINT_CONFIDENCE,
) -> float:
    """
    Liczy udział keypoints, którym detektor sam nie ufa.

    Args:
        confidences: Pewności keypoints o kształcie (46,)
        threshold: Próg, poniżej którego punkt uznajemy za słaby

    Returns:
        Udział słabych punktów w zakresie [0, 1]
    """
    if confidences.size == 0:
        return 1.0
    return float((confidences < threshold).mean())


def face_width(coords: np.ndarray) -> float:
    """
    Szerokość mordy w pikselach — rozstaw policzków.

    Args:
        coords: Współrzędne keypoints o kształcie (46, 2)

    Returns:
        Odległość między policzkami w pikselach
    """
    return float(
        np.linalg.norm(coords[KP.LEFT_CHEEK_UPPER] - coords[KP.RIGHT_CHEEK_UPPER])
    )


# Zapas wokół boksu psa, w którym punkt jeszcze uznajemy za jego. Ucho albo
# broda potrafią wystawać poza boks ciała, więc bez zapasu odcinalibyśmy
# poprawne pomiary. Poza tym zapasem punkt twarzy nie należy już do tego psa.
BBOX_MARGIN_RATIO: float = 0.25


def hide_out_of_frame(
    keypoints: KeypointsInput,
    image_size: Optional[tuple[float, float]] = None,
    bbox: Optional[Sequence[float]] = None,
    margin_ratio: float = BBOX_MARGIN_RATIO,
) -> list[float]:
    """
    Ukrywa punkty leżące poza kadrem albo poza psem.

    Punkt poza obrazem to ekstrapolacja modelu, a nie pomiar — nie ma tam czego
    zobaczyć. Punkt poza boksem psa (z zapasem) należy do czegoś innego niż ta
    morda. W obu wypadkach zostawienie go widocznym psuje zbiór po cichu, a przy
    okazji jest nie do poprawienia ręcznie: edytor kadruje widok do boksu, więc
    anotator fizycznie nie może takiego punktu kliknąć.

    Args:
        keypoints: 138 wartości COCO [x0, y0, v0, ...]
        image_size: (szerokość, wysokość) kadru; None wyłącza to sprawdzenie
        bbox: Boks psa [x, y, w, h]; None wyłącza to sprawdzenie
        margin_ratio: Zapas wokół boksu jako ułamek jego wymiarów

    Returns:
        Nowa lista keypoints z wyzerowaną widocznością punktów poza kadrem
    """
    coords, confidences = split_keypoints(keypoints)
    hidden = np.zeros(len(confidences), dtype=bool)

    if image_size is not None:
        width, height = image_size
        hidden |= (
            (coords[:, 0] < 0)
            | (coords[:, 1] < 0)
            | (coords[:, 0] >= width)
            | (coords[:, 1] >= height)
        )

    if bbox is not None and len(bbox) == 4:
        x, y, box_width, box_height = (float(value) for value in bbox)
        margin_x, margin_y = box_width * margin_ratio, box_height * margin_ratio
        hidden |= (
            (coords[:, 0] < x - margin_x)
            | (coords[:, 0] > x + box_width + margin_x)
            | (coords[:, 1] < y - margin_y)
            | (coords[:, 1] > y + box_height + margin_y)
        )

    corrected = confidences.copy()
    corrected[hidden] = 0.0
    flat = np.column_stack([coords, corrected]).reshape(-1)
    return [float(value) for value in flat]


def assess_frame(
    keypoints: Optional[KeypointsInput],
    thresholds: Optional[QualityThresholds] = None,
) -> FrameQuality:
    """
    Ocenia, czy z klatki wolno mierzyć AU.

    Args:
        keypoints: 138 wartości COCO albo None, gdy detektor nic nie znalazł
        thresholds: Progi bramki; domyślnie `QualityThresholds()`

    Returns:
        `FrameQuality` z wartościami miar i powodami odrzucenia
    """
    limits = thresholds or QualityThresholds()
    if keypoints is None:
        return FrameQuality(
            asymmetry=1.0,
            weak_ratio=1.0,
            face_width=0.0,
            is_usable=False,
            reasons=(REASON_NO_KEYPOINTS,),
        )

    coords, confidences = split_keypoints(keypoints)
    asymmetry = face_asymmetry(coords)
    weak = weak_keypoint_ratio(confidences)
    width = face_width(coords)

    reasons: list[str] = []
    if asymmetry > limits.max_asymmetry:
        reasons.append(REASON_ASYMMETRY)
    if weak > limits.max_weak_ratio:
        reasons.append(REASON_WEAK_KEYPOINTS)
    if width < limits.min_face_width:
        reasons.append(REASON_SMALL_FACE)

    return FrameQuality(
        asymmetry=asymmetry,
        weak_ratio=weak,
        face_width=width,
        is_usable=not reasons,
        reasons=tuple(reasons),
    )


def assess_pair(
    peak_keypoints: Optional[KeypointsInput],
    neutral_keypoints: Optional[KeypointsInput],
    thresholds: Optional[QualityThresholds] = None,
) -> PairQuality:
    """
    Ocenia parę (klatka szczytowa, klatka neutralna).

    Para jest jednostką pomiaru AU: delta liczy się względem klatki neutralnej,
    więc zepsuta neutralna unieważnia pomiar tak samo jak zepsuty szczyt.

    Args:
        peak_keypoints: Keypoints klatki szczytowej (138 wartości albo None)
        neutral_keypoints: Keypoints klatki neutralnej (138 wartości albo None)
        thresholds: Progi bramki; domyślnie `QualityThresholds()`

    Returns:
        `PairQuality`; `is_usable` tylko wtedy, gdy przechodzą OBIE klatki
    """
    limits = thresholds or QualityThresholds()
    peak = assess_frame(peak_keypoints, limits)
    neutral = assess_frame(neutral_keypoints, limits)
    reasons = tuple(
        [f"szczytowa: {reason}" for reason in peak.reasons]
        + [f"neutralna: {reason}" for reason in neutral.reasons]
    )
    return PairQuality(
        peak=peak,
        neutral=neutral,
        is_usable=peak.is_usable and neutral.is_usable,
        reasons=reasons,
    )
