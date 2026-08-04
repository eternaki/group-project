"""
Normalizacja kształtu twarzy psa metodą superpozycji Prokrustesa.

Superpozycja Prokrustesa usuwa przesunięcie, skalę i obrót naraz, zostawiając
czysty kształt. To standard geometrycznej morfometrii (Gower 1975; Rohlf & Slice
1990): konfiguracje punktów przesuwa się do początku układu, skaluje do
jednostkowego rozmiaru centroidu i obraca metodą najmniejszych kwadratów.
Dzięki temu porównujemy mimikę mopsa i owczarka w jednej przestrzeni, a poza
głowy przestaje wpływać na wartości.

Obecne w pipeline dzielenie odległości przez rozstaw oczu usuwa tylko skalę i
sprzęga AU z pozą: przy obrocie głowy rozstaw oczu maleje perspektywicznie, więc
wszystkie ratio rosną, mimo że mimika się nie zmieniła.

Metoda jest sprawdzona dokładnie na naszym zbiorze punktów: Martvel i Riemer
(„Automated analysis of emotional expressions in dogs based on geometric
morphometrics", Sci Rep 2025) analizują mimikę psów na landmarkach DogFLW —
„Procrustes superimposition was applied to all the landmark coordinates to
eliminate scaling, translation, and rotation effects between individuals and
poses". Ta sama praca jest źródłem naszych metryk pozy głowy (packages/models/
head_pose.py). Uwaga dla kolejnych czytelników: pracy o samym DETEKTORZE
landmarków DogFLW (Sci Rep 2025, PMC12218811) nie należy z nią mylić — tam
o Prokrustesie nie ma mowy.

Dwie świadome decyzje:

1. **Bez odbicia.** Klasyczny algorytm Kabscha dopuszcza wyznacznik ujemny, co
   odpowiada odbiciu lustrzanemu. Tutaj jest to zabronione — lustrzana morda to
   inny kształt (np. AU po lewej i po prawej stronie pyska to osobne jednostki),
   a odbicie sztucznie zmniejszałoby odległość Prokrustesa między psem patrzącym
   w lewo i w prawo.

2. **Ważenie widocznością.** Punkty niewidoczne mają w naszym pipeline
   współrzędne bliskie (0, 0) albo wprost nieufne. Wpuszczone do dopasowania z
   pełną wagą przesunęłyby centroid i obrót całego kształtu. Dlatego centroid,
   rozmiar i obrót liczymy z wagami równymi widoczności punktu (Gower 1975,
   uogólnienie na wagi). Współrzędne punktów niewidocznych są mimo to
   transformowane i zwracane — konsument ma je odfiltrować po widoczności.

Wynik trafia do anotacji COCO jako `procrustes_keypoints` — wejście dla przyszłej
sieci AU (Sprint 16).
"""

import numpy as np

from packages.data.schemas import NUM_KEYPOINTS

# Wymiar przestrzeni kształtu (anotacje DogFLW są dwuwymiarowe)
NUM_SHAPE_DIMS: int = 2

# Liczba wartości w płaskim zapisie keypoints: [x, y, v] na punkt
FLAT_KEYPOINTS_LENGTH: int = NUM_KEYPOINTS * 3

# Poniżej tego rozmiaru centroidu kształt uznajemy za zdegenerowany (brak wariancji)
MIN_CENTROID_SIZE: float = 1e-9

# Poniżej tej sumy wag widoczność nie niesie informacji — wracamy do wag jednostkowych
MIN_WEIGHT_SUM: float = 1e-9

# Domyślny limit iteracji uzgadniania średniej w GPA. Na podobnych do siebie
# kształtach (jak mordy z DogFLW) zbieżność jest osiągana w kilku iteracjach, ale
# na kształtach losowych zbieżność jest liniowa i wolna (ok. 0.72 na iterację),
# więc limit musi być wysoki — inaczej wynik cicho zależałby od limitu, nie od danych.
DEFAULT_MAX_GPA_ITERATIONS: int = 200

# Próg zbieżności GPA: zmiana kształtu referencyjnego między iteracjami
DEFAULT_GPA_TOLERANCE: float = 1e-10


def _require_finite(array: np.ndarray, label: str) -> None:
    """
    Sprawdza, że tablica nie zawiera NaN ani nieskończoności.

    Anotacje DogFLW zawierają NaN dla punktów zasłoniętych. Bez tej kontroli NaN
    przechodzi przez centrowanie i wysypuje `np.linalg.svd` z komunikatem
    "SVD did not converge", z którego nie wynika, co naprawdę jest nie tak.

    Raises:
        ValueError: Gdy tablica zawiera wartości nieskończone lub NaN
    """
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{label} muszą być skończone (bez NaN i nieskończoności)")


def _split_flat(keypoints_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Rozdziela płaską tablicę na współrzędne (46, 2) i wagi z widoczności.

    Raises:
        ValueError: Gdy liczba wartości keypoints jest nieprawidłowa lub nie są skończone
    """
    array = np.asarray(keypoints_flat, dtype=float).ravel()
    if array.size != FLAT_KEYPOINTS_LENGTH:
        raise ValueError(
            f"Oczekiwano {FLAT_KEYPOINTS_LENGTH} wartości keypoints, otrzymano {array.size}"
        )
    _require_finite(array, "Wartości keypoints")
    reshaped = array.reshape(NUM_KEYPOINTS, 3)
    return reshaped[:, :NUM_SHAPE_DIMS], _weights_from_visibility(reshaped[:, 2])


def _weights_from_visibility(visibility: np.ndarray) -> np.ndarray:
    """
    Zamienia widoczność na wagi dopasowania o sumie równej liczbie punktów.

    Taka normalizacja sprawia, że przy jednakowej widoczności wagi są jednostkowe,
    a rozmiar centroidu pokrywa się z klasyczną definicją z morfometrii.
    """
    weights = np.clip(np.asarray(visibility, dtype=float), 0.0, None)
    total = float(weights.sum())
    if total < MIN_WEIGHT_SUM:
        return np.ones(len(weights), dtype=float)
    return np.asarray(weights * (len(weights) / total), dtype=float)


def _center_and_scale(coords: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Centruje kształt w ważonym centroidzie i skaluje do jednostkowego rozmiaru.

    Kształt zdegenerowany (wszystkie punkty w jednym miejscu) zwracamy tylko
    wyśrodkowany — dzielenie przez zerowy rozmiar dałoby wartości nieskończone.
    """
    centroid = (weights[:, np.newaxis] * coords).sum(axis=0) / weights.sum()
    centered = coords - centroid
    size = float(np.sqrt(np.sum(weights * np.sum(centered**2, axis=1))))
    if size < MIN_CENTROID_SIZE:
        return np.asarray(centered, dtype=float)
    return np.asarray(centered / size, dtype=float)


def _optimal_rotation(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Znajduje obrót najlepiej dopasowujący `source` do `target` (rozkład SVD).

    Wyznacznik ujemny oznacza odbicie lustrzane — jest korygowany odwróceniem
    ostatniej kolumny (poprawka Kabscha), więc zwracana macierz jest zawsze
    właściwym obrotem.
    """
    correlation = (weights[:, np.newaxis] * source).T @ target
    u, _, vt = np.linalg.svd(correlation)
    correction = np.eye(NUM_SHAPE_DIMS)
    if float(np.linalg.det(u @ vt)) < 0.0:
        correction[-1, -1] = -1.0
    return np.asarray(u @ correction @ vt, dtype=float)


def _prepare_reference(reference_shape: np.ndarray) -> np.ndarray:
    """
    Sprowadza kształt referencyjny do postaci znormalizowanej (46, 2).

    Raises:
        ValueError: Gdy kształt referencyjny ma złe wymiary lub wartości nieskończone
    """
    reference = np.asarray(reference_shape, dtype=float)
    if reference.shape != (NUM_KEYPOINTS, NUM_SHAPE_DIMS):
        raise ValueError(
            f"Kształt referencyjny musi mieć wymiary ({NUM_KEYPOINTS}, {NUM_SHAPE_DIMS}), "
            f"otrzymano {reference.shape}"
        )
    _require_finite(reference, "Współrzędne kształtu referencyjnego")

    uniform = np.ones(NUM_KEYPOINTS, dtype=float)
    normalized = _center_and_scale(reference, uniform)
    if float(np.sqrt(np.sum(normalized**2))) < MIN_CENTROID_SIZE:
        # Referencja bez rozpiętości nie definiuje orientacji — dopasowanie byłoby
        # dowolnym obrotem, a wynik wyglądałby na poprawny. Lepiej powiedzieć wprost.
        raise ValueError("Kształt referencyjny jest zdegenerowany (zerowy rozmiar centroidu)")
    return normalized


def _weighted_mean(aligned: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Uśrednia dopasowane kształty punkt po punkcie, ważąc widocznością.

    Samo ważenie dopasowania nie wystarcza: punkt niewidoczny ma współrzędne
    bliskie (0, 0) i wpuszczony do średniej z pełną wagą przesuwa nie tylko
    siebie, ale — przez centroid i rozmiar — cały kształt referencyjny.

    Args:
        aligned: Dopasowane kształty (N, 46, 2)
        weights: Wagi z widoczności (N, 46)

    Returns:
        Uśredniony kształt (46, 2)
    """
    totals = weights.sum(axis=0)
    weighted = (weights[:, :, np.newaxis] * aligned).sum(axis=0)
    fallback = aligned.mean(axis=0)
    usable = totals > MIN_WEIGHT_SUM
    safe_totals = np.where(usable, totals, 1.0)[:, np.newaxis]
    return np.asarray(
        np.where(usable[:, np.newaxis], weighted / safe_totals, fallback), dtype=float
    )


def procrustes_align(
    keypoints_flat: np.ndarray,
    reference_shape: np.ndarray,
) -> np.ndarray:
    """
    Nakłada kształt na referencję metodą Prokrustesa.

    Args:
        keypoints_flat: Keypoints [x0, y0, v0, ...] (138 wartości)
        reference_shape: Kształt referencyjny (46, 2), np. z `mean_shape()`

    Returns:
        Płaska tablica 138 wartości w przestrzeni kształtu (wyśrodkowana,
        o jednostkowym rozmiarze centroidu); widoczność przepisana bez zmian

    Raises:
        ValueError: Gdy keypoints lub kształt referencyjny mają złe wymiary
    """
    coords, weights = _split_flat(keypoints_flat)
    normalized = _center_and_scale(coords, weights)
    reference = _prepare_reference(reference_shape)

    aligned = normalized @ _optimal_rotation(normalized, reference, weights)

    result = np.asarray(keypoints_flat, dtype=float).ravel().copy()
    result[0::3] = aligned[:, 0]
    result[1::3] = aligned[:, 1]
    return np.asarray(result, dtype=float)


def mean_shape(
    shapes: list[np.ndarray],
    max_iterations: int = DEFAULT_MAX_GPA_ITERATIONS,
    tolerance: float = DEFAULT_GPA_TOLERANCE,
) -> np.ndarray:
    """
    Liczy kształt referencyjny metodą uogólnionej analizy Prokrustesa (GPA).

    Algorytm: znormalizuj wszystkie kształty, przyjmij pierwszy jako wstępną
    referencję, a potem naprzemiennie dopasowuj kształty do referencji i licz
    z nich nową referencję, aż przestanie się zmieniać. Zwykła średnia z surowych
    kształtów nie wystarcza — bez obrotu do wspólnego układu uśrednianie rozmywa
    kształt tym bardziej, im bardziej pozy się różnią.

    Args:
        shapes: Lista płaskich tablic keypoints (po 138 wartości)
        max_iterations: Górny limit iteracji uzgadniania średniej. Po jego
            wyczerpaniu bez zbieżności zwracane jest ostatnie przybliżenie.
        tolerance: Próg zbieżności — zmiana referencji poniżej kończy iterowanie

    Returns:
        Kształt referencyjny (46, 2), wyśrodkowany i o jednostkowym rozmiarze

    Raises:
        ValueError: Gdy lista kształtów jest pusta lub kształt ma złe wymiary
    """
    if not shapes:
        raise ValueError("Potrzeba co najmniej jednego kształtu")

    prepared: list[tuple[np.ndarray, np.ndarray]] = []
    for shape in shapes:
        coords, weights = _split_flat(shape)
        prepared.append((_center_and_scale(coords, weights), weights))

    uniform = np.ones(NUM_KEYPOINTS, dtype=float)
    weights_stack = np.array([weights for _, weights in prepared], dtype=float)

    # GPA wyznacza średni kształt tylko z dokładnością do obrotu, więc orientacja
    # wyniku bierze się z punktu startowego — pierwszy kształt jest tu wygodną,
    # deterministyczną kotwicą. Jego niewidoczne punkty są jednak niewiarygodne,
    # więc na starcie ważymy referencję jego własną widocznością; po pierwszej
    # iteracji referencją jest już czysta średnia ważona.
    reference = prepared[0][0]
    reference_weights = prepared[0][1]

    for _ in range(max_iterations):
        aligned = [
            shape @ _optimal_rotation(shape, reference, weights * reference_weights)
            for shape, weights in prepared
        ]
        updated = _center_and_scale(_weighted_mean(np.array(aligned), weights_stack), uniform)
        shift = float(np.sqrt(np.sum((updated - reference) ** 2)))
        reference = updated
        reference_weights = uniform
        if shift < tolerance:
            break

    return reference
