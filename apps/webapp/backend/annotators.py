"""
Zespół anotatorów i podział NAGRAŃ między nich.

Wszyscy pracują na jednym, wspólnym katalogu — dzieli się nie materiał na dysku,
tylko przydział. Jednostką podziału jest NAGRANIE, nie pojedyncza para: pary
z jednego nagrania to ten sam pies w tej samej scenie, więc rozdzielenie ich
między ludzi kazałoby czterem osobom wdrażać się w ten sam kontekst.

Przydział liczy się ze SKRÓTU nazwy nagrania, a nie z jego pozycji na liście.
To jedyna własność, która pozwala dosypywać wideo bez rozwalania pracy: przy
podziale po pozycji dołożenie jednego nagrania przesunęłoby przydział wszystkich
następnych i ludzie zaczęliby oceniać cudze materiały, a część zostałaby
oceniona dwa razy. Przy podziale po skrócie nowe nagranie dostaje właściciela,
a wszystkie dotychczasowe zostają tam, gdzie były.

Skrót musi być STABILNY między uruchomieniami i maszynami, dlatego sha256,
a nie wbudowany `hash()` — ten jest w Pythonie losowany per proces, więc
przydział zmieniałby się po każdym restarcie backendu.
"""

import hashlib
from dataclasses import dataclass

# Ile procent nagrań ocenia KAŻDY — materiał do policzenia zgodności (kappa).
# Bez ani jednego nagrania ocenionego niezależnie przez dwie osoby nie da się
# wykazać, że etykiety są powtarzalne. Wybór przez skrót, więc dosypanie wideo
# nie zmienia, które nagrania są wspólne.
SHARED_PERCENT: int = 5

# Podstawa reszty przy wyznaczaniu bloku wspólnego
_SHARED_MODULO: int = 100

# Fragment skrótu czytany przy decyzji „czy wspólne" — inny niż przy „czyje",
# żeby obie decyzje były niezależne
_SHARED_DIGEST_OFFSET: int = 8


@dataclass(frozen=True)
class Annotator:
    """
    Osoba anotująca.

    Attributes:
        key: Identyfikator w nazwach plików etykiet (bez spacji i znaków spoza ASCII)
        display: Nazwa pokazywana w interfejsie
    """

    key: str
    display: str


# Kolejność wyznacza, którą część przydziału dostaje dana osoba. Dopisanie
# kogoś w środku przesunęłoby przydziały wszystkim po nim — nowe osoby
# dopisujemy NA KOŃCU.
TEAM: tuple[Annotator, ...] = (
    Annotator(key="anton", display="Антон"),
    Annotator(key="masha", display="Маша"),
    Annotator(key="mafin", display="Мафин"),
    Annotator(key="danek", display="Данек"),
)

TEAM_BY_KEY: dict[str, Annotator] = {member.key: member for member in TEAM}


def shard_index(annotator: str) -> int:
    """
    Zwraca numer części przydziału danej osoby.

    Args:
        annotator: Klucz anotatora

    Returns:
        Pozycja na liście zespołu; 0 dla nieznanej osoby
    """
    for index, member in enumerate(TEAM):
        if member.key == annotator:
            return index
    return 0


def _digest(video: str, offset: int = 0) -> int:
    """
    Liczy stabilny skrót nazwy nagrania.

    Dwie decyzje — „czy wspólne" i „czyje" — biorą NIEZALEŻNE fragmenty skrótu.
    Wyprowadzone z tej samej liczby byłyby skorelowane: reszty przypisane blokowi
    wspólnemu nie rozkładają się równo modulo cztery, więc odsianie ich krzywiło
    podział między ludzi.

    Args:
        video: Klucz nagrania (katalog klatek)
        offset: Od którego znaku skrótu czytamy

    Returns:
        Liczba wyprowadzona ze skrótu sha256
    """
    digest = hashlib.sha256(video.encode("utf-8")).hexdigest()
    return int(digest[offset : offset + 8], 16)


def is_shared_video(video: str, shared_percent: int = SHARED_PERCENT) -> bool:
    """
    Mówi, czy nagranie oceniają WSZYSCY (materiał na zgodność).

    Args:
        video: Klucz nagrania
        shared_percent: Jaki procent nagrań ma być wspólny

    Returns:
        True, gdy nagranie należy do bloku wspólnego
    """
    return _digest(video, _SHARED_DIGEST_OFFSET) % _SHARED_MODULO < shared_percent


def owns_video(annotator: str, video: str, shared_percent: int = SHARED_PERCENT) -> bool:
    """
    Mówi, czy dane nagranie należy do tej osoby.

    Args:
        annotator: Klucz anotatora
        video: Klucz nagrania (katalog klatek)
        shared_percent: Jaki procent nagrań ocenia każdy

    Returns:
        True, gdy osoba ma to nagranie ocenić
    """
    if is_shared_video(video, shared_percent):
        return True
    return _digest(video) % len(TEAM) == shard_index(annotator)
