"""
Zespół anotatorów i podział kolejki między nich.

Cztery osoby pracujące na jednym zbiorze muszą dostać ROZŁĄCZNE części, inaczej
połowa wysiłku idzie na to samo. Podział bierze co n-tą parę z kolejki, a nie
kolejne bloki: kolejka jest już poukładana tak, że na początku stoją pary
najbardziej niepewne, więc podział blokami dałby jednej osobie same trudne
przypadki, a innej same oczywiste.

Wyjątkiem jest blok WSPÓLNY na początku. Te pary dostają wszyscy — z rozbieżności
między nimi liczy się zgodność (kappa Cohena), a bez ani jednej pary ocenionej
przez dwie osoby nie da się jej policzyć w ogóle. To dwie godziny pracy, które
odróżniają zbiór danych od zestawu obrazków.
"""

from dataclasses import dataclass

# Ile par z początku kolejki ocenia KAŻDY — materiał do policzenia zgodności.
# Bez tego bloku zbiór nie ma jak udowodnić, że etykiety są powtarzalne.
SHARED_PREFIX_PAIRS: int = 40


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


# Kolejność ma znaczenie: pozycja na liście wyznacza, którą część kolejki
# dostaje dana osoba. Dopisanie kogoś na środku przesunęłoby przydziały
# wszystkim po nim, więc nowe osoby dopisujemy NA KOŃCU.
TEAM: tuple[Annotator, ...] = (
    Annotator(key="anton", display="Антон"),
    Annotator(key="masha", display="Маша"),
    Annotator(key="mafin", display="Мафин"),
    Annotator(key="danek", display="Данек"),
)

TEAM_BY_KEY: dict[str, Annotator] = {member.key: member for member in TEAM}


def shard_index(annotator: str) -> int:
    """
    Zwraca numer części kolejki przypisanej danej osobie.

    Args:
        annotator: Klucz anotatora

    Returns:
        Pozycja na liście zespołu; 0 dla nieznanej osoby (dostaje pierwszą część)
    """
    for index, member in enumerate(TEAM):
        if member.key == annotator:
            return index
    return 0


def owns_pair(annotator: str, review_order: int, shared_prefix: int = SHARED_PREFIX_PAIRS) -> bool:
    """
    Mówi, czy dana para należy do tej osoby.

    Args:
        annotator: Klucz anotatora
        review_order: Pozycja pary w kolejce weryfikacji
        shared_prefix: Ile pierwszych par ocenia każdy (blok na zgodność)

    Returns:
        True, gdy osoba ma tę parę ocenić
    """
    if review_order < shared_prefix:
        return True
    return (review_order - shared_prefix) % len(TEAM) == shard_index(annotator)
