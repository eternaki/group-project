#!/usr/bin/env python3
"""
Ile warte są reguły AU — zmierzone na parach ocenionych przez człowieka.

    python -m scripts.annotation.eval_au_rules

Reguły geometryczne (`packages/models/delta_action_units.py`) zapalają AU, gdy
zmiana odległości między punktami przekroczy jeden wspólny próg. Ten skrypt
sprawdza, ile z tych zapaleń potwierdza człowiek, i porównuje cztery warianty
etykiety automatycznej:

    reguły surowe          samo `is_active`
    szumowy gejt           `au_auto_verdict` (aktywacja tylko gdy sygnał > szum)
    próg kalibrowany       osobny próg na każde AU, dobrany na danych
    cel precyzji           próg podniesiony pod zadaną precyzję

DLACZEGO PODZIAŁ IDZIE PO NAGRANIACH, NIE PO PARACH. Kadry jednego nagrania
pokazują tego samego psa w tej samej scenie i te same wartości `ratio` wracają
w nich wielokrotnie. Próg dobrany na części kadrów nagrania i sprawdzony na
pozostałych mierzy pamięć, nie zdolność uogólnienia — przy podziale losowym po
parach precyzja wychodzi zawyżona. Fold trzyma całe nagranie po jednej stronie.

CO JEST CECHĄ. `|ratio - 1|`, a nie samo `ratio`. Reguła porównuje wartość
z progiem od góry, ale część AU aktywuje się przez ZMNIEJSZENIE odległości
(zmierzone: AD19 ma AUC 0.325 na surowym `ratio` i 0.723 na module) — miarą
aktywacji jest odchylenie od klatki neutralnej w OBIE strony.
"""

import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
DEFAULT_RELEASE: Path = REPO_ROOT / "data" / "dataset_final" / "release" / "annotations.json"

# Werdykty człowieka, które w ogóle są oceną. `not_observable` znaczy „nie wiem"
# i nie wolno go liczyć ani jako trafienie, ani jako pomyłkę.
OCENA_AKTYWNA: str = "active"
OCENA_SPOCZYNEK: str = "inactive"

# Poniżej tylu potwierdzeń człowieka próg dobrany na danym AU jest szumem —
# przy 5 obserwacjach jeden kadr przesuwa precyzję o 20 punktów.
MIN_POTWIERDZEN: int = 10

# Liczba części w sprawdzianie krzyżowym i ziarno przydziału nagrań do części.
FOLDY: int = 5
ZIARNO: int = 0

# Minimalna liczba orzeczeń, przy której precyzja na zbiorze uczącym w ogóle
# coś znaczy — bez tego próg „1 trafienie na 1 orzeczenie" daje 100%.
MIN_ORZECZEN_DO_PROGU: int = 3


class Komorka:
    """Jedna ocena: konkretne AU na konkretnej parze."""

    __slots__ = ("video", "odchylenie", "aktywne")

    def __init__(self, video: str, odchylenie: float, aktywne: bool) -> None:
        self.video = video
        self.odchylenie = odchylenie
        self.aktywne = aktywne


def wczytaj_komorki(release: Path) -> dict[str, list[Komorka]]:
    """
    Zbiera oceny człowieka razem z pomiarem reguł, w rozbiciu na AU.

    Args:
        release: Ścieżka `annotations.json` złotego podzbioru

    Returns:
        Słownik AU -> lista ocenionych komórek
    """
    coco = json.loads(release.read_text(encoding="utf-8"))
    obrazy = {obraz["id"]: obraz for obraz in coco.get("images", [])}
    komorki: dict[str, list[Komorka]] = defaultdict(list)
    for anotacja in coco.get("annotations", []):
        werdykty = anotacja.get("au_verdicts")
        if not werdykty:
            continue
        video = obrazy.get(anotacja.get("image_id"), {}).get("source_video") or "?"
        analiza = anotacja.get("au_analysis") or {}
        for au, werdykt in werdykty.items():
            if werdykt not in (OCENA_AKTYWNA, OCENA_SPOCZYNEK):
                continue
            wartosc = analiza.get(au)
            if not isinstance(wartosc, dict):
                continue
            odchylenie = abs(float(wartosc.get("ratio", 1.0)) - 1.0)
            komorki[au].append(Komorka(video, odchylenie, werdykt == OCENA_AKTYWNA))
    return dict(komorki)


def podziel_nagrania(komorki: dict[str, list[Komorka]]) -> dict[str, int]:
    """
    Rozdziela nagrania na części sprawdzianu krzyżowego.

    Args:
        komorki: Oceny w rozbiciu na AU

    Returns:
        Słownik nagranie -> numer części
    """
    nagrania = sorted({komorka.video for lista in komorki.values() for komorka in lista})
    # Kolejność wyznacza skrót nazwy, nie los — ten sam podział przy każdym
    # uruchomieniu, bez zależności od wersji generatora liczb losowych.
    posortowane = sorted(nagrania, key=lambda nazwa: _skrot(nazwa))
    return {nazwa: numer % FOLDY for numer, nazwa in enumerate(posortowane)}


def _skrot(tekst: str) -> int:
    """
    Liczy powtarzalny skrót nazwy (`hash()` w Pythonie jest solony na proces).

    Args:
        tekst: Nazwa nagrania

    Returns:
        Liczba całkowita
    """
    wynik = 0
    for znak in tekst:
        wynik = (wynik * 131 + ord(znak)) % (2**31)
    return wynik


class Wynik:
    """Zliczenia trafień i pomyłek jednego wariantu etykiety."""

    def __init__(self) -> None:
        self.trafione = 0
        self.falszywe = 0
        self.przeoczone = 0

    def dodaj(self, orzeczono: bool, naprawde: bool) -> None:
        """
        Dokłada jedną ocenę.

        Args:
            orzeczono: Czy wariant uznał AU za aktywne
            naprawde: Czy człowiek uznał AU za aktywne
        """
        if orzeczono and naprawde:
            self.trafione += 1
        elif orzeczono:
            self.falszywe += 1
        elif naprawde:
            self.przeoczone += 1

    @property
    def precyzja(self) -> float:
        """Udział trafień wśród orzeczeń."""
        orzeczenia = self.trafione + self.falszywe
        return self.trafione / orzeczenia if orzeczenia else 0.0

    @property
    def pokrycie(self) -> float:
        """Udział znalezionych aktywacji wśród wszystkich."""
        aktywacje = self.trafione + self.przeoczone
        return self.trafione / aktywacje if aktywacje else 0.0

    @property
    def f1(self) -> float:
        """Średnia harmoniczna precyzji i pokrycia."""
        suma = self.precyzja + self.pokrycie
        return 2 * self.precyzja * self.pokrycie / suma if suma else 0.0

    def __iadd__(self, inny: "Wynik") -> "Wynik":
        self.trafione += inny.trafione
        self.falszywe += inny.falszywe
        self.przeoczone += inny.przeoczone
        return self


def prog_na_f1(uczace: list[Komorka]) -> Optional[float]:
    """
    Dobiera próg odchylenia maksymalizujący F1 na zbiorze uczącym.

    Args:
        uczace: Komórki zbioru uczącego

    Returns:
        Próg albo None, gdy w zbiorze nie ma ani jednej aktywacji
    """
    if not any(komorka.aktywne for komorka in uczace):
        return None
    return _najlepszy_prog(uczace, lambda wynik: wynik.f1)


def prog_na_precyzje(uczace: list[Komorka], cel: float) -> Optional[float]:
    """
    Szuka najniższego progu, przy którym precyzja na uczącym sięga celu.

    Najniższy, bo przy równej precyzji chcemy jak najszerszego pokrycia.

    Args:
        uczace: Komórki zbioru uczącego
        cel: Wymagana precyzja (0-1)

    Returns:
        Próg albo None, gdy celu nie da się osiągnąć
    """
    znaleziony: Optional[float] = None
    for prog in sorted({komorka.odchylenie for komorka in uczace}, reverse=True):
        wynik = _zlicz(uczace, prog)
        orzeczenia = wynik.trafione + wynik.falszywe
        if orzeczenia >= MIN_ORZECZEN_DO_PROGU and wynik.precyzja >= cel:
            znaleziony = prog
    return znaleziony


def _najlepszy_prog(uczace: list[Komorka], ocena: Callable[[Wynik], float]) -> float:
    """
    Przechodzi kandydatów na próg i zwraca ten o najwyższej ocenie.

    Args:
        uczace: Komórki zbioru uczącego
        ocena: Funkcja oceniająca zliczenia

    Returns:
        Wybrany próg
    """
    najlepszy, najwyzsza = 0.0, -1.0
    for prog in sorted({komorka.odchylenie for komorka in uczace}):
        wartosc = ocena(_zlicz(uczace, prog))
        if wartosc > najwyzsza:
            najlepszy, najwyzsza = prog, wartosc
    return najlepszy


def _zlicz(komorki: list[Komorka], prog: float) -> Wynik:
    """
    Zlicza trafienia przy zadanym progu.

    Args:
        komorki: Oceniane komórki
        prog: Próg odchylenia

    Returns:
        Zliczenia
    """
    wynik = Wynik()
    for komorka in komorki:
        wynik.dodaj(komorka.odchylenie >= prog, komorka.aktywne)
    return wynik


def sprawdzian_krzyzowy(
    komorki: list[Komorka],
    przydzial: dict[str, int],
    dobierz: Callable[[list[Komorka]], Optional[float]],
) -> Wynik:
    """
    Mierzy wariant progowy na nagraniach nieoglądanych przy dobieraniu progu.

    Args:
        komorki: Wszystkie oceny jednego AU
        przydzial: Nagranie -> numer części
        dobierz: Funkcja dobierająca próg na zbiorze uczącym

    Returns:
        Zliczenia zsumowane po częściach
    """
    laczny = Wynik()
    for fold in range(FOLDY):
        uczace = [k for k in komorki if przydzial[k.video] != fold]
        testowe = [k for k in komorki if przydzial[k.video] == fold]
        if not testowe or not uczace:
            continue
        prog = dobierz(uczace)
        for komorka in testowe:
            orzeczono = prog is not None and komorka.odchylenie >= prog
            laczny.dodaj(orzeczono, komorka.aktywne)
    return laczny


def auc(komorki: list[Komorka]) -> Optional[float]:
    """
    Liczy pole pod krzywą ROC — 0.5 znaczy zero informacji w pomiarze.

    Args:
        komorki: Oceny jednego AU

    Returns:
        AUC albo None, gdy brakuje jednej z klas
    """
    aktywne = [k.odchylenie for k in komorki if k.aktywne]
    spoczynek = [k.odchylenie for k in komorki if not k.aktywne]
    if not aktywne or not spoczynek:
        return None
    lepsze = sum(
        1.0 if a > s else 0.5 if math.isclose(a, s) else 0.0 for a in aktywne for s in spoczynek
    )
    return lepsze / (len(aktywne) * len(spoczynek))


def raport_regul_surowych(release: Path) -> Wynik:
    """
    Mierzy samo `is_active` — punkt odniesienia bez żadnego dobierania.

    Args:
        release: Ścieżka `annotations.json`

    Returns:
        Zliczenia
    """
    coco = json.loads(release.read_text(encoding="utf-8"))
    wynik = Wynik()
    for anotacja in coco.get("annotations", []):
        analiza = anotacja.get("au_analysis") or {}
        for au, werdykt in (anotacja.get("au_verdicts") or {}).items():
            if werdykt not in (OCENA_AKTYWNA, OCENA_SPOCZYNEK):
                continue
            wartosc = analiza.get(au)
            if isinstance(wartosc, dict):
                wynik.dodaj(bool(wartosc.get("is_active")), werdykt == OCENA_AKTYWNA)
    return wynik


def raport_gejtu(release: Path) -> Wynik:
    """
    Mierzy etykietę po szumowym gejcie (`au_auto_verdict`).

    Args:
        release: Ścieżka `annotations.json`

    Returns:
        Zliczenia
    """
    coco = json.loads(release.read_text(encoding="utf-8"))
    wynik = Wynik()
    for anotacja in coco.get("annotations", []):
        auto = anotacja.get("au_auto_verdict") or {}
        for au, werdykt in (anotacja.get("au_verdicts") or {}).items():
            if werdykt not in (OCENA_AKTYWNA, OCENA_SPOCZYNEK):
                continue
            if au in auto:
                wynik.dodaj(auto[au] == OCENA_AKTYWNA, werdykt == OCENA_AKTYWNA)
    return wynik


def _linia(nazwa: str, wynik: Wynik) -> str:
    """
    Formatuje jeden wiersz raportu.

    Args:
        nazwa: Opis wariantu
        wynik: Zliczenia

    Returns:
        Gotowy wiersz
    """
    return (
        f"{nazwa:34} precyzja {wynik.precyzja:6.1%}   pokrycie {wynik.pokrycie:6.1%}   "
        f"F1 {wynik.f1:6.1%}   ({wynik.trafione} traf. / "
        f"{wynik.trafione + wynik.falszywe} orzeczen)"
    )


def main() -> None:
    """Uruchamia pomiar i wypisuje raport."""
    parser = argparse.ArgumentParser(description="Ocena regul AU wzgledem czlowieka")
    parser.add_argument("--release", default=str(DEFAULT_RELEASE), help="annotations.json zbioru")
    parser.add_argument(
        "--cel-precyzji", type=float, default=0.5, help="Precyzja wymagana od wariantu ostrego"
    )
    args = parser.parse_args()

    release = Path(args.release)
    komorki = wczytaj_komorki(release)
    przydzial = podziel_nagrania(komorki)
    ocenione = sum(len(lista) for lista in komorki.values())
    logger.info("Nagran : %d", len(przydzial))
    logger.info("Ocen AU: %d", ocenione)

    logger.info("")
    logger.info("%s", _linia("reguly surowe (is_active)", raport_regul_surowych(release)))
    logger.info("%s", _linia("szumowy gejt (au_auto_verdict)", raport_gejtu(release)))

    mocne = {au: lista for au, lista in komorki.items() if sum(k.aktywne for k in lista) >= MIN_POTWIERDZEN}
    kalibrowany, ostry = Wynik(), Wynik()
    for lista in mocne.values():
        kalibrowany += sprawdzian_krzyzowy(lista, przydzial, prog_na_f1)
        ostry += sprawdzian_krzyzowy(
            lista, przydzial, lambda u: prog_na_precyzje(u, args.cel_precyzji)
        )
    logger.info("%s", _linia(f"prog kalibrowany ({len(mocne)} AU)", kalibrowany))
    logger.info("%s", _linia(f"cel precyzji {args.cel_precyzji:.0%}", ostry))

    logger.info("")
    logger.info("%-10s %6s %6s %7s   %s", "AU", "ocen", "aktyw", "AUC", "prog kalibrowany")
    for au, lista in sorted(komorki.items(), key=lambda kv: -sum(k.aktywne for k in kv[1])):
        potwierdzenia = sum(k.aktywne for k in lista)
        pole = auc(lista)
        if au in mocne:
            wynik = sprawdzian_krzyzowy(lista, przydzial, prog_na_f1)
            opis = f"precyzja {wynik.precyzja:5.1%}  pokrycie {wynik.pokrycie:5.1%}"
        else:
            opis = "za malo potwierdzen"
        logger.info(
            "%-10s %6d %6d %7s   %s",
            au,
            len(lista),
            potwierdzenia,
            f"{pole:.3f}" if pole is not None else "-",
            opis,
        )


if __name__ == "__main__":
    main()
