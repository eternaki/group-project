#!/usr/bin/env python3
"""
Raport wyników projektu do PDF — dokument dla prowadzącego.

    python -m scripts.report.build_pdf_report

Liczby, które da się policzyć szybko (zbiór do oddania, kuracja), liczone są
przy każdym uruchomieniu, żeby raport nie rozjechał się z danymi. Liczby
pochodzące z surowego COCO batcha (265 MB) i z pomiarów jakości modeli są
stałymi z podanym w komentarzu sposobem odtworzenia — wczytywanie ćwierć
gigabajta przy każdym generowaniu raportu nie jest tego warte.

Czcionka musi być osadzona: wbudowana Helvetica nie ma polskich znaków
(ą, ć, ę, ł, ń, ś, ź, ż leżą poza WinAnsi) i wypadłyby jako puste prostokąty.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
RELEASE_DIR: Path = REPO_ROOT / "data" / "dataset_final" / "release"
DEFAULT_OUTPUT: Path = REPO_ROOT / "docs" / "Raport_wyniki_DogFACS.pdf"

# Rodzina DejaVu jedzie z matplotlibem, więc jest zawsze pod ręką i ma komplet
# polskich znaków. Arial z Windows byłby drugą opcją, ale nie na każdej maszynie.
FONT_DIR: Path = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
FONT_REGULAR: str = "DejaVu"
FONT_BOLD: str = "DejaVu-Bold"

# --- Liczby z surowego COCO batcha -----------------------------------------
# Odtworzenie: patrz `docs/ODTWARZANIE_LICZB.md`
VIDEOS_ATTEMPTED: int = 4553
RAW_FRAMES: int = 25447
RAW_PEAK_CANDIDATES: int = 18929
GATE_PASSED: int = 5284
REJECTION_REASONS: tuple[tuple[str, int], ...] = (
    ("morda za mała do weryfikacji", 14290),
    ("za dużo niepewnych punktów kluczowych", 3103),
    ("profil lub obrót głowy", 825),
    ("punkty postawione byle gdzie", 344),
    ("punkty rozdzielone między dwa psy", 128),
)
SOURCE_VIDEO_GB: float = 4.9

# Praca człowieka: `build_final_dataset` podaje te liczby przy każdym złożeniu.
# W pliku leży mniej par niż przyjęto, bo powtórzenia tej samej klatki
# szczytowej są scalane.
HUMAN_REVIEWED: int = 613
HUMAN_REJECTED: int = 84
HUMAN_DISPUTED: int = 11

# --- Jakość modeli (metryki z treningu) -------------------------------------
MODEL_QUALITY: tuple[tuple[str, str, str, str], ...] = (
    ("Detekcja psa", "YOLOv8m", "—", "gotowe wagi, bez dotrenowania"),
    ("Klasyfikacja rasy", "EfficientNet-B4", "Top-1 91.5%", "120 ras"),
    ("Detekcja mordy", "YOLOv8n", "mAP50 0.99", "kadrowanie przed punktami"),
    ("Punkty kluczowe", "HRNet-W48", "NME 0.091 / PCK 0.748", "46 punktów DogFLW"),
    ("Jednostki ruchu (AU)", "regresja na geometrii", "precyzja 32.7%", "21 AU DogFACS"),
    ("Emocje", "reguły na AU", "brak metryki", "9 klas, bez danych odniesienia"),
)

# --- Wyniki AU (sprawdzian krzyżowy z podziałem po nagraniach) --------------
AU_RESULTS: tuple[tuple[str, str, str, str], ...] = (
    ("Reguły geometryczne (stan wyjściowy)", "5.0%", "40.5%", "8.9%"),
    ("Reguły + szumowy gejt", "5.7%", "29.3%", "9.5%"),
    ("Reguły + próg dobrany na każde AU", "10.7%", "38.5%", "16.7%"),
    ("Model uczony z pełnej geometrii", "26.8%", "22.5%", "24.5%"),
    ("Model + spójny standard anotacji", "32.7%", "26.3%", "29.1%"),
)

AU_PER_UNIT: tuple[tuple[str, str, str, str], ...] = (
    ("AU25 — rozchylenie warg", "35", "52.6%", "57.1%"),
    ("AU116 — obniżenie wargi dolnej", "17", "44.4%", "47.1%"),
    ("AD19 — pokazanie języka", "35", "41.0%", "45.7%"),
    ("AU27 — rozciągnięcie pyska", "24", "37.0%", "41.7%"),
    ("AU26 — opuszczenie żuchwy", "20", "35.3%", "30.0%"),
    ("AU118 — wysunięcie warg", "12", "60.0%", "25.0%"),
)


@dataclass(frozen=True)
class Counts:
    """Liczby policzone ze zbioru przy uruchomieniu raportu."""

    pairs_total: int
    frames_total: int
    videos_with_output: int
    human_pairs: int
    human_frames: int
    emotions: tuple[tuple[str, int], ...]
    breeds: int


def register_fonts() -> None:
    """Rejestruje czcionkę z polskimi znakami."""
    pdfmetrics.registerFont(TTFont(FONT_REGULAR, str(FONT_DIR / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont(FONT_BOLD, str(FONT_DIR / "DejaVuSans-Bold.ttf")))


def collect_counts() -> Counts:
    """
    Liczy to, co da się policzyć ze zbioru do oddania.

    Returns:
        Komplet liczb do raportu
    """
    full = json.loads((RELEASE_DIR / "annotations_full.json").read_text(encoding="utf-8"))
    gold = json.loads((RELEASE_DIR / "annotations.json").read_text(encoding="utf-8"))

    peaks = [a for a in full["annotations"] if a.get("frame_role") == "peak"]
    videos = {i.get("source_video") for i in full["images"] if i.get("source_video")}
    human = [a for a in gold["annotations"] if a.get("frame_role") == "peak"]

    emotions: dict[str, int] = {}
    breeds: set[str] = set()
    for annotation in human:
        emotion = annotation.get("emotion")
        if emotion:
            emotions[emotion] = emotions.get(emotion, 0) + 1
        if annotation.get("breed"):
            breeds.add(annotation["breed"])

    return Counts(
        pairs_total=len(peaks),
        frames_total=len(full["images"]),
        videos_with_output=len(videos),
        human_pairs=len(human),
        human_frames=len(gold["images"]),
        emotions=tuple(sorted(emotions.items(), key=lambda kv: -kv[1])),
        breeds=len(breeds),
    )


def _styles() -> dict:
    """
    Buduje style akapitów oparte na osadzonej czcionce.

    Returns:
        Mapa nazwa stylu -> styl
    """
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Tytul", parent=base["Title"], fontName=FONT_BOLD, fontSize=19, leading=24,
            spaceAfter=2 * mm,
        ),
        "subtitle": ParagraphStyle(
            "Podtytul", parent=base["Normal"], fontName=FONT_REGULAR, fontSize=10.5,
            leading=15, textColor=colors.HexColor("#555555"), alignment=1, spaceAfter=7 * mm,
        ),
        "h2": ParagraphStyle(
            "Naglowek", parent=base["Heading2"], fontName=FONT_BOLD, fontSize=13,
            leading=17, spaceBefore=6 * mm, spaceAfter=2.5 * mm,
            textColor=colors.HexColor("#1a3d5c"),
        ),
        "body": ParagraphStyle(
            "Tresc", parent=base["Normal"], fontName=FONT_REGULAR, fontSize=9.7,
            leading=14.5, alignment=TA_JUSTIFY, spaceAfter=2.5 * mm,
        ),
        "note": ParagraphStyle(
            "Nota", parent=base["Normal"], fontName=FONT_REGULAR, fontSize=8.6,
            leading=12.5, textColor=colors.HexColor("#666666"), spaceAfter=2 * mm,
        ),
        "caption": ParagraphStyle(
            "Podpis", parent=base["Normal"], fontName=FONT_REGULAR, fontSize=8.4,
            leading=11.5, textColor=colors.HexColor("#666666"), spaceBefore=1 * mm,
            spaceAfter=3.5 * mm,
        ),
    }


def _cell(text: str, bold: bool = False, right: bool = False) -> Paragraph:
    """
    Owija treść komórki w akapit, żeby długi tekst się ZAWIJAŁ, a nie wychodził
    poza tabelę.

    Zwykły napis w komórce `Table` nie ma gdzie się złamać i po cichu wychodzi
    na sąsiednią kolumnę — przy opisach modeli i nazwach jednostek ruchu było
    to widać gołym okiem.

    Args:
        text: Treść komórki
        bold: Czy pogrubić
        right: Czy wyrównać do prawej

    Returns:
        Akapit gotowy do wstawienia w tabelę
    """
    style = ParagraphStyle(
        "Komorka",
        fontName=FONT_BOLD if bold else FONT_REGULAR,
        fontSize=8.8,
        leading=11.5,
        alignment=2 if right else 0,
        textColor=colors.white if bold and right is False and text == "" else colors.black,
    )
    return Paragraph(text, style)


def _table(
    rows: Sequence[Sequence[str]], widths: Sequence[float], highlight: Optional[int] = None
) -> Table:
    """
    Składa tabelę o jednolitym wyglądzie.

    Args:
        rows: Wiersze, pierwszy jest nagłówkiem
        widths: Szerokości kolumn
        highlight: Numer wiersza do wyróżnienia (licząc z nagłówkiem) albo None

    Returns:
        Gotowa tabela
    """
    naglowek = ParagraphStyle(
        "Naglowek", fontName=FONT_BOLD, fontSize=8.8, leading=11.5, textColor=colors.white
    )
    dane: list[list] = [
        [
            Paragraph(text, ParagraphStyle("H", parent=naglowek, alignment=2 if index else 0))
            for index, text in enumerate(rows[0])
        ]
    ]
    for numer, wiersz in enumerate(rows[1:], start=1):
        pogrub = highlight is not None and numer == highlight
        dane.append(
            [_cell(text, bold=pogrub, right=index > 0) for index, text in enumerate(wiersz)]
        )

    table = Table(dane, colWidths=list(widths), hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3d5c")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8d4de")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6f9")]),
    ]
    if highlight is not None:
        style.append(
            ("BACKGROUND", (0, highlight), (-1, highlight), colors.HexColor("#dcecd8"))
        )
    table.setStyle(TableStyle(style))
    return table


def _funnel_section(counts: Counts, st: dict) -> list:
    """
    Buduje sekcję o skali przetwarzania.

    Args:
        counts: Policzone liczby
        st: Style akapitów

    Returns:
        Elementy dokumentu
    """
    bez_wyniku = VIDEOS_ATTEMPTED - counts.videos_with_output
    rows = [
        ["Etap", "Ilość", "Przechodzi dalej"],
        ["Nagrania pobrane i przetworzone", f"{VIDEOS_ATTEMPTED:,}".replace(",", " "), "—"],
        [
            "Nagrania, z których cokolwiek powstało",
            f"{counts.videos_with_output:,}".replace(",", " "),
            f"{counts.videos_with_output / VIDEOS_ATTEMPTED:.1%}",
        ],
        ["Klatki wybrane przez modele", f"{RAW_FRAMES:,}".replace(",", " "), "—"],
        ["Kandydaci na klatki szczytowe", f"{RAW_PEAK_CANDIDATES:,}".replace(",", " "), "—"],
        [
            "Pary po bramce jakości",
            f"{counts.pairs_total:,}".replace(",", " "),
            f"{counts.pairs_total / RAW_PEAK_CANDIDATES:.1%}",
        ],
        [
            "Pary ocenione ręcznie przez zespół",
            f"{counts.human_pairs:,}".replace(",", " "),
            f"{counts.human_pairs / counts.pairs_total:.1%}",
        ],
    ]
    return [
        Paragraph("2. Skala przetwarzania", st["h2"]),
        Paragraph(
            f"Przetworzyliśmy <b>{VIDEOS_ATTEMPTED} nagrań</b> o łącznej objętości około "
            f"{SOURCE_VIDEO_GB} GB. Każde przeszło pełny potok: detekcja psa, śledzenie, "
            f"kadrowanie mordy, {46} punktów kluczowych, rasa, jednostki ruchu i emocja. "
            f"Poniższa tabela pokazuje, ile materiału przeżywa kolejne etapy — i to jest "
            f"najważniejsza liczba w całym raporcie, bo tłumaczy rozmiar zbioru.",
            st["body"],
        ),
        _table(rows, [78 * mm, 30 * mm, 34 * mm], highlight=6),
        Paragraph(
            f"Z {VIDEOS_ATTEMPTED} nagrań aż <b>{bez_wyniku}</b> "
            f"({bez_wyniku / VIDEOS_ATTEMPTED:.1%}) nie dało ani jednej pary nadającej się "
            f"do oceny. Nie znaczy to, że przetwarzanie się nie udało — znaczy, że na tych "
            f"nagraniach nie ma ujęcia psiej mordy dość dużego i dość spokojnego, żeby "
            f"zmierzyć na nim mimikę.",
            st["caption"],
        ),
    ]


def _why_section(st: dict) -> list:
    """
    Buduje sekcję o powodach odsiewu materiału.

    Args:
        st: Style akapitów

    Returns:
        Elementy dokumentu
    """
    total = sum(count for _, count in REJECTION_REASONS)
    rows = [["Powód odrzucenia", "Klatek", "Udział"]]
    rows += [
        [reason, f"{count:,}".replace(",", " "), f"{count / total:.1%}"]
        for reason, count in REJECTION_REASONS
    ]
    return [
        Paragraph("3. Dlaczego odpadła połowa materiału", st["h2"]),
        Paragraph(
            "Bramka jakości odrzuca klatkę, na której pomiar mimiki byłby zmyśleniem. "
            "Rozkład powodów nie zostawia wątpliwości, gdzie leży ograniczenie:",
            st["body"],
        ),
        _table(rows, [96 * mm, 24 * mm, 22 * mm], highlight=1),
        Paragraph(
            "Liczby dotyczą pojedynczych klatek, a para odpada, gdy zawiedzie choćby jedna "
            "z dwóch — dlatego suma powodów przewyższa liczbę odrzuconych par. Część "
            "materiału odrzuconego automatycznie wróciła później do zbioru po ręcznym "
            "przejrzeniu przez zespół.",
            st["caption"],
        ),
        Paragraph(
            "<b>Trzy czwarte odrzuceń to jeden powód: morda w kadrze jest za mała.</b> Jednostki ruchu "
            "DogFACS to przesunięcia rzędu kilku pikseli — zmarszczenie nosa, uniesienie "
            "wargi, obrót ucha. Na mordzie o szerokości 40 pikseli taki ruch tonie w "
            "niepewności samego detektora punktów. To ograniczenie materiału, nie potoku: "
            "dostępne nagrania psów to w większości ujęcia całej sylwetki, plany ogólne i "
            "materiał z mediów społecznościowych kręcony z ręki, gdzie pysk zajmuje ułamek "
            "kadru i rzadko jest zwrócony do obiektywu.",
            st["body"],
        ),
    ]


def _limits_section(counts: Counts, st: dict) -> list:
    """
    Buduje sekcję o ograniczeniach przedsięwzięcia.

    Args:
        counts: Policzone liczby
        st: Style akapitów

    Returns:
        Elementy dokumentu
    """
    return [
        Paragraph("6. Ograniczenia, które ukształtowały wynik", st["h2"]),
        Paragraph(
            "<b>Materiał.</b> Nie istnieje gotowy zbiór nagrań psów nadający się do kodowania "
            "DogFACS. Zbieraliśmy go sami z materiału stockowego, serwisów wideo i archiwum "
            "udostępnionego na dysku współdzielonym. Zmierzona wydajność takiego materiału "
            f"jest niska: z {VIDEOS_ATTEMPTED} nagrań użyteczne okazało się co drugie, a "
            "bramka jakości odrzuca połowę tego, co z nich zostało. Zwiększenie zbioru "
            "wymaga nie tyle "
            "więcej nagrań, ile nagrań INNEGO rodzaju — zbliżeń psiej mordy zwróconej do "
            "obiektywu. Takich w otwartych źródłach jest niewiele.",
            st["body"],
        ),
        Paragraph(
            "<b>Moc obliczeniowa.</b> Pracowaliśmy na komputerach osobistych, bez dostępu do "
            "klastra ani do kart graficznych klasy serwerowej. Potok uruchamia na każde "
            "nagranie pięć sieci neuronowych po kolei, a każdy proces roboczy zajmuje około "
            "1,3 GB pamięci na same wagi modeli, do czego dochodzi bufor pełnych klatek. "
            "Ogranicza nas pamięć, nie procesor: liczba równoległych procesów, którą maszyna "
            "utrzyma, wyznacza tempo całego przetwarzania. Pobieranie materiału ma własne "
            "ograniczenie — anonimowy dostęp do dysku współdzielonego przyjmuje kilkadziesiąt "
            "plików dziennie, więc sam transfer materiału był rozłożony na wiele podejść.",
            st["body"],
        ),
        Paragraph(
            "<b>Powtarzalność etykiety.</b> Najpoważniejsze ograniczenie nie jest ani "
            "sprzętowe, ani materiałowe. Na parach ocenionych niezależnie przez dwie osoby "
            "zgodność co do tego, które AU są aktywne, wynosi 7.4%, a współczynnik kappa "
            "Cohena 0.132 — niewiele powyżej losu. Zgodność co do tego, czy para w ogóle "
            "nadaje się do oceny, jest wysoka (90.2%), podobnie jak co do emocji (62.9%); "
            "rozjeżdżają się wyłącznie jednostki ruchu. Dopóki ludzie nie kodują ich tak "
            "samo, żaden model nie może zostać rzetelnie zmierzony — bo nie ma stabilnego "
            "punktu odniesienia.",
            st["body"],
        ),
    ]


def build_story(counts: Counts, st: dict) -> list:
    """
    Buduje treść dokumentu.

    Args:
        counts: Policzone liczby
        st: Style akapitów

    Returns:
        Lista elementów do złożenia
    """
    story: list = [
        Paragraph("Dog FACS Dataset — raport z realizacji", st["title"]),
        Paragraph(
            "Automatyczna anotacja mimiki psów w formacie COCO<br/>"
            "Politechnika Gdańska, Wydział ETI — projekt grupowy, semestr 1",
            st["subtitle"],
        ),
        Paragraph("1. Co powstało", st["h2"]),
        Paragraph(
            "Celem projektu było zbudowanie zbioru danych opisującego mimikę psów według "
            "systemu DogFACS: 46 punktów kluczowych twarzy, 21 jednostek ruchu (AU) i 9 klas "
            "emocji, w formacie COCO. Powstał kompletny potok — od pobrania nagrania do "
            "gotowej anotacji — oraz stanowisko do ręcznej weryfikacji wyników przez "
            "człowieka. Oddajemy dwa artefakty:",
            st["body"],
        ),
        _table(
            [
                ["Artefakt", "Pary", "Klatki", "Etykieta AU"],
                [
                    "Zbiór zweryfikowany przez człowieka",
                    f"{counts.human_pairs}",
                    f"{counts.human_frames}",
                    "ocena człowieka",
                ],
                [
                    "Zbiór pełny",
                    f"{counts.pairs_total:,}".replace(",", " "),
                    f"{counts.frames_total:,}".replace(",", " "),
                    "model, precyzja 32.7%",
                ],
            ],
            [70 * mm, 24 * mm, 24 * mm, 40 * mm],
        ),
        Paragraph(
            "Każda próbka to PARA klatek: spoczynkowa i szczytowa. Jednostki ruchu są z "
            "definicji różnicą względem spoczynku, więc pojedyncza klatka nie niesie "
            f"informacji o AU. Zbiór obejmuje {counts.breeds} ras rozpoznanych przez "
            f"klasyfikator. Zespół obejrzał i ocenił {HUMAN_REVIEWED} par, z czego "
            f"{HUMAN_REJECTED} odrzucił jako nienadające się do kodowania mimiki; "
            f"{HUMAN_DISPUTED} par oceniono rozbieżnie i są oznaczone jako sporne.",
            st["caption"],
        ),
    ]
    story += [
        Paragraph(
            "Rozkład emocji w podzbiorze zweryfikowanym przez człowieka (ocena anotatora, "
            "nie etykieta z nazwy katalogu):",
            st["body"],
        ),
        _table(
            [["Emocja", "Par"]] + [[nazwa, str(ile)] for nazwa, ile in counts.emotions],
            [70 * mm, 24 * mm],
        ),
        Paragraph(
            "Przewaga stanów spokojnych jest cechą materiału źródłowego: nagrania psów "
            "w otwartych źródłach to w większości sceny spokojne. Stany silne (ból, strach, "
            "agresja) są rzadkie i pozostają najsłabiej reprezentowane.",
            st["caption"],
        ),
    ]
    story += _funnel_section(counts, st)
    story += _why_section(st)
    story.append(PageBreak())

    story += [
        Paragraph("4. Jakość poszczególnych modeli", st["h2"]),
        Paragraph(
            "Potok składa się z pięciu modeli uruchamianych po kolei. Ich jakość jest bardzo "
            "różna i to rozwarstwienie jest istotnym wynikiem samym w sobie:",
            st["body"],
        ),
        _table(
            [["Zadanie", "Architektura", "Jakość", "Uwagi"]] + [list(r) for r in MODEL_QUALITY],
            [34 * mm, 38 * mm, 38 * mm, 42 * mm],
        ),
        Paragraph(
            "Zadania rozpoznawania obiektu — gdzie jest pies, gdzie morda, jaka rasa — są "
            "rozwiązane dobrze. Trudność zaczyna się przy mimice.",
            st["caption"],
        ),
        Paragraph("5. Jednostki ruchu — najtrudniejsza część", st["h2"]),
        Paragraph(
            "Wyjściowo AU wyznaczaliśmy regułami geometrycznymi: każda jednostka to jedna "
            "odległość między punktami, porównana ze wspólnym progiem. Zmierzenie tych reguł "
            "wobec ocen człowieka pokazało, że dają precyzję 5% — czyli dziewiętnaście "
            "sygnałów na dwadzieścia jest fałszywych. Zastąpiliśmy je modelem uczonym z "
            "przesunięcia wszystkich 46 punktów naraz. Wszystkie wyniki poniżej pochodzą ze "
            "sprawdzianu krzyżowego z podziałem po nagraniach, w którym próg decyzyjny "
            "dobierany jest wyłącznie na części uczącej:",
            st["body"],
        ),
        _table(
            [["Metoda", "Precyzja", "Pokrycie", "F1"]] + [list(r) for r in AU_RESULTS],
            [78 * mm, 22 * mm, 22 * mm, 20 * mm],
            highlight=5,
        ),
        Paragraph(
            "Precyzja wzrosła sześciokrotnie, z 5.0% na 32.7%. Warto zaznaczyć, skąd wziął "
            "się ostatni krok: nie z większej sieci, lecz z ODRZUCENIA części etykiet. "
            "Członkowie zespołu stosowali różne progi tego, co uznać za aktywację — udział "
            "komórek oznaczonych jako aktywne wahał się od 0.17% do 5.4%. Uczenie na 273 "
            "parach o spójnym standardzie wypada lepiej niż na wszystkich dostępnych.",
            st["caption"],
        ),
        Paragraph(
            "Jakość mocno zależy od jednostki. Najlepiej mierzalne są ruchy pyska, bo są "
            "duże i wyraźne geometrycznie; ruchy uszu i górnej części twarzy pozostają poza "
            "zasięgiem obecnych punktów kluczowych:",
            st["body"],
        ),
        _table(
            [["Jednostka ruchu", "Potwierdzeń", "Precyzja", "Pokrycie"]]
            + [list(r) for r in AU_PER_UNIT],
            [58 * mm, 32 * mm, 26 * mm, 26 * mm],
        ),
        Paragraph(
            "Cztery jednostki (AD33, AD35, EAD101, EAD102) nie zostały potwierdzone przez "
            "człowieka ani razu — nie da się ich zatem ani nauczyć, ani ocenić.",
            st["caption"],
        ),
    ]

    story.append(PageBreak())
    story += _limits_section(counts, st)
    story += [
        Paragraph("7. Wnioski", st["h2"]),
        Paragraph(
            "<b>Potok działa i zbiór istnieje.</b> Powstało "
            f"{counts.pairs_total:,}".replace(",", " ")
            + " par klatek z pełną anotacją geometryczną, w tym "
            f"{counts.human_pairs} zweryfikowanych przez człowieka. Rozpoznawanie psa, mordy "
            "i rasy działa na poziomie użytkowym.",
            st["body"],
        ),
        Paragraph(
            "<b>Wąskim gardłem AU okazały się reguły, a nie punkty kluczowe.</b> Ten sam "
            "materiał, z którego reguły wyciągały 5% precyzji, po zastosowaniu modelu "
            "uczonego daje 33%. Informacja była w danych przez cały czas — poprzednia metoda "
            "jej nie wykorzystywała. To najważniejszy wniosek techniczny projektu.",
            st["body"],
        ),
        Paragraph(
            "<b>Dalszy postęp wymaga lepszej etykiety, nie lepszego modelu.</b> Przy "
            "zgodności między anotatorami na poziomie 7.4% model o precyzji 33% zbliża się "
            "do granicy powtarzalności samego zjawiska. Kolejnym krokiem jest pisemna "
            "instrukcja kodowania dla dziewięciu mierzalnych jednostek i sesja kalibracyjna "
            "zespołu na wspólnym podzbiorze par — dopiero potem ma sens rozbudowa modelu.",
            st["body"],
        ),
        Paragraph(
            "<b>Uczciwość etykiet.</b> W zbiorze rozróżniamy trzy źródła oceny: człowiek, "
            "model i reguły; mówi o tym pole label_source. Etykieta automatyczna jest "
            "wyraźnie oznaczona jako słaba — przy precyzji 33% dwie aktywacje na trzy są "
            "błędne. Nie podajemy jej jako prawdy, bo zbiór z ukrytym błędem tej skali "
            "byłby gorszy niż brak zbioru.",
            st["body"],
        ),
        Spacer(1, 5 * mm),
        Paragraph(
            "Wszystkie liczby w raporcie pochodzą z pomiarów na repozytorium projektu i dają "
            "się odtworzyć poleceniami opisanymi w dokumentacji technicznej "
            "(eval_au_rules, train_au_model, build_full_dataset).",
            st["note"],
        ),
    ]
    return [KeepTogether(item) if isinstance(item, Table) else item for item in story]


def main() -> None:
    """Składa raport i zapisuje go do PDF."""
    register_fonts()
    counts = collect_counts()
    output = DEFAULT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)

    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="Dog FACS Dataset — raport z realizacji",
        author="Politechnika Gdańska WETI",
    )
    document.build(build_story(counts, _styles()))

    logger.info("Par w zbiorze      : %d", counts.pairs_total)
    logger.info("Par od czlowieka   : %d", counts.human_pairs)
    logger.info("Zapisano: %s (%.0f kB)", output, output.stat().st_size / 1024)


if __name__ == "__main__":
    main()
