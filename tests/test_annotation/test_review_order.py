"""Testy kolejności par podawanych anotatorowi."""

from packages.pipeline.quality_gate import FrameQuality
from scripts.annotation.curate_for_review import ReviewPair, order_for_review


def _quality() -> FrameQuality:
    return FrameQuality(
        asymmetry=0.1,
        weak_ratio=0.0,
        face_width=80.0,
        is_usable=True,
        reasons=(),
    )


def _pair(video: str, signal: int, ambiguity: int = 0) -> ReviewPair:
    return ReviewPair(
        peak={"id": f"{video}-{signal}-{ambiguity}"},
        neutral={"id": f"{video}-neutral"},
        peak_name=f"{video}/{signal}-{ambiguity}.jpg",
        video=video,
        signal=signal,
        ambiguity=ambiguity,
        peak_quality=_quality(),
        neutral_quality=_quality(),
    )


def test_pary_jednego_nagrania_ida_obok_siebie() -> None:
    """Anotator kończy nagranie, zanim zobaczy następne."""
    pairs = [
        _pair("A", 1),
        _pair("B", 5),
        _pair("A", 3),
        _pair("B", 2),
        _pair("A", 2),
    ]

    videos = [pair.video for pair in order_for_review(pairs)]

    # Każde nagranie tworzy jeden ciągły blok — liczba przejść między
    # nagraniami równa się liczbie nagrań minus jeden.
    switches = sum(1 for before, after in zip(videos, videos[1:]) if before != after)
    assert switches == 1
    assert videos == ["B", "B", "A", "A", "A"]


def test_nagrania_od_najmocniejszego_sygnalu() -> None:
    """Nagranie z jedną mocną parą wyprzedza nagranie z wieloma pustymi."""
    pairs = [
        _pair("cichy", 0),
        _pair("cichy", 0),
        _pair("cichy", 0),
        _pair("glosny", 7),
    ]

    assert order_for_review(pairs)[0].video == "glosny"


def test_w_obrebie_nagrania_sygnal_przed_cisza() -> None:
    """Wewnątrz nagrania mocniejsze pary idą pierwsze."""
    pairs = [_pair("A", 0), _pair("A", 4), _pair("A", 2)]

    assert [pair.signal for pair in order_for_review(pairs)] == [4, 2, 0]


def test_zadna_para_nie_ginie() -> None:
    """Przestawianie kolejności nie może gubić ani dublować par."""
    pairs = [_pair("A", 1), _pair("A", 2), _pair("B", 1), _pair("C", 9)]

    ordered = order_for_review(pairs)

    assert len(ordered) == len(pairs)
    assert {id(pair) for pair in ordered} == {id(pair) for pair in pairs}
