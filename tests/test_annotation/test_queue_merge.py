"""
Dokładanie par do kolejki nie może gubić tych, które już w niej były.

Kolejka jest publikowana anotatorom, a werdykty przypinają się do ŚCIEŻKI
KLATKI. Para, która z kolejki zniknie, zabiera ze sobą cudzą pracę — i robi to
po cichu, bo dziennik w `data/labels/` przeżywa nietknięty, tylko przestaje się
z czymkolwiek wiązać. Zmierzone 27.08.2026: przebudowa od zera wyrzuciła 1533
kadry i 603 z 605 werdyktów dwóch osób.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.annotation.queue_merge import (  # noqa: E402
    merge_queues,
    renumber_queue,
)


def _kolejka(pary: list[tuple[str, str]], pierwsze_id: int = 1) -> dict:
    """
    Buduje kolejkę z listy par (klatka neutralna, klatka szczytowa).

    Args:
        pary: Ścieżki klatek tworzących pary
        pierwsze_id: Od jakiego identyfikatora zacząć numerowanie

    Returns:
        Zbiór w formacie COCO
    """
    images: list[dict] = []
    annotations: list[dict] = []
    identyfikator = pierwsze_id
    numer_anotacji = pierwsze_id
    for neutralna, szczytowa in pary:
        id_neutralnej, id_szczytowej = identyfikator, identyfikator + 1
        identyfikator += 2
        images.append({"id": id_neutralnej, "file_name": neutralna})
        images.append({"id": id_szczytowej, "file_name": szczytowa})
        # DWA wiersze na pare — tak jak w prawdziwej kolejce. Wiersz neutralny
        # wskazuje `neutral_frame_id` na samego siebie: baza AU klatki
        # neutralnej to ona sama.
        annotations.append(
            {
                "id": numer_anotacji,
                "image_id": id_neutralnej,
                "neutral_frame_id": id_neutralnej,
                "frame_role": "neutral",
                "znacznik": neutralna,
            }
        )
        annotations.append(
            {
                "id": numer_anotacji + 1,
                "image_id": id_szczytowej,
                "neutral_frame_id": id_neutralnej,
                "frame_role": "peak",
                "znacznik": szczytowa,
            }
        )
        numer_anotacji += 2
    return {"images": images, "annotations": annotations, "categories": [{"id": 1}]}


class TestDokladanieDoKolejki:
    """Reguła: para raz wydana anotatorom zostaje w kolejce."""

    def test_pary_z_bazy_przezywaja_dolozenie(self) -> None:
        baza = _kolejka([("psA/neutral.jpg", "psA/peak.jpg")])
        nowe = _kolejka([("psB/neutral.jpg", "psB/peak.jpg")], pierwsze_id=1)

        wynik = merge_queues(baza, nowe)

        nazwy = {obraz["file_name"] for obraz in wynik["images"]}
        assert "psA/peak.jpg" in nazwy, "para z bazy nie może zniknąć"
        assert "psB/peak.jpg" in nazwy, "nowa para ma dojść"
        assert len(wynik["annotations"]) == 4

    def test_identyfikatory_nie_zderzaja_sie(self) -> None:
        """Obie kolejki liczą id od 1 — dołożone muszą dostać własne."""
        baza = _kolejka([("psA/neutral.jpg", "psA/peak.jpg")])
        nowe = _kolejka([("psB/neutral.jpg", "psB/peak.jpg")], pierwsze_id=1)

        wynik = merge_queues(baza, nowe)

        identyfikatory = [obraz["id"] for obraz in wynik["images"]]
        assert len(identyfikatory) == len(set(identyfikatory)), "id muszą być unikalne"

    def test_klatka_neutralna_zostaje_przy_swoim_psie(self) -> None:
        """Najdroższa cicha usterka: peak związany z bazą AU innego psa."""
        baza = _kolejka([("psA/neutral.jpg", "psA/peak.jpg")])
        nowe = _kolejka([("psB/neutral.jpg", "psB/peak.jpg")], pierwsze_id=1)

        wynik = merge_queues(baza, nowe)

        obrazy = {obraz["id"]: obraz["file_name"] for obraz in wynik["images"]}
        for anotacja in wynik["annotations"]:
            pies_szczytowej = obrazy[anotacja["image_id"]].split("/")[0]
            pies_neutralnej = obrazy[anotacja["neutral_frame_id"]].split("/")[0]
            assert pies_szczytowej == pies_neutralnej, "peak wskazuje cudzą klatkę neutralną"

    def test_para_obecna_w_bazie_nie_dubluje_sie(self) -> None:
        baza = _kolejka([("psA/neutral.jpg", "psA/peak.jpg")])
        nowe = _kolejka([("psA/neutral.jpg", "psA/peak.jpg")], pierwsze_id=100)

        wynik = merge_queues(baza, nowe)

        assert len(wynik["images"]) == 2, "ta sama para nie może wejść dwa razy"
        assert len(wynik["annotations"]) == 2

    def test_limit_ogranicza_tylko_dokladane(self) -> None:
        baza = _kolejka([("psA/neutral.jpg", "psA/peak.jpg")])
        nowe = _kolejka(
            [
                ("psB/neutral.jpg", "psB/peak.jpg"),
                ("psC/neutral.jpg", "psC/peak.jpg"),
            ],
            pierwsze_id=1,
        )

        wynik = merge_queues(baza, nowe, limit=1)

        nazwy = {obraz["file_name"] for obraz in wynik["images"]}
        assert "psA/peak.jpg" in nazwy, "limit nie może dotykać bazy"
        assert len(wynik["annotations"]) == 4, "baza plus jedna dołożona, po dwa wiersze"

    def test_dolozona_para_ma_OBA_wiersze(self) -> None:
        """
        Kolejka trzyma wiersz neutralny i szczytowy. Wcześniejsza wersja brała
        wiersz neutralny za osobną parę: zajmował nazwę kadru, prawdziwa para
        odpadała jako „już jest" i do kolejki wchodziły osierocone klatki
        neutralne bez swoich szczytowych. Zmierzone: 542 takie kadry.
        """
        baza = _kolejka([("psA/neutral.jpg", "psA/peak.jpg")])
        nowe = _kolejka([("psB/neutral.jpg", "psB/peak.jpg")], pierwsze_id=1)

        wynik = merge_queues(baza, nowe)

        role = [a.get("frame_role") for a in wynik["annotations"]]
        assert role.count("peak") == 2, "każda para musi mieć klatkę szczytową"
        assert role.count("neutral") == 2, "każda para musi mieć klatkę neutralną"

        nazwy = {o["file_name"] for o in wynik["images"]}
        assert {"psB/neutral.jpg", "psB/peak.jpg"} <= nazwy, "dołożona para w całości"

    def test_osierocony_wiersz_neutralny_nie_wchodzi_sam(self) -> None:
        """Sam wiersz neutralny, bez swojej klatki szczytowej, to nie para."""
        baza = _kolejka([("psA/neutral.jpg", "psA/peak.jpg")])
        sierota = {
            "images": [{"id": 1, "file_name": "psC/neutral.jpg"}],
            "annotations": [
                {"id": 1, "image_id": 1, "neutral_frame_id": 1, "frame_role": "neutral"}
            ],
            "categories": [{"id": 1}],
        }

        wynik = merge_queues(baza, sierota)

        nazwy = {o["file_name"] for o in wynik["images"]}
        assert "psC/neutral.jpg" not in nazwy, "klatka bez pary nie ma czego robić w kolejce"

    def test_drugi_szczyt_tego_samego_psa_wchodzi(self) -> None:
        """
        Klatka neutralna jest WSPÓLNA dla wszystkich szczytów jednego treku.

        Odrzucanie pary, gdy w kolejce jest już którakolwiek z jej klatek,
        gubiło drugi i każdy następny szczyt tego samego psa — a każdy opisuje
        inną chwilę mimiki. Zmierzone: z 3230 par weszło 231.
        """
        baza = _kolejka([("psA/neutral.jpg", "psA/peak1.jpg")])
        # ten sam pies, ta sama klatka neutralna, INNY szczyt
        nowe = _kolejka([("psA/neutral.jpg", "psA/peak2.jpg")], pierwsze_id=50)

        wynik = merge_queues(baza, nowe)

        nazwy = {o["file_name"] for o in wynik["images"]}
        assert "psA/peak2.jpg" in nazwy, "drugi szczyt tego psa musi wejść"
        assert sorted(nazwy).count("psA/neutral.jpg") == 1, "klatka neutralna bez duplikatu"
        assert len([o for o in wynik["images"] if o["file_name"] == "psA/neutral.jpg"]) == 1

    def test_wspoldzielona_neutralna_wskazuje_ten_sam_obraz(self) -> None:
        """Oba szczyty muszą pokazywać na TĘ SAMĄ, jedyną klatkę neutralną."""
        baza = _kolejka([("psA/neutral.jpg", "psA/peak1.jpg")])
        nowe = _kolejka([("psA/neutral.jpg", "psA/peak2.jpg")], pierwsze_id=50)

        wynik = merge_queues(baza, nowe)

        obrazy = {o["id"]: o["file_name"] for o in wynik["images"]}
        szczyty = [a for a in wynik["annotations"]
                   if a.get("neutral_frame_id") not in (None, a["image_id"])]
        assert len(szczyty) == 2, "dwie pary"
        wskazania = {a["neutral_frame_id"] for a in szczyty}
        assert len(wskazania) == 1, "obie pary wskazują jedną klatkę neutralną"
        assert obrazy[wskazania.pop()] == "psA/neutral.jpg"

    def test_numery_porzadkowe_nie_zderzaja_sie(self) -> None:
        """
        Stanowisko skleja parę po `review_order` (`coco_import._group_pairs`).

        Obie kolejki numerują od zera, więc przepisany numer zderza się
        z numerem pary już obecnej — wtedy wpis zostaje NADPISANY i szczyt
        jednego psa dostaje klatkę neutralną drugiego. Zmierzone: 1840 zderzeń.
        """
        baza = _kolejka([("psA/neutral.jpg", "psA/peak.jpg")])
        for a in baza["annotations"]:
            a["review_order"] = 0
        nowe = _kolejka([("psB/neutral.jpg", "psB/peak.jpg")], pierwsze_id=1)
        for a in nowe["annotations"]:
            a["review_order"] = 0

        wynik = merge_queues(baza, nowe)

        numery = [a["review_order"] for a in wynik["annotations"]]
        assert len(set(numery)) == 2, "każda para ma własny numer porządkowy"

    def test_kazda_para_ma_wlasny_wiersz_neutralny(self) -> None:
        """
        Wiersz neutralny nie może być współdzielony, choć OBRAZ może.

        Jeden wiersz obsługuje dokładnie jeden `review_order`, więc przy
        współdzieleniu wszystkie szczyty poza jednym zostają bez klatki
        neutralnej i para po cichu znika z kolejki.
        """
        baza = _kolejka([("psA/neutral.jpg", "psA/peak1.jpg")])
        for i, a in enumerate(baza["annotations"]):
            a["review_order"] = 0
        nowe = _kolejka([("psA/neutral.jpg", "psA/peak2.jpg")], pierwsze_id=50)
        for a in nowe["annotations"]:
            a["review_order"] = 0

        wynik = merge_queues(baza, nowe)

        # tak samo jak stanowisko: grupujemy po numerze i roli
        po_numerze: dict[int, dict[str, dict]] = {}
        for a in wynik["annotations"]:
            po_numerze.setdefault(a["review_order"], {})[a["frame_role"]] = a
        kompletne = [n for n, e in po_numerze.items() if "neutral" in e and "peak" in e]
        assert len(kompletne) == 2, "obie pary muszą się złożyć"

        obrazy = {o["id"]: o["file_name"] for o in wynik["images"]}
        assert len([o for o in wynik["images"] if o["file_name"] == "psA/neutral.jpg"]) == 1, \
            "obraz klatki neutralnej bez duplikatu"
        for numer in kompletne:
            e = po_numerze[numer]
            assert obrazy[e["neutral"]["image_id"]] == "psA/neutral.jpg"


class TestNaprawaNumeracji:
    """Kolejka ze zderzonymi numerami daje się naprawić bez utraty par."""

    def _zepsuta(self) -> dict:
        """Dwie pary różnych psów z TYM SAMYM numerem porządkowym."""
        kolejka = _kolejka([("psA/n.jpg", "psA/p.jpg"), ("psB/n.jpg", "psB/p.jpg")])
        for a in kolejka["annotations"]:
            a["review_order"] = 0
        return kolejka

    def test_stanowisko_gubi_pary_przed_naprawa(self) -> None:
        """Dowód, że problem jest realny: grupowanie po numerze zjada parę."""
        zepsuta = self._zepsuta()
        po_numerze: dict = {}
        for a in zepsuta["annotations"]:
            po_numerze.setdefault(a["review_order"], {})[a["frame_role"]] = a
        assert len(po_numerze) == 1, "obie pary siedzą pod jednym numerem"

    def test_po_naprawie_kazda_para_ma_swoj_numer(self) -> None:
        wynik = renumber_queue(self._zepsuta())

        po_numerze: dict = {}
        for a in wynik["annotations"]:
            po_numerze.setdefault(a["review_order"], {})[a["frame_role"]] = a
        kompletne = [e for e in po_numerze.values() if "neutral" in e and "peak" in e]
        assert len(kompletne) == 2, "obie pary muszą się złożyć"

    def test_naprawa_nie_miesza_psow(self) -> None:
        wynik = renumber_queue(self._zepsuta())

        obrazy = {o["id"]: o["file_name"] for o in wynik["images"]}
        po_numerze: dict = {}
        for a in wynik["annotations"]:
            po_numerze.setdefault(a["review_order"], {})[a["frame_role"]] = a
        for wpis in po_numerze.values():
            pies_n = obrazy[wpis["neutral"]["image_id"]].split("/")[0]
            pies_p = obrazy[wpis["peak"]["image_id"]].split("/")[0]
            assert pies_n == pies_p, "szczyt dostał klatkę neutralną innego psa"

    def test_naprawa_nie_rusza_obrazow(self) -> None:
        """Ścieżki obrazów to klucze werdyktów — muszą przeżyć nietknięte."""
        zepsuta = self._zepsuta()
        przed = [o["file_name"] for o in zepsuta["images"]]

        wynik = renumber_queue(zepsuta)

        assert [o["file_name"] for o in wynik["images"]] == przed
