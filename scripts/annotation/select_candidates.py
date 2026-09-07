#!/usr/bin/env python3
"""
Strona do RECZNEGO wyboru kadrow odsianych przez bramke jakosci.

Po co. Bramka odrzuca 63% kandydatow, ale obejrzane probki pokazuja, ze czesc
z nich jest w pelni czytelna. Zamiast przesuwac prog na slepo (i wpuszczac
razem z dobrymi cala mase slabych), pokazujemy czlowiekowi WSZYSTKIE odrzucone
kadry z narysowanymi punktami i pozwalamy wybrac te, ktore sie nadaja.

Punkty rysujemy w dwoch kolorach — pewne zielone, niepewne czerwone — bo to
wlasnie one decyduja o odrzuceniu i czlowiek musi widziec, co model zrobil.

Zaznaczenie PRZEZYWA zamkniecie karty. Pierwsza wersja trzymala je wylacznie
w pamieci JS i trzysta klikniec przepadlo przy odswiezeniu — teraz kazde
klikniecie zapisuje sie w `localStorage`, a co `AUTOZAPIS_CO` zmian leci tez
na serwer do `postep.json`. Czlowiek przegladajacy 3700 kadrow robi to na raty
i nie ma prawa stracic pracy przez przypadkowe zamkniecie okna.

Wybor jest ODWROCONY: klikniecie ODRZUCA kadr, a do kolejki idzie cala reszta.
Tak jest mniej klikania, bo dobrych kadrow jest wiecej niz zlych — czlowiek
przeglada wszystko i zaznacza wyjatki, zamiast potwierdzac kazdy dobry kadr.

Klatka neutralna NIE jest wybierana osobno: para powstaje z klatki szczytowej,
a jej baza AU jedzie razem z nia. Zapisujemy obie sciezki, zeby kuracja mogla
odtworzyc pare bez zgadywania.

Uzycie:
    python -m scripts.annotation.select_candidates
    (potem otworz http://127.0.0.1:8010)
"""

import argparse
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

DEFAULT_DIR: str = "data/kandydaci"
DEFAULT_PORT: int = 8010
SELECTION_NAME: str = "wybrane.json"
PROGRESS_NAME: str = "postep.json"

STRONA = """<!doctype html>
<meta charset="utf-8"><title>Wybor kadrow</title>
<style>
 body{font:14px/1.5 system-ui,sans-serif;margin:0;background:#f6f7f4;color:#1b211c}
 header{position:sticky;top:0;background:#fff;border-bottom:1px solid #d5dacd;
        padding:12px 20px;display:flex;gap:16px;align-items:center;z-index:10}
 h1{font-size:16px;margin:0;font-weight:600}
 .licznik{font-variant-numeric:tabular-nums;color:#5f6a60}
 button{font:inherit;padding:8px 16px;border-radius:6px;border:1px solid #3c6e52;
        background:#3c6e52;color:#fff;cursor:pointer}
 button.szary{background:#fff;color:#1b211c;border-color:#d5dacd}
 button:disabled{opacity:.45;cursor:default}
 #siatka{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));
         gap:12px;padding:16px}
 figure{margin:0;position:relative;cursor:pointer;border:3px solid transparent;
        border-radius:6px;overflow:hidden;background:#fff}
 figure.odrzucony{border-color:#a4472c;opacity:.35}
 figure.odrzucony img{filter:grayscale(1)}
 figure img{display:block;width:100%;aspect-ratio:1;object-fit:cover}
 figcaption{font:13px/1.4 ui-monospace,monospace;padding:6px 8px;color:#5f6a60;
            display:flex;justify-content:space-between}
 figure.odrzucony figcaption{color:#a4472c;font-weight:600}
 #stan{padding:0 20px 20px;color:#5f6a60}
</style>
<header>
  <h1>Kadry odsiane przez bramke — klikaj te ZLE</h1>
  <span class="licznik" id="licznik">wczytuje...</span>
  <button class="szary" id="zaden">Cofnij odrzucenia</button>
  <button id="zapisz">Dodaj pozostale do kolejki</button>
</header>
<div id="siatka"></div>
<p id="stan"></p>
<script>
// Zaznaczenie musi przezyc zamkniecie karty — inaczej setki klikniec przepadaja
// przy jednym odswiezeniu. Klucz wiaze sie z lista kandydatow, wiec po jej
// przeliczeniu stare zaznaczenie samo sie nie doklei do innych kadrow.
const KLUCZ = 'dogfacs.odrzucone.v2';
const AUTOZAPIS_CO = 10;

const siatka = document.getElementById('siatka');
const licznik = document.getElementById('licznik');
const zapisz = document.getElementById('zapisz');
const stan = document.getElementById('stan');
let dane = [], odrzucone = new Set(), odKopii = 0;

function wczytajZapamietane() {
  try { return new Set(JSON.parse(localStorage.getItem(KLUCZ) || '[]')); }
  catch (e) { return new Set(); }
}
function zapamietaj() {
  try { localStorage.setItem(KLUCZ, JSON.stringify([...odrzucone])); }
  catch (e) { stan.textContent = 'Uwaga: przegladarka nie zapisuje stanu lokalnie.'; }
  if (++odKopii >= AUTOZAPIS_CO) { odKopii = 0; wyslij('/postep', false); }
}

function odswiez() {
  const zostaje = dane.length - odrzucone.size;
  licznik.textContent = `do kolejki ${zostaje}, odrzucone ${odrzucone.size} z ${dane.length}`;
  zapisz.disabled = zostaje === 0;
}

function wyslij(adres, koncowy) {
  const pary = dane
    .filter(k => !odrzucone.has(k.peak))
    .map(k => ({peak: k.peak, neutral: k.neutral}));
  return fetch(adres, {method: 'POST', body: JSON.stringify(pary)})
    .then(r => r.json())
    .then(o => {
      stan.textContent = koncowy
        ? `Zapisano ${o.zapisane} par do kolejki (odrzuconych ${odrzucone.size}). Mozesz zamknac strone.`
        : `Kopia robocza zapisana (${o.zapisane} par, odrzuconych ${odrzucone.size}).`;
    })
    .catch(e => { stan.textContent = 'Blad zapisu: ' + e; });
}

fetch('candidates.json').then(r => r.json()).then(lista => {
  dane = lista;
  odrzucone = wczytajZapamietane();
  siatka.innerHTML = '';
  lista.forEach(k => {
    const f = document.createElement('figure');
    f.dataset.peak = k.peak;
    if (odrzucone.has(k.peak)) f.classList.add('odrzucony');
    f.innerHTML = `<img loading="lazy" src="thumbs/${k.thumb}" alt="${k.peak}">
      <figcaption><span>${k.face_px}px</span><span>w${k.weak}</span></figcaption>`;
    f.onclick = () => {
      if (odrzucone.has(k.peak)) { odrzucone.delete(k.peak); f.classList.remove('odrzucony'); }
      else { odrzucone.add(k.peak); f.classList.add('odrzucony'); }
      zapamietaj(); odswiez();
    };
    siatka.appendChild(f);
  });
  odswiez();
  if (odrzucone.size) stan.textContent = `Wczytano poprzednie zaznaczenie: ${odrzucone.size} odrzuconych.`;
});

document.getElementById('zaden').onclick = () => {
  if (!confirm('Cofnac WSZYSTKIE odrzucenia?')) return;
  odrzucone.clear();
  document.querySelectorAll('#siatka figure').forEach(f => f.classList.remove('odrzucony'));
  zapamietaj(); odswiez();
};
zapisz.onclick = () => wyslij('/zapisz', true);
window.addEventListener('beforeunload', () => {
  try { localStorage.setItem(KLUCZ, JSON.stringify([...odrzucone])); } catch (e) {}
});
</script>
"""


class Handler(SimpleHTTPRequestHandler):
    """Serwuje strone wyboru i przyjmuje zapis zaznaczenia."""

    katalog: Path = Path(DEFAULT_DIR)

    def do_GET(self) -> None:  # noqa: N802 (nazwa wymagana przez biblioteke)
        """Oddaje strone albo plik z katalogu kandydatow."""
        if self.path in ("/", "/index.html"):
            body = STRONA.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        """Zapisuje zaznaczenie: `/postep` w trakcie pracy, `/zapisz` na koniec."""
        if self.path not in ("/zapisz", "/postep"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        pary = json.loads(self.rfile.read(length).decode("utf-8"))
        cel = self.katalog / (
            PROGRESS_NAME if self.path == "/postep" else SELECTION_NAME
        )
        cel.write_text(json.dumps(pary, ensure_ascii=False, indent=1), encoding="utf-8")
        odpowiedz = json.dumps({"zapisane": len(pary), "plik": str(cel)}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(odpowiedz)))
        self.end_headers()
        self.wfile.write(odpowiedz)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Cisza w konsoli — kazdy kafelek to osobne zadanie."""


def main() -> None:
    """Punkt wejscia CLI."""
    parser = argparse.ArgumentParser(description="Strona do recznego wyboru kadrow")
    parser.add_argument("--dir", default=DEFAULT_DIR)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    katalog = Path(args.dir).resolve()
    Handler.katalog = katalog

    def handler(*a: object, **k: object) -> Handler:
        return Handler(*a, directory=str(katalog), **k)  # type: ignore[arg-type]

    print(f"Kandydaci z {katalog}")
    print(f"Otworz http://127.0.0.1:{args.port}")
    HTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
