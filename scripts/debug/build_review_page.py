#!/usr/bin/env python3
"""
Składa jedną stronę HTML z próbek wyrenderowanych przez `sample_rejected_frames`.

Obrazy idą w treść jako data URI, bo strona ma działać z jednego pliku — także
wtedy, gdy ktoś przekaże ją dalej bez katalogu `data/`.

Użycie:
    python -m scripts.debug.build_review_page --out data/review_samples/index.html
"""

import argparse
import base64
import json
from pathlib import Path

DEFAULT_SAMPLES: str = "data/review_samples"

# Kolejność sekcji: najpierw materiał przyjęty jako punkt odniesienia, potem
# straty od najłatwiejszych do odzyskania po beznadziejne.
GROUP_ORDER: tuple[str, ...] = (
    "przyjete",
    "zla_neutralna",
    "mala_morda",
    "obrot_glowy",
    "slabe_punkty",
    "wiele_powodow",
)

GROUP_META: dict[str, dict[str, str]] = {
    "przyjete": {
        "title": "Отобрано в очередь",
        "verdict": "эталон",
        "tone": "keep",
        "note": (
            "Это то, что размечается сейчас. Смотри на них первыми — остальные "
            "группы имеет смысл оценивать только в сравнении с этой планкой."
        ),
    },
    "zla_neutralna": {
        "title": "Кадр хороший, нет нейтрали",
        "verdict": "не спасти",
        "tone": "lost",
        "note": (
            "Сам кадр прошёл бы бракеровку. Выпал потому, что во всём треке нет "
            "годной нейтральной клетки, а без неё не от чего считать AU. "
            "Потеря не в этом кадре — и починить её порогами нельзя."
        ),
    },
    "mala_morda": {
        "title": "Морда мелкая",
        "verdict": "спорно",
        "tone": "maybe",
        "note": (
            "Единственная претензия — разрешение. Здесь важно твоё мнение: "
            "если ты различаешь ухо и пасть, порог можно опустить."
        ),
    },
    "obrot_glowy": {
        "title": "Поворот головы",
        "verdict": "часть вернуть",
        "tone": "maybe",
        "note": (
            "Самая большая группа и самая спорная. Правила здесь врут: всё "
            "делится на расстояние между глазами, а оно сжимается при повороте, "
            "поэтому AU включаются «на рост» сами. Но читает-то человек, и на "
            "ракурсе три четверти он видит достаточно. Смотри, где именно "
            "закрывается дальняя половина морды — это и есть граница."
        ),
    },
    "slabe_punkty": {
        "title": "Точки неуверенные",
        "verdict": "скорее нет",
        "tone": "lost",
        "note": (
            "Оранжевые точки — те, в которых модель сомневается. Здесь их "
            "слишком много: геометрия становится выдумкой, и поправить руками "
            "пришлось бы половину разметки."
        ),
    },
    "wiele_powodow": {
        "title": "Несколько причин сразу",
        "verdict": "не брать",
        "tone": "lost",
        "note": (
            "Поворот вместе с мелкой мордой или потерянными точками. "
            "Дно выборки — показано для полноты картины."
        ),
    },
}


def _data_uri(path: Path) -> str:
    """
    Koduje plik JPEG jako data URI.

    Args:
        path: Ścieżka do obrazu

    Returns:
        Napis `data:image/jpeg;base64,...`
    """
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _section(group: str, entries: list[dict], total: int, samples: Path) -> str:
    """
    Buduje sekcję HTML jednej grupy.

    Args:
        group: Klucz grupy
        entries: Wpisy z manifestu
        total: Ile klatek grupa ma w całym zbiorze
        samples: Katalog z obrazami

    Returns:
        Fragment HTML
    """
    meta = GROUP_META[group]
    tiles = []
    for entry in entries:
        uri = _data_uri(samples / entry["file"])
        source = entry["src"].rsplit("/", 1)[-1]
        tiles.append(
            f'<figure class="tile"><img src="{uri}" alt="{source}" loading="lazy">'
            f'<figcaption>{entry["caption"]}</figcaption></figure>'
        )
    return f"""<section class="group" id="{group}">
  <header class="group-head" data-tone="{meta["tone"]}">
    <div class="group-id">
      <h2>{meta["title"]}</h2>
      <span class="verdict">{meta["verdict"]}</span>
    </div>
    <p class="note">{meta["note"]}</p>
    <dl class="counts">
      <div><dt>всего в наборе</dt><dd>{total}</dd></div>
      <div><dt>показано</dt><dd>{len(entries)}</dd></div>
    </dl>
  </header>
  <div class="ramp"><span>легче</span><i></i><span>тяжелее</span></div>
  <div class="grid">{"".join(tiles)}</div>
</section>"""


def build(samples: Path) -> str:
    """
    Składa całą stronę.

    Args:
        samples: Katalog wyjściowy `sample_rejected_frames`

    Returns:
        Treść HTML
    """
    manifest = json.loads((samples / "manifest.json").read_text(encoding="utf-8"))
    totals = manifest["totals"]
    groups = manifest["groups"]

    nav = "".join(
        f'<a href="#{key}"><span>{GROUP_META[key]["title"]}</span>'
        f'<b>{totals.get(key, 0)}</b></a>'
        for key in GROUP_ORDER
        if key in groups
    )
    sections = "".join(
        _section(key, groups[key], totals.get(key, 0), samples)
        for key in GROUP_ORDER
        if key in groups
    )
    rejected = sum(count for key, count in totals.items() if key != "przyjete")
    kept = totals.get("przyjete", 0)
    return _TEMPLATE.format(nav=nav, sections=sections, kept=kept, rejected=rejected)


_TEMPLATE = """<title>Смотровой стол бракеровки</title>
<style>
:root {{
  color-scheme: light dark;
  --ground: #e9ebef;
  --panel: #ffffff;
  --edge: #d2d7de;
  --ink: #191d24;
  --muted: #626b79;
  --accent: #2f6094;
  --keep: #1f7a5a;
  --maybe: #9a6412;
  --lost: #9c3b46;
  --shadow: 0 1px 2px rgba(20, 26, 35, .07);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground: #12151a;
    --panel: #1a1e25;
    --edge: #2b323c;
    --ink: #e3e7ed;
    --muted: #8d97a5;
    --accent: #7fb0e0;
    --keep: #55c39a;
    --maybe: #d6a24c;
    --lost: #e08490;
    --shadow: 0 1px 2px rgba(0, 0, 0, .35);
  }}
}}
:root[data-theme="dark"] {{
  --ground: #12151a;
  --panel: #1a1e25;
  --edge: #2b323c;
  --ink: #e3e7ed;
  --muted: #8d97a5;
  --accent: #7fb0e0;
  --keep: #55c39a;
  --maybe: #d6a24c;
  --lost: #e08490;
  --shadow: 0 1px 2px rgba(0, 0, 0, .35);
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font: 15px/1.55 ui-sans-serif, system-ui, "Segoe UI", Roboto, sans-serif;
}}
.wrap {{ max-width: 1280px; margin: 0 auto; padding: 32px 20px 80px; }}
header.page {{ display: flex; flex-direction: column; gap: 10px; margin-bottom: 8px; }}
h1 {{ font-size: 27px; line-height: 1.2; margin: 0; text-wrap: balance; letter-spacing: -.01em; }}
.lede {{ margin: 0; max-width: 68ch; color: var(--muted); }}
.legend {{
  display: flex; flex-wrap: wrap; gap: 8px 20px; margin-top: 6px;
  font-size: 12.5px; color: var(--muted);
}}
.legend i {{
  width: 9px; height: 9px; border-radius: 50%; display: inline-block;
  margin-right: 6px; vertical-align: -1px;
}}
nav {{
  position: sticky; top: 0; z-index: 5; margin: 22px 0 30px;
  display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1px; background: var(--edge); border: 1px solid var(--edge);
  border-radius: 7px; overflow: hidden; box-shadow: var(--shadow);
}}
nav a {{
  background: var(--panel); padding: 9px 12px; text-decoration: none;
  color: var(--ink); display: flex; justify-content: space-between;
  align-items: baseline; gap: 8px; font-size: 13px;
}}
nav a:hover, nav a:focus-visible {{ background: var(--ground); outline: none; }}
nav a:focus-visible {{ box-shadow: inset 0 0 0 2px var(--accent); }}
nav b {{
  font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
  font-variant-numeric: tabular-nums; color: var(--muted); font-weight: 500;
}}
.group {{ margin-bottom: 46px; scroll-margin-top: 64px; }}
.group-head {{
  background: var(--panel); border: 1px solid var(--edge);
  border-left: 4px solid var(--tone); border-radius: 7px;
  padding: 15px 18px; box-shadow: var(--shadow);
  display: flex; flex-direction: column; gap: 9px;
}}
.group-head[data-tone="keep"] {{ --tone: var(--keep); }}
.group-head[data-tone="maybe"] {{ --tone: var(--maybe); }}
.group-head[data-tone="lost"] {{ --tone: var(--lost); }}
.group-id {{ display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }}
.group-id h2 {{ font-size: 19px; margin: 0; letter-spacing: -.01em; }}
.verdict {{
  font-size: 11.5px; text-transform: uppercase; letter-spacing: .07em;
  color: var(--tone); border: 1px solid var(--tone); border-radius: 3px;
  padding: 2px 7px; white-space: nowrap;
}}
.note {{ margin: 0; max-width: 74ch; color: var(--muted); font-size: 14px; }}
.counts {{ display: flex; gap: 26px; margin: 2px 0 0; }}
.counts div {{ display: flex; flex-direction: column; gap: 1px; }}
.counts dt {{
  font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
  color: var(--muted);
}}
.counts dd {{
  margin: 0; font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
  font-variant-numeric: tabular-nums; font-size: 17px;
}}
.ramp {{
  display: flex; align-items: center; gap: 9px; margin: 13px 2px 9px;
  font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
  color: var(--muted);
}}
.ramp i {{
  flex: 1; height: 2px; border-radius: 1px;
  background: linear-gradient(90deg, var(--keep), var(--maybe), var(--lost));
}}
.grid {{
  display: grid; gap: 12px;
  grid-template-columns: repeat(auto-fill, minmax(178px, 1fr));
}}
.tile {{
  margin: 0; background: var(--panel); border: 1px solid var(--edge);
  border-radius: 6px; overflow: hidden; box-shadow: var(--shadow);
}}
.tile img {{ display: block; width: 100%; height: auto; background: #0c0e12; }}
figcaption {{
  padding: 5px 7px; font-size: 10.5px; color: var(--muted);
  font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
  font-variant-numeric: tabular-nums; border-top: 1px solid var(--edge);
  white-space: nowrap; overflow-x: auto;
}}
@media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; }} }}
</style>

<div class="wrap">
  <header class="page">
    <h1>Что бракеровка выбросила и стоит ли это возвращать</h1>
    <p class="lede">
      Пики набора dataset_final, разложенные по причине отсева. Внутри каждой группы
      выборка взята <strong>равномерно по всему диапазону</strong>, а не с лучшего
      края, и отсортирована от лёгких случаев к тяжёлым — так видно, где проходит
      граница, а не только удачные примеры. В очередь отобрано {kept},
      отброшено {rejected}.
    </p>
    <p class="legend">
      <span><i style="background:#22b06a"></i>точка уверенная</span>
      <span><i style="background:#e08a1e"></i>модель сомневается</span>
      <span>подпись: асимметрия &middot; доля слабых точек &middot; ширина морды</span>
    </p>
  </header>
  <nav>{nav}</nav>
  {sections}
</div>
"""


def main() -> None:
    """Punkt wejścia: buduje stronę i zapisuje ją na dysk."""
    parser = argparse.ArgumentParser(description="Strona z podgladem odrzuconych klatek")
    parser.add_argument("--samples", type=Path, default=Path(DEFAULT_SAMPLES))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or args.samples / "index.html"
    out.write_text(build(args.samples), encoding="utf-8")
    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"Zapisano {out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
