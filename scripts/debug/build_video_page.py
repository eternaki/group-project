#!/usr/bin/env python3
"""
Składa stronę HTML z widoku nagraniami (`sample_by_video`).

Użycie:
    python -m scripts.debug.build_video_page --out data/review_by_video/index.html
"""

import argparse
import base64
import json
from pathlib import Path

DEFAULT_SAMPLES: str = "data/review_by_video"

GROUP_LABEL: dict[str, str] = {
    "obrot_glowy": "поворот головы",
    "slabe_punkty": "точки неуверенные",
    "mala_morda": "морда мелкая",
    "wiele_powodow": "несколько причин",
    "zla_neutralna": "нет нейтрали",
}

# Powyżej tylu przyjętych klatek nagranie uznajemy za wydajne.
PRODUCTIVE_KEPT: int = 4


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


def _video_card(entry: dict, samples: Path) -> str:
    """
    Buduje kartę jednego nagrania.

    Args:
        entry: Wpis z manifestu
        samples: Katalog z obrazami

    Returns:
        Fragment HTML
    """
    kept, total = entry["kept"], entry["total"]
    tone = "keep" if kept >= PRODUCTIVE_KEPT else ("maybe" if kept else "lost")
    tiles = []
    for frame in entry["frames"]:
        uri = _data_uri(samples / frame["file"])
        state = "ok" if frame["accepted"] else "no"
        badge = (
            "в очередь"
            if frame["accepted"]
            else GROUP_LABEL.get(frame["group"], frame["group"])
        )
        tiles.append(
            f'<figure class="tile" data-state="{state}">'
            f'<img src="{uri}" alt="" loading="lazy">'
            f'<figcaption><b>{badge}</b>{frame["caption"]}</figcaption></figure>'
        )
    losses = " · ".join(
        f"{GROUP_LABEL.get(key, key)} {count}"
        for key, count in sorted(entry["reasons"].items(), key=lambda kv: -kv[1])
    )
    bar = round(100 * kept / total) if total else 0
    return f"""<article class="video" data-tone="{tone}">
  <header>
    <h3>{entry["video"]}</h3>
    <div class="tally">
      <span class="score"><b>{kept}</b> из {total}</span>
      <div class="bar"><i style="width:{bar}%"></i></div>
    </div>
    <p class="losses">{losses or "потерь нет"}</p>
  </header>
  <div class="strip">{"".join(tiles)}</div>
</article>"""


def _distribution(rows: list[dict], videos: int) -> str:
    """
    Buduje tabelę rozkładu klatek na nagranie.

    Args:
        rows: Wiersze rozkładu
        videos: Liczba nagrań ogółem

    Returns:
        Fragment HTML
    """
    peak = max((row["videos"] for row in rows), default=1)
    body = []
    for row in rows:
        share = 100 * row["videos"] / videos
        body.append(
            f'<tr><th>{row["kept"]}</th>'
            f'<td class="num">{row["videos"]}</td>'
            f'<td class="num">{share:.1f}%</td>'
            f'<td class="num">{row["frames"]}</td>'
            f'<td class="plot"><i style="width:{100 * row["videos"] / peak:.1f}%"></i></td>'
            f"</tr>"
        )
    return f"""<table>
  <thead><tr><th>кадров в очередь</th><th>роликов</th><th>доля</th>
  <th>кадров всего</th><th></th></tr></thead>
  <tbody>{"".join(body)}</tbody>
</table>"""


def build(samples: Path) -> str:
    """
    Składa całą stronę.

    Args:
        samples: Katalog wyjściowy `sample_by_video`

    Returns:
        Treść HTML
    """
    manifest = json.loads((samples / "manifest.json").read_text(encoding="utf-8"))
    summary = manifest["summary"]
    shown = manifest["shown"]

    productive = [entry for entry in shown if entry["kept"] >= PRODUCTIVE_KEPT]
    barren = [entry for entry in shown if entry["kept"] < PRODUCTIVE_KEPT]
    empty = summary["videos"] - summary["videos_with_any"]

    return _TEMPLATE.format(
        videos=summary["videos"],
        peaks=summary["peaks"],
        kept=summary["kept"],
        with_any=summary["videos_with_any"],
        empty=empty,
        empty_share=round(100 * empty / summary["videos"]),
        avg=f"{summary['kept'] / max(summary['videos_with_any'], 1):.1f}",
        table=_distribution(summary["distribution"], summary["videos"]),
        productive="".join(_video_card(entry, samples) for entry in productive),
        barren="".join(_video_card(entry, samples) for entry in barren),
    )


_TEMPLATE = """<title>Урожай с одного ролика</title>
<style>
:root {{
  color-scheme: light dark;
  --ground: #e9ebef; --panel: #ffffff; --edge: #d2d7de;
  --ink: #191d24; --muted: #626b79; --accent: #2f6094;
  --keep: #1f7a5a; --maybe: #9a6412; --lost: #9c3b46;
  --shadow: 0 1px 2px rgba(20, 26, 35, .07);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground: #12151a; --panel: #1a1e25; --edge: #2b323c;
    --ink: #e3e7ed; --muted: #8d97a5; --accent: #7fb0e0;
    --keep: #55c39a; --maybe: #d6a24c; --lost: #e08490;
    --shadow: 0 1px 2px rgba(0, 0, 0, .35);
  }}
}}
:root[data-theme="dark"] {{
  --ground: #12151a; --panel: #1a1e25; --edge: #2b323c;
  --ink: #e3e7ed; --muted: #8d97a5; --accent: #7fb0e0;
  --keep: #55c39a; --maybe: #d6a24c; --lost: #e08490;
  --shadow: 0 1px 2px rgba(0, 0, 0, .35);
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: var(--ground); color: var(--ink);
  font: 15px/1.55 ui-sans-serif, system-ui, "Segoe UI", Roboto, sans-serif;
}}
.wrap {{ max-width: 1280px; margin: 0 auto; padding: 32px 20px 80px; }}
h1 {{ font-size: 27px; line-height: 1.2; margin: 0 0 10px; letter-spacing: -.01em; }}
.lede {{ margin: 0; max-width: 70ch; color: var(--muted); }}
h2 {{
  font-size: 13px; text-transform: uppercase; letter-spacing: .08em;
  color: var(--muted); margin: 42px 0 4px; font-weight: 600;
}}
h2 + p {{ margin: 0 0 16px; max-width: 74ch; color: var(--muted); font-size: 14px; }}
.stats {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(132px, 1fr));
  gap: 1px; background: var(--edge); border: 1px solid var(--edge);
  border-radius: 7px; overflow: hidden; margin: 24px 0 0; box-shadow: var(--shadow);
}}
.stats div {{ background: var(--panel); padding: 12px 14px; }}
.stats dt {{
  font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
  color: var(--muted); margin-bottom: 3px;
}}
.stats dd {{
  margin: 0; font-size: 23px; letter-spacing: -.02em;
  font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
  font-variant-numeric: tabular-nums;
}}
.stats dd small {{ font-size: 12px; color: var(--muted); letter-spacing: 0; }}
.scroll {{ overflow-x: auto; }}
table {{
  width: 100%; border-collapse: collapse; background: var(--panel);
  border: 1px solid var(--edge); border-radius: 7px; overflow: hidden;
  box-shadow: var(--shadow); font-size: 13.5px;
}}
thead th {{
  text-align: left; font-size: 11px; text-transform: uppercase;
  letter-spacing: .06em; color: var(--muted); font-weight: 600;
  padding: 9px 12px; border-bottom: 1px solid var(--edge); white-space: nowrap;
}}
tbody th, tbody td {{ padding: 6px 12px; border-bottom: 1px solid var(--edge); }}
tbody tr:last-child th, tbody tr:last-child td {{ border-bottom: 0; }}
tbody th {{ text-align: left; font-weight: 600; }}
.num {{
  font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
  font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap;
}}
.plot {{ width: 34%; min-width: 90px; }}
.plot i {{ display: block; height: 7px; border-radius: 2px; background: var(--accent); opacity: .55; }}
.video {{
  background: var(--panel); border: 1px solid var(--edge);
  border-left: 4px solid var(--tone); border-radius: 7px;
  padding: 13px 15px; margin-bottom: 12px; box-shadow: var(--shadow);
}}
.video[data-tone="keep"] {{ --tone: var(--keep); }}
.video[data-tone="maybe"] {{ --tone: var(--maybe); }}
.video[data-tone="lost"] {{ --tone: var(--lost); }}
.video header {{ display: flex; flex-direction: column; gap: 5px; margin-bottom: 11px; }}
.video h3 {{
  margin: 0; font-size: 14px; font-weight: 600; word-break: break-word;
  font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
}}
.tally {{ display: flex; align-items: center; gap: 12px; }}
.score {{
  font-size: 12.5px; color: var(--muted); white-space: nowrap;
  font-variant-numeric: tabular-nums;
}}
.score b {{ color: var(--tone); font-size: 15px; }}
.bar {{ flex: 1; max-width: 220px; height: 5px; border-radius: 3px; background: var(--edge); }}
.bar i {{ display: block; height: 100%; border-radius: 3px; background: var(--tone); }}
.losses {{ margin: 0; font-size: 12px; color: var(--muted); }}
.strip {{ display: flex; gap: 9px; overflow-x: auto; padding-bottom: 5px; }}
.tile {{
  margin: 0; flex: 0 0 152px; border-radius: 5px; overflow: hidden;
  border: 1px solid var(--edge); border-top: 3px solid var(--state);
  background: var(--ground);
}}
.tile[data-state="ok"] {{ --state: var(--keep); }}
.tile[data-state="no"] {{ --state: var(--lost); }}
.tile img {{ display: block; width: 100%; height: auto; background: #0c0e12; }}
figcaption {{
  padding: 4px 6px; font-size: 9.5px; color: var(--muted); line-height: 1.45;
  font-family: ui-monospace, "Cascadia Mono", Consolas, monospace;
  font-variant-numeric: tabular-nums;
}}
figcaption b {{ display: block; color: var(--state); font-size: 10px; }}
@media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; }} }}
</style>

<div class="wrap">
  <h1>Урожай с одного ролика</h1>
  <p class="lede">
    Тот же набор dataset_v2, но разложенный по видео, а не по причинам отсева.
    Кадры не независимы: одно удачное видео даёт дюжину, а видео с собакой
    в профиль не даст ни одного, даже снятое в 4K. Здесь видно, чем одно
    отличается от другого.
  </p>

  <dl class="stats">
    <div><dt>роликов с пиками</dt><dd>{videos}</dd></div>
    <div><dt>пиков найдено</dt><dd>{peaks}</dd></div>
    <div><dt>в очередь</dt><dd>{kept}</dd></div>
    <div><dt>роликов дали хоть кадр</dt><dd>{with_any}</dd></div>
    <div><dt>роликов впустую</dt><dd>{empty}<small> · {empty_share}%</small></dd></div>
    <div><dt>кадров с урожайного</dt><dd>{avg}</dd></div>
  </dl>

  <h2>Сколько роликов дало сколько кадров</h2>
  <p>
    Больше половины роликов не дали ничего. Средний по набору — это ролик
    с одним-двумя кадрами, а не с дюжиной.
  </p>
  <div class="scroll">{table}</div>

  <h2>Что даёт урожай</h2>
  <p>
    Самые продуктивные ролики набора со всеми их пиками. Зелёная полоса сверху —
    кадр ушёл в очередь, красная — отсеян. Полоса прокручивается вбок.
  </p>
  {productive}

  <h2>Что уходит впустую</h2>
  <p>
    Ролики, где пайплайн нашёл четыре и больше пиков, а выжил один или ни одного.
    Здесь самый большой единичный выигрыш от послабления порогов — и здесь же
    видно, стоит ли он того.
  </p>
  {barren}
</div>
"""


def main() -> None:
    """Punkt wejścia: buduje stronę i zapisuje ją na dysk."""
    parser = argparse.ArgumentParser(description="Strona z widokiem nagraniami")
    parser.add_argument("--samples", type=Path, default=Path(DEFAULT_SAMPLES))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or args.samples / "index.html"
    out.write_text(build(args.samples), encoding="utf-8")
    print(f"Zapisano {out} ({out.stat().st_size / (1024 * 1024):.1f} MB)")


if __name__ == "__main__":
    main()
