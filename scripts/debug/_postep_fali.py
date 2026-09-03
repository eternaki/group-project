"""
Zlicza postep fali: skonczone nagrania, kandydatow na pary i odrzucone treki.

UWAGA na liczenie kadrow. Klatki NIE leza w katalogu wyjsciowym fali, tylko
w `--frames-dir` (domyslnie `data/frames`). Liczenie `*.jpg` pod katalogiem
fali daje wiec zawsze zero i wyglada, jakby material nic nie dawal — pomylka
kosztowala kilka godzin blednego wniosku 31.08.2026. Zrodlem prawdy sa
czesciowe pliki COCO: to one mowia, ile kandydatow naprawde powstalo.

Uzycie:
    python scripts/debug/_postep_fali.py data/fala_tiktok/out
"""

import glob
import json
import sys
from pathlib import Path


def main() -> None:
    """Wypisuje jedna linie z postepem fali."""
    katalog = sys.argv[1] if len(sys.argv) > 1 else "data/fala_tiktok/out"
    skonczone = odrzucone = peaki = neutralne = 0
    for plik in glob.glob(f"{katalog}/shard_*/progress.json"):
        try:
            dane = json.loads(Path(plik).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        zrobione = dane.get("processed_videos", 0)
        skonczone += len(zrobione) if isinstance(zrobione, list) else zrobione
        odrzucone += dane.get("rejected_track_count", 0)

    for plik in glob.glob(f"{katalog}/shard_*/annotations.json"):
        try:
            coco = json.loads(Path(plik).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for anotacja in coco.get("annotations", []):
            if anotacja.get("frame_role") == "peak":
                peaki += 1
            elif anotacja.get("frame_role") == "neutral":
                neutralne += 1

    print(
        f"skonczonych {skonczone}, kandydatow {peaki} (neutralnych {neutralne}), "
        f"odrzuconych trekow {odrzucone}"
    )


if __name__ == "__main__":
    main()
