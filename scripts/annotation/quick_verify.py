#!/usr/bin/env python3
"""
Szybka weryfikacja SAMEJ emocji — kadr i przyciski, bez punktów i AU.

    python scripts/annotation/quick_verify.py            # port 8001

Pokazuje klatki szczytowe (najpierw najbardziej prawdopodobnie wyraziste), a
anotator wybiera tylko emocję (myszą lub klawiszem 1-9), albo pomija. Werdykt
ląduje w `data/labels/<zbior>/quick_<kto>.jsonl` w formacie dziennika, więc
`build_final_dataset` / `build_full_dataset` bierze go jako `human_verified`.
Punkty i AU zostają automatyczne — tego minimum od prowadzącego nie wymaga.
"""

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

REPO = Path(__file__).resolve().parent.parent.parent
DATASET = "dataset_final"
FRAMES = REPO / "data" / DATASET / "work" / "frames"
CURATED = REPO / "data" / DATASET / "work" / "curated.json"
LABELS = REPO / "data" / "labels" / DATASET
PORT = 8001

EMOTIONS = [
    "happy", "sad", "angry", "fearful", "surprise",
    "pain", "submission", "relaxed", "neutral",
]
CALM = {"relaxed", "neutral"}

_coco = json.loads(CURATED.read_text(encoding="utf-8"))
_img = {i["id"]: i for i in _coco["images"]}
_peaks = [a for a in _coco["annotations"] if a.get("frame_role") == "peak"]


def _score(annotation: dict) -> int:
    """Im wyżej, tym bardziej prawdopodobnie wyrazisty kadr — idzie na górę kolejki."""
    non_calm = 0 if annotation.get("emotion") in CALM else 1000
    active = sum(
        1
        for value in (annotation.get("au_analysis") or {}).values()
        if isinstance(value, dict) and value.get("is_active")
    )
    return non_calm + active


_peaks.sort(key=_score, reverse=True)
QUEUE = [_img[a["image_id"]]["file_name"] for a in _peaks]
EMO_BY_KEY = {_img[a["image_id"]]["file_name"]: a.get("emotion") for a in _peaks}

app = FastAPI()


def _seen(annotator: str) -> set[str]:
    """Klatki już ocenione LUB pominięte przez tę osobę — nie pokazujemy ich ponownie."""
    keys: set[str] = set()
    for name in (f"{annotator}.jsonl", f"quick_{annotator}.jsonl"):
        path = LABELS / name
        if not path.is_file():
            continue
        for line in path.open(encoding="utf-8"):
            if line.strip():
                record = json.loads(line)
                if record.get("emotion"):
                    keys.add(record["pair_key"])
    skip = LABELS / f"skip_{annotator}.txt"
    if skip.is_file():
        keys.update(line.strip() for line in skip.open(encoding="utf-8") if line.strip())
    return keys


@app.get("/next")
def next_frame(annotator: str) -> JSONResponse:
    """Zwraca następną nieocenioną klatkę tej osoby."""
    done = _seen(annotator)
    todo = [k for k in QUEUE if k not in done]
    payload = {"total": len(QUEUE), "done": len(QUEUE) - len(todo)}
    if todo:
        payload["pair_key"] = todo[0]
        payload["auto_emotion"] = EMO_BY_KEY.get(todo[0])
    return JSONResponse(payload)


@app.get("/img")
def image(path: str) -> FileResponse:
    """Serwuje pełną klatkę."""
    return FileResponse(FRAMES / path)


@app.post("/save")
async def save(payload: dict) -> JSONResponse:
    """Zapisuje werdykt emocji, albo pomija klatkę (bez zapisu do dziennika)."""
    annotator = payload["annotator"]
    pair_key = payload["pair_key"]
    emotion = payload.get("emotion")
    LABELS.mkdir(parents=True, exist_ok=True)

    # Pominięcie nie jest etykietą — nie brudzimy dziennika. Zapamiętujemy tylko,
    # że tej klatki już nie pokazywać (osobny plik, którego build nie czyta).
    if not emotion:
        with (LABELS / f"skip_{annotator}.txt").open("a", encoding="utf-8") as handle:
            handle.write(pair_key + "\n")
        return JSONResponse({"ok": True, "skipped": True})

    record = {
        "pair_key": pair_key,
        "annotator": annotator,
        "emotion": emotion,
        "usable": True,
        "keypoints": None,
        "keypoints_ok": None,
        "breed": None,
        "au_verdicts": {},
        "roles_swapped": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with (LABELS / f"quick_{annotator}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return JSONResponse({"ok": True})


TARGET = 200
ENOUGH = {"sad", "relaxed", "neutral"}


@app.get("/stats")
def stats() -> JSONResponse:
    """Ile zebrano na emocję (zweryfikowane człowiekiem) i ile jeszcze trzeba."""
    counter: Counter = Counter()
    for path in LABELS.glob("*.jsonl"):
        for line in path.open(encoding="utf-8"):
            if line.strip():
                record = json.loads(line)
                if record.get("usable") and record.get("emotion"):
                    counter[record["emotion"]] += 1
    out = []
    for emotion in EMOTIONS:
        have = counter.get(emotion, 0)
        target = None if emotion in ENOUGH else TARGET
        out.append({"emotion": emotion, "have": have, "target": target})
    return JSONResponse(out)


HTML = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1"><title>Dog emocje</title>
<style>
body{background:#111;color:#eee;font-family:system-ui;margin:0;text-align:center}
#img{max-height:70vh;max-width:96vw;margin-top:8px;border-radius:8px}
.bar{position:fixed;bottom:0;left:0;right:0;background:#1b1b1b;padding:10px}
button{font-size:16px;margin:4px;padding:10px 14px;border:0;border-radius:8px;cursor:pointer;background:#2a2a2a;color:#eee}
button:hover{background:#3a6}
.skip{background:#553}
#hdr{padding:8px;font-size:15px}
.k{opacity:.5;font-size:12px}
#stats{position:fixed;top:6px;right:6px;background:#1b1b1b;padding:8px 10px;border-radius:8px;font-size:13px;text-align:left;line-height:1.5;z-index:9}
#stats b{display:block;margin-bottom:4px}
.done{color:#5c6}.need{color:#e88}
</style></head><body>
<div id="stats"></div>
<div id="hdr"></div>
<img id="img">
<div class="bar" id="bar"></div>
<script>
const EMO=[["happy","радость"],["sad","грусть"],["angry","злость"],["fearful","страх"],["surprise","удивление"],["pain","боль"],["submission","подчинение"],["relaxed","расслаблен"],["neutral","нейтрально"]];
let annotator=new URLSearchParams(location.search).get("kto")||prompt("Kto anotuje? (danek/masha/anton/mafin)","danek");
let cur=null;
const bar=document.getElementById("bar");
EMO.forEach((e,i)=>{const b=document.createElement("button");b.innerHTML=e[0]+" · "+e[1]+" <span class='k'>"+(i+1)+"</span>";b.onclick=()=>save(e[0]);bar.appendChild(b);});
const sk=document.createElement("button");sk.className="skip";sk.innerHTML="skip · пропустить <span class='k'>0</span>";sk.onclick=()=>save(null);bar.appendChild(sk);
async function load(){const r=await fetch("/next?annotator="+annotator);const d=await r.json();
 document.getElementById("hdr").innerHTML=annotator+" — "+d.done+" / "+d.total+(d.pair_key?(" · авто: "+(d.auto_emotion||"?")):"");
 if(!d.pair_key){document.getElementById("img").style.display="none";document.getElementById("hdr").innerHTML="Готово! Всё размечено.";return;}
 cur=d.pair_key;document.getElementById("img").src="/img?path="+encodeURIComponent(cur);}
async function save(emo){if(!cur)return;await fetch("/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({annotator,pair_key:cur,emotion:emo})});load();stats();}
async function stats(){const r=await fetch("/stats");const d=await r.json();
 let h="<b>Собрано / нужно</b>";
 for(const s of d){if(s.target===null){h+=s.emotion+": "+s.have+" ✓<br>";}
  else{const left=Math.max(0,s.target-s.have);const cls=left===0?"done":"need";h+="<span class='"+cls+"'>"+s.emotion+": "+s.have+" / "+s.target+" (ещё "+left+")</span><br>";}}
 document.getElementById("stats").innerHTML=h;}
document.addEventListener("keydown",e=>{if(e.key>="1"&&e.key<="9")save(EMO[+e.key-1][0]);if(e.key==="0")save(null);});
load();stats();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Strona weryfikacji."""
    return HTML


if __name__ == "__main__":
    print(f"Kolejka: {len(QUEUE)} klatek. Otwórz http://localhost:{PORT}/", file=sys.stderr)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
