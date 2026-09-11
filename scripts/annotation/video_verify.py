#!/usr/bin/env python3
"""
Weryfikacja NA POZIOMIE WIDEO z ODTWARZANIEM prawdziwego wideo (z Google Drive).

Dla każdego wideo serwer pobiera plik z Drive (cache), oddaje do <video>.
Anotator ogląda, wybiera JEDNĄ emocję i zaznacza czas POCZĄTKU i KOŃCA emocji
(przyciski łapią bieżący czas odtwarzania). Werdykt -> video_<kto>.jsonl.
Kolejka globalna z rezerwacją. Publiczny link (tunel) serwuje wideo wszystkim —
klienci nie potrzebują dostępu do Drive, robi to serwer.
"""

import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from scripts.download.tiktok.config import (
    GDRIVE_CREDENTIALS_PATH,
    GDRIVE_FOLDER_ID,
    GDRIVE_TOKEN_PATH,
)
from scripts.download.tiktok.drive_uploader import GoogleDriveUploader

REPO = Path(__file__).resolve().parent.parent.parent
DATASET = "dataset_final"
SRC = REPO / "data" / DATASET / "work" / "annotations_full.json"
LABELS = REPO / "data" / "labels" / DATASET
CACHE = Path("/tmp/dogvids")
CACHE.mkdir(exist_ok=True)
PORT = 8001

EMOTIONS = ["neutral", "sad", "happy", "surprise", "angry", "fearful"]
EMO_PL = {"neutral": "нейтрально", "sad": "грусть", "happy": "радость", "surprise": "удивление",
          "angry": "злость", "fearful": "страх", "submission": "подчинение"}
TARGET = 50

# Kolejka wideo (unikalne source_video)
_coco = json.loads(SRC.read_text(encoding="utf-8"))
_seen: set[str] = set()
VIDEOS: list[str] = []
for _i in _coco["images"]:
    _v = _i.get("source_video") or _i["file_name"].split("/")[-2]
    if _v not in _seen:
        _seen.add(_v)
        VIDEOS.append(_v)

# Emocja modelu na wideo (dominująca) — TYLKO jako podpowiedź do kolejności kolejki,
# nie etykieta. Pozwala pokazywać najpierw wideo prawdopodobnie deficytowych emocji.
_img_sv = {i["id"]: (i.get("source_video") or i["file_name"].split("/")[-2]) for i in _coco["images"]}
_votes: dict[str, Counter] = defaultdict(Counter)
for _a in _coco["annotations"]:
    _e = _a.get("emotion")
    if _e:
        _votes[_img_sv[_a["image_id"]]][_e] += 1
VIDEO_EMO = {v: c.most_common(1)[0][0] for v, c in _votes.items()}

# Drive
_drive = GoogleDriveUploader(GDRIVE_CREDENTIALS_PATH, GDRIVE_TOKEN_PATH, GDRIVE_FOLDER_ID)
_drive.authenticate()
_name2id: dict[str, str | None] = {}
_dl_lock = Lock()

_CLAIMS: dict[str, float] = {}
_CLAIM_TTL = 300.0
_LOCK = Lock()

app = FastAPI()


def _find_drive(name: str) -> str | None:
    """Znajduje ID pliku wideo na Drive po nazwie (z cache)."""
    if name in _name2id:
        return _name2id[name]
    svc = _drive._service
    for cand in (f"{name}.mp4", f"{name}.webm", name):
        safe = cand.replace("'", "\\'")
        r = svc.files().list(
            q=f"name = '{safe}' and trashed = false",
            fields="files(id,name)", pageSize=1,
        ).execute().get("files", [])
        if r:
            _name2id[name] = r[0]["id"]
            return r[0]["id"]
    safe = name[:40].replace("'", "\\'")
    r = svc.files().list(
        q=f"name contains '{safe}' and trashed = false",
        fields="files(id,name)", pageSize=1,
    ).execute().get("files", [])
    _name2id[name] = r[0]["id"] if r else None
    return _name2id[name]


def _done_and_stats() -> tuple[set[str], Counter]:
    done: set[str] = set()
    per: Counter = Counter()
    for path in LABELS.glob("video_*.jsonl"):
        for line in path.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("emotion"):
                    done.add(r["video"])
                    per[r["emotion"]] += 1
    for path in LABELS.glob("video_skip_*.txt"):
        done.update(l.strip() for l in path.open(encoding="utf-8") if l.strip())
    return done, per


@app.get("/next")
def next_video(annotator: str) -> JSONResponse:
    """Wydaje wideo prawdopodobnie DEFICYTOWEJ emocji — kolejka sama się wyrównuje."""
    done, per = _done_and_stats()
    now = time.time()
    with _LOCK:
        for k in [k for k, t in _CLAIMS.items() if now - t > _CLAIM_TTL]:
            del _CLAIMS[k]
        payload = {"total": len(VIDEOS), "done": len(done)}
        avail = [v for v in VIDEOS if v not in done and v not in _CLAIMS]
        if not avail:
            return JSONResponse(payload)

        def priority(v: str):
            # Emocja spoza 6 docelowych (np. relaxed) -> na koniec. W obrębie
            # docelowych: najpierw ta, której zebrano najmniej (wyrównanie).
            emo = VIDEO_EMO.get(v)
            if emo not in EMOTIONS:
                return (2, 0)
            return (0 if per.get(emo, 0) < TARGET else 1, per.get(emo, 0))

        best = min(avail, key=priority)
        _CLAIMS[best] = now
        payload["video"] = best
        # Kandydat do wstępnego pobrania (NIE rezerwujemy — zostaje w kolejce),
        # klient grzeje nim cache serwera, żeby następne wideo ruszyło od razu.
        rest = [v for v in avail if v != best]
        if rest:
            payload["prefetch"] = min(rest, key=priority)
        return JSONResponse(payload)


def _ensure_cached(v: str) -> Path | None:
    """Zapewnia plik wideo w cache (pobiera z Drive raz). Zwraca ścieżkę lub None."""
    dst = CACHE / (v.replace("/", "_") + ".mp4")
    if not dst.is_file():
        with _dl_lock:
            if not dst.is_file():
                fid = _find_drive(v)
                if not fid:
                    return None
                tmp = dst.with_suffix(".part")
                _drive.download_file(fid, tmp)
                tmp.rename(dst)
    return dst


@app.get("/video")
def video(v: str) -> FileResponse:
    """Pobiera wideo z Drive (cache) i oddaje do odtwarzacza (obsługa seek)."""
    dst = _ensure_cached(v)
    if dst is None:
        return JSONResponse({"error": "not found on Drive"}, status_code=404)
    return FileResponse(dst, media_type="video/mp4")


@app.get("/warm")
def warm(v: str) -> JSONResponse:
    """Grzeje cache serwera (pobiera z Drive) BEZ wysyłania wideo do klienta."""
    return JSONResponse({"ok": _ensure_cached(v) is not None})


@app.get("/stats")
def stats() -> JSONResponse:
    _, per = _done_and_stats()
    return JSONResponse([{"emotion": e, "have": per.get(e, 0), "target": TARGET} for e in EMOTIONS])


@app.post("/save")
async def save(payload: dict) -> JSONResponse:
    annotator = payload["annotator"]
    LABELS.mkdir(parents=True, exist_ok=True)
    if not payload.get("emotion"):
        with (LABELS / f"video_skip_{annotator}.txt").open("a", encoding="utf-8") as h:
            h.write(payload["video"] + "\n")
        return JSONResponse({"ok": True, "skipped": True})
    record = {
        "video": payload["video"],
        "annotator": annotator,
        "emotion": payload["emotion"],
        "start_time": payload.get("start_time"),
        "end_time": payload.get("end_time"),
        "usable": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with (LABELS / f"video_{annotator}.jsonl").open("a", encoding="utf-8") as h:
        h.write(json.dumps(record, ensure_ascii=False) + "\n")
    return JSONResponse({"ok": True})


HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dog — wideo</title><style>
body{background:#111;color:#eee;font-family:system-ui;margin:0;padding-bottom:150px;text-align:center}
#hdr{padding:6px;font-size:14px}
#stats{padding:4px;font-size:13px;color:#bbb}
video{max-height:60vh;max-width:97vw;margin-top:6px;border-radius:8px;background:#000}
.bar{position:fixed;bottom:0;left:0;right:0;background:#1b1b1b;padding:8px}
button{font-size:15px;margin:3px;padding:9px 12px;border:0;border-radius:8px;background:#2a2a2a;color:#eee}
.mk{background:#345}.mk.set{background:#3c6;color:#000}
.emo.sel{background:#3a6}
#save{background:#2a7;color:#000;font-weight:bold}#save:disabled{opacity:.4}
.done{color:#5c6}.need{color:#e88}
</style></head><body>
<div id="stats"></div>
<div id="hdr"></div>
<video id="vid" controls playsinline preload="auto"></video>
<div style="font-size:11px;color:#666;padding:2px">клавиши: 1‑6 эмоция · q начало · w конец · r сброс · s skip · Enter сохранить</div>
<div class="bar" id="bar"></div>
<script>
const EMO=%EMO%;
let kto=new URLSearchParams(location.search).get("kto");
let cur=null,emo=null,startT=null,endT=null;
const vid=document.getElementById("vid"),bar=document.getElementById("bar");
function fmt(t){return t==null?"—":t.toFixed(2)+"с";}
function updateBar(){
 bar.innerHTML="";
 const b1=document.createElement("button");b1.className="mk"+(startT!=null?" set":"");b1.textContent="① начало "+fmt(startT);b1.onclick=()=>{startT=vid.currentTime;updateBar();};bar.appendChild(b1);
 const b2=document.createElement("button");b2.className="mk"+(endT!=null?" set":"");b2.textContent="② конец "+fmt(endT);b2.onclick=()=>{endT=vid.currentTime;updateBar();};bar.appendChild(b2);
 const rs=document.createElement("button");rs.textContent="✕ сброс";rs.onclick=()=>{startT=null;endT=null;updateBar();};bar.appendChild(rs);
 bar.appendChild(document.createElement("br"));
 EMO.forEach(e=>{const b=document.createElement("button");b.className="emo"+(emo===e[0]?" sel":"");b.textContent=e[1];b.onclick=()=>{emo=e[0];updateBar();};bar.appendChild(b);});
 bar.appendChild(document.createElement("br"));
 const sk=document.createElement("button");sk.textContent="skip";sk.onclick=()=>save(true);bar.appendChild(sk);
 const s=document.createElement("button");s.id="save";s.textContent="Сохранить →";s.disabled=!(emo&&startT!=null&&endT!=null);s.onclick=()=>save(false);bar.appendChild(s);
}
async function save(skip){if(!cur)return;
 await fetch("/save",{method:"POST",headers:{"Content-Type":"application/json","ngrok-skip-browser-warning":"1"},
  body:JSON.stringify(skip?{annotator:kto,video:cur,emotion:null}:{annotator:kto,video:cur,emotion:emo,start_time:startT,end_time:endT})});
 emo=null;startT=null;endT=null;load();stats();}
async function load(){const r=await fetch("/next?annotator="+kto,{headers:{"ngrok-skip-browser-warning":"1"}});const d=await r.json();
 document.getElementById("hdr").innerHTML=kto+" — видео "+d.done+" / "+d.total;
 if(!d.video){vid.style.display="none";document.getElementById("hdr").innerHTML="Готово! Все видео размечены.";bar.innerHTML="";return;}
 cur=d.video;vid.src="/video?v="+encodeURIComponent(cur);vid.load();vid.play().catch(()=>{});updateBar();
 if(d.prefetch){fetch("/warm?v="+encodeURIComponent(d.prefetch),{headers:{"ngrok-skip-browser-warning":"1"}}).catch(()=>{});}}
async function stats(){const r=await fetch("/stats",{headers:{"ngrok-skip-browser-warning":"1"}});const d=await r.json();
 let h="";for(const s of d){const left=Math.max(0,s.target-s.have);h+="<span class='"+(left===0?"done":"need")+"'>"+s.emotion+" "+s.have+"/"+s.target+"</span>&nbsp;&nbsp; ";}
 document.getElementById("stats").innerHTML=h;}
function pickUser(){document.getElementById("hdr").innerHTML="<b>Кто размечает?</b>";document.getElementById("stats").innerHTML="";vid.style.display="none";bar.innerHTML="";["danek","masha","anton","mafin"].forEach(n=>{const b=document.createElement("button");b.textContent=n;b.style.fontSize="19px";b.style.padding="14px 22px";b.onclick=()=>{kto=n;vid.style.display="";load();stats();};bar.appendChild(b);});}
document.addEventListener("keydown",e=>{
 if(!cur||!kto)return;const k=e.key.toLowerCase();
 if(k>="1"&&k<="9"){const i=+k-1;if(i<EMO.length){emo=EMO[i][0];updateBar();}}
 else if(k==="q"){startT=vid.currentTime;updateBar();}
 else if(k==="w"){endT=vid.currentTime;updateBar();}
 else if(k==="r"){startT=null;endT=null;updateBar();}
 else if(k==="s"){save(true);}
 else if(k==="enter"){if(emo&&startT!=null&&endT!=null)save(false);}});
if(kto){load();stats();}else{pickUser();}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    emo_js = json.dumps([[e, f"{e} · {EMO_PL[e]}"] for e in EMOTIONS])
    return HTML.replace("%EMO%", emo_js)


if __name__ == "__main__":
    print(f"Wideo w kolejce: {len(VIDEOS)}. http://localhost:{PORT}/", file=sys.stderr)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
