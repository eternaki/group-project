#!/usr/bin/env python3
"""
Wybór kadrów (początek/koniec emocji) — dwa źródła, prosto z Google Drive.

Dwa tryby (przyciski):
- **Odebrane** (`otobrane`): foldery DataSet_neutral/sad/happy — emocja znana z
  folderu, anotator zaznacza TYLKO kadry (początek/koniec).
- **Surowe** (`surowe`): dog_tv_24_7_nareski + tiktok_playlist_nareski — emocja
  nieznana, anotator wybiera emocję ORAZ kadry.

Serwer pobiera plik z Drive po ID (cache) i oddaje do odtwarzacza. Werdykty ->
data/labels/dataset_final/select_<kto>.jsonl. Kolejka globalna z rezerwacją.
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
LABELS = REPO / "data" / "labels" / "dataset_final"
CACHE = Path("/tmp/dogvids")
CACHE.mkdir(exist_ok=True)
PORT = 8001

EMOTIONS = ["neutral", "sad", "happy", "surprise", "angry", "fearful"]
EMO_PL = {"neutral": "нейтрально", "sad": "грусть", "happy": "радость",
          "surprise": "удивление", "angry": "злость", "fearful": "страх"}

# Źródła: folder na Drive -> (pool, emocja lub None)
OTOBRANE = {"DataSet_neutral": "neutral", "DataSet_sad": "sad", "DataSet_happy": "happy"}
SUROWE = ["dog_tv_24_7_nareski", "tiktok_playlist_nareski"]

_drive = GoogleDriveUploader(GDRIVE_CREDENTIALS_PATH, GDRIVE_TOKEN_PATH, GDRIVE_FOLDER_ID)
_drive.authenticate()
_dl_lock = Lock()
_LOCK = Lock()
_CLAIMS: dict[str, float] = {}
_CLAIM_TTL = 300.0

app = FastAPI()


def _find_folder(name: str) -> str | None:
    """ID podfolderu o danej nazwie w folderze głównym."""
    q = (f"name = '{name}' and '{GDRIVE_FOLDER_ID}' in parents "
         "and mimeType = 'application/vnd.google-apps.folder' and trashed = false")
    r = _drive._service.files().list(q=q, fields="files(id)", pageSize=1).execute().get("files", [])
    return r[0]["id"] if r else None


def _build_pools() -> dict[str, list[dict]]:
    """Buduje dwie kolejki z Drive: {fid, name, emotion|None}."""
    pools: dict[str, list[dict]] = {"otobrane": [], "surowe": []}
    for folder, emotion in OTOBRANE.items():
        fid = _find_folder(folder)
        if not fid:
            continue
        for f in _drive.list_files(fid, fields="id,name"):
            pools["otobrane"].append({"fid": f["id"], "name": f["name"], "emotion": emotion})
    for folder in SUROWE:
        fid = _find_folder(folder)
        if not fid:
            continue
        for f in _drive.list_files(fid, fields="id,name"):
            pools["surowe"].append({"fid": f["id"], "name": f["name"], "emotion": None})
    return pools


POOLS = _build_pools()
print(f"Odebrane: {len(POOLS['otobrane'])} | Surowe: {len(POOLS['surowe'])}", file=sys.stderr)

# Ile wideo w folderach Odebrane per emocja (do liczników done/total)
OTOB_TOTAL = Counter(it["emotion"] for it in POOLS["otobrane"])


def _prev_stats() -> dict:
    """Poprzednie statystyki: ile wideo przetworzono i dawna rozmowa start/koniec."""
    full = REPO / "data" / "dataset_final" / "work" / "annotations_full.json"
    processed = 0
    if full.is_file():
        coco = json.loads(full.read_text(encoding="utf-8"))
        processed = len({(i.get("source_video") or i["file_name"].split("/")[-2])
                         for i in coco["images"]})
    labeled: Counter = Counter()
    for path in LABELS.glob("video_*.jsonl"):  # dawne metki start/koniec (302 wideo)
        for line in path.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("emotion"):
                    labeled[r["emotion"]] += 1
    return {"processed": processed, "labeled": dict(labeled)}


PREV = _prev_stats()


def _scan() -> tuple[set[str], Counter, Counter, int]:
    """(fid zrobione, licznik emocji ogółem, licznik Odebrane, ile w Surowe)."""
    done: set[str] = set()
    now: Counter = Counter()
    otob: Counter = Counter()
    surowe_done = 0
    for path in LABELS.glob("select_*.jsonl"):
        for line in path.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done.add(r["fid"])
                if r.get("emotion"):
                    now[r["emotion"]] += 1
                if r.get("pool") == "otobrane" and r.get("emotion"):
                    otob[r["emotion"]] += 1
                elif r.get("pool") == "surowe":
                    surowe_done += 1
    for path in LABELS.glob("select_skip_*.txt"):
        done.update(x.strip() for x in path.open(encoding="utf-8") if x.strip())
    return done, now, otob, surowe_done


@app.get("/next")
def next_video(pool: str, annotator: str, emotion: str | None = None) -> JSONResponse:
    """Wydaje następne wideo z kolejki (Odebrane można zawęzić do jednej emocji)."""
    items = POOLS.get(pool, [])
    if pool == "otobrane" and emotion:
        items = [it for it in items if it["emotion"] == emotion]
    done, *_ = _scan()
    now = time.time()
    with _LOCK:
        for k in [k for k, t in _CLAIMS.items() if now - t > _CLAIM_TTL]:
            del _CLAIMS[k]
        payload = {"total": len(items), "pool": pool}
        for it in items:
            if it["fid"] in done or it["fid"] in _CLAIMS:
                continue
            _CLAIMS[it["fid"]] = now
            payload.update({"fid": it["fid"], "name": it["name"], "emotion": it["emotion"]})
            return JSONResponse(payload)
        return JSONResponse(payload)


def _ensure_cached(fid: str) -> Path | None:
    """Pobiera plik z Drive po ID (raz) do cache."""
    dst = CACHE / f"{fid}.mp4"
    if not dst.is_file():
        with _dl_lock:
            if not dst.is_file():
                tmp = dst.with_suffix(".part")
                try:
                    _drive.download_file(fid, tmp)
                    tmp.rename(dst)
                except Exception:  # noqa: BLE001
                    return None
    return dst


@app.get("/video")
def video(fid: str) -> FileResponse:
    """Oddaje wideo z Drive (cache) do odtwarzacza."""
    dst = _ensure_cached(fid)
    if dst is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(dst, media_type="video/mp4")


@app.get("/warm")
def warm(fid: str) -> JSONResponse:
    """Grzeje cache serwera bez wysyłania wideo do klienta."""
    return JSONResponse({"ok": _ensure_cached(fid) is not None})


@app.get("/stats")
def stats() -> JSONResponse:
    _, now, otob, surowe_done = _scan()
    return JSONResponse({
        "prev": PREV,
        "now": {e: now.get(e, 0) for e in EMOTIONS},
        "otobrane": {e: {"done": otob.get(e, 0), "total": OTOB_TOTAL.get(e, 0)}
                     for e in OTOB_TOTAL},
        "surowe": {"done": surowe_done, "total": len(POOLS["surowe"])},
    })


@app.post("/save")
async def save(payload: dict) -> JSONResponse:
    LABELS.mkdir(parents=True, exist_ok=True)
    annotator = payload["annotator"]
    if payload.get("skip"):
        with (LABELS / f"select_skip_{annotator}.txt").open("a", encoding="utf-8") as h:
            h.write(payload["fid"] + "\n")
        return JSONResponse({"ok": True, "skipped": True})
    record = {
        "fid": payload["fid"],
        "video": payload["name"],
        "pool": payload["pool"],
        "emotion": payload["emotion"],
        "start_time": payload.get("start_time"),
        "end_time": payload.get("end_time"),
        "usable": True,
        "annotator": annotator,
        "label_source": "human_verified",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with (LABELS / f"select_{annotator}.jsonl").open("a", encoding="utf-8") as h:
        h.write(json.dumps(record, ensure_ascii=False) + "\n")
    return JSONResponse({"ok": True})


HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Кадры</title><style>
body{background:#111;color:#eee;font-family:system-ui;margin:0;padding-bottom:160px;text-align:center}
#stats{padding:4px;font-size:13px;color:#bbb}
#top{padding:6px;position:sticky;top:0;background:#111;z-index:2}
video{max-height:56vh;max-width:97vw;margin-top:6px;border-radius:8px;background:#000}
.bar{position:fixed;bottom:0;left:0;right:0;background:#1b1b1b;padding:8px}
button{font-size:15px;margin:3px;padding:9px 12px;border:0;border-radius:8px;background:#2a2a2a;color:#eee}
.pool.sel{background:#46a;color:#fff}
.mk{background:#345}.mk.set{background:#3c6;color:#000}
.emo.sel{background:#3a6;color:#000}
#save{background:#2a7;color:#000;font-weight:bold}#save:disabled{opacity:.4}
</style></head><body>
<div id="prev" style="font-size:12px;color:#888;padding:3px"></div>
<div id="stats"></div>
<div id="top"><div id="poolbar"></div></div>
<div id="hdr" style="font-size:13px;color:#999;padding:2px"></div>
<video id="vid" controls playsinline preload="auto"></video>
<div style="font-size:11px;color:#666">клавиши: q начало · w конец · r сброс · s skip · Enter сохранить · 1-6 эмоция</div>
<div class="bar" id="bar"></div>
<script>
const EMO=%EMO%;
const OTOB=[["neutral","нейтр."],["sad","грусть"],["happy","радость"]];
let kto=new URLSearchParams(location.search).get("kto");
let pool="otobrane",otobEmo="neutral",cur=null,emo=null,startT=null,endT=null;
const vid=document.getElementById("vid"),bar=document.getElementById("bar");
function fmt(t){return t==null?"—":t.toFixed(2)+"с";}
function poolbar(){const b=document.getElementById("poolbar");b.innerHTML="";
 [["otobrane","✅ Отобранное"],["surowe","🎞 Сырьё"]].forEach(p=>{const x=document.createElement("button");x.className="pool"+(pool===p[0]?" sel":"");x.textContent=p[1];x.onclick=()=>{pool=p[0];poolbar();reset();load();stats();};b.appendChild(x);});
 if(pool==="otobrane"){b.appendChild(document.createElement("br"));OTOB.forEach(e=>{const x=document.createElement("button");x.className="emo"+(otobEmo===e[0]?" sel":"");x.textContent=e[1];x.onclick=()=>{otobEmo=e[0];poolbar();reset();load();stats();};b.appendChild(x);});}}
function reset(){emo=null;startT=null;endT=null;}
function updateBar(){
 bar.innerHTML="";
 const b1=document.createElement("button");b1.className="mk"+(startT!=null?" set":"");b1.textContent="① начало "+fmt(startT);b1.onclick=()=>{startT=vid.currentTime;updateBar();};bar.appendChild(b1);
 const b2=document.createElement("button");b2.className="mk"+(endT!=null?" set":"");b2.textContent="② конец "+fmt(endT);b2.onclick=()=>{endT=vid.currentTime;updateBar();};bar.appendChild(b2);
 const rs=document.createElement("button");rs.textContent="✕ сброс";rs.onclick=()=>{startT=null;endT=null;updateBar();};bar.appendChild(rs);
 bar.appendChild(document.createElement("br"));
 if(cur&&cur.emotion===null){EMO.forEach(e=>{const b=document.createElement("button");b.className="emo"+(emo===e[0]?" sel":"");b.textContent=e[1];b.onclick=()=>{emo=e[0];updateBar();};bar.appendChild(b);});bar.appendChild(document.createElement("br"));}
 const sk=document.createElement("button");sk.textContent="skip";sk.onclick=()=>save(true);bar.appendChild(sk);
 const need_emo=cur&&cur.emotion===null;
 const s=document.createElement("button");s.id="save";s.textContent="Сохранить →";s.disabled=!((need_emo?emo:true)&&startT!=null&&endT!=null);s.onclick=()=>save(false);bar.appendChild(s);
}
async function save(skip){if(!cur)return;
 const body=skip?{annotator:kto,fid:cur.fid,skip:true}:{annotator:kto,fid:cur.fid,name:cur.name,pool:pool,emotion:cur.emotion||emo,start_time:startT,end_time:endT};
 await fetch("/save",{method:"POST",headers:{"Content-Type":"application/json","ngrok-skip-browser-warning":"1"},body:JSON.stringify(body)});
 reset();load();stats();}
async function load(){const u="/next?pool="+pool+"&annotator="+kto+(pool==="otobrane"?"&emotion="+otobEmo:"");const r=await fetch(u,{headers:{"ngrok-skip-browser-warning":"1"}});const d=await r.json();
 if(!d.fid){vid.style.display="none";document.getElementById("hdr").innerHTML="Готово в этом режиме.";bar.innerHTML="";return;}
 vid.style.display="";cur=d;
 document.getElementById("hdr").innerHTML=kto+" · "+(pool==="otobrane"?("эмоция: <b>"+d.emotion+"</b>"):"<b>выбери эмоцию</b>")+" · "+d.name.slice(0,40);
 vid.src="/video?fid="+d.fid;vid.load();vid.play().catch(()=>{});updateBar();
 if(d.prefetch)fetch("/warm?fid="+d.prefetch,{headers:{"ngrok-skip-browser-warning":"1"}}).catch(()=>{});}
async function stats(){const r=await fetch("/stats",{headers:{"ngrok-skip-browser-warning":"1"}});const d=await r.json();
 const p=d.prev||{};let pv="Ранее обработано: <b>"+(p.processed||0)+"</b> видео";const lab=p.labeled||{};const parts=[];for(const e of EMO){if(lab[e[0]])parts.push(e[0]+" "+lab[e[0]]);}if(parts.length)pv+=" · старая разметка: "+parts.join(", ");document.getElementById("prev").innerHTML=pv;
 let h="";
 if(pool==="otobrane"){const o=d.otobrane||{};for(const e of OTOB){const x=o[e[0]]||{done:0,total:0};h+="<span class='"+(x.total>0&&x.done>=x.total?"done":"need")+"'>"+e[0]+" "+x.done+"/"+x.total+"</span>&nbsp;&nbsp; ";}}
 else{const s=d.surowe||{done:0,total:0};h="🎞 Сырьё: размечено <b>"+s.done+"</b> / "+s.total;}
 document.getElementById("stats").innerHTML=h;}
document.addEventListener("keydown",e=>{if(!cur||!kto)return;const k=e.key.toLowerCase();
 if(k>="1"&&k<="9"){const i=+k-1;if(cur.emotion===null&&i<EMO.length){emo=EMO[i][0];updateBar();}}
 else if(k==="q"){startT=vid.currentTime;updateBar();}else if(k==="w"){endT=vid.currentTime;updateBar();}
 else if(k==="r"){startT=null;endT=null;updateBar();}else if(k==="s"){save(true);}
 else if(k==="enter"){const s=document.getElementById("save");if(s&&!s.disabled)save(false);}});
function pickUser(){document.getElementById("hdr").innerHTML="<b>Кто размечает?</b>";document.getElementById("stats").innerHTML="";vid.style.display="none";document.getElementById("poolbar").innerHTML="";bar.innerHTML="";["danek","masha","anton","mafin"].forEach(n=>{const b=document.createElement("button");b.textContent=n;b.style.fontSize="19px";b.style.padding="14px 22px";b.onclick=()=>{kto=n;poolbar();load();stats();};bar.appendChild(b);});}
if(kto){poolbar();load();stats();}else{pickUser();}
setInterval(()=>{if(kto)stats();},5000);
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    emo_js = json.dumps([[e, f"{e} · {EMO_PL[e]}"] for e in EMOTIONS])
    return HTML.replace("%EMO%", emo_js)


if __name__ == "__main__":
    print(f"Wybór kadrów: http://localhost:{PORT}/", file=sys.stderr)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
