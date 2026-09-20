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
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
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
# fearful i surprise znów rozdzielone (na życzenie prowadzącego) — bez scalania
MERGE: dict[str, str] = {}

# Źródła: folder na Drive -> (pool, emocja lub None)
OTOBRANE = {"DataSet_neutral": "neutral", "DataSet_sad": "sad", "DataSet_happy": "happy",
            "new_angry_dogs": "angry", "new_surprised_dogs": "surprise",
            "new_happy_dogs": "happy", "angry_dogs_2": "angry", "angry_dogs_3": "angry",
            "dog_fearful_anton": "fearful", "dog_fearful_anton_2": "fearful",
            "surprised_dogs_mafin": "surprise",
            "neutral_dog_masha": "neutral", "angry_dogs_masha": "angry"}
SUROWE = ["tiktok_playlist_nareski"]

_drive = GoogleDriveUploader(GDRIVE_CREDENTIALS_PATH, GDRIVE_TOKEN_PATH, GDRIVE_FOLDER_ID)
_drive.authenticate()
_LOCK = Lock()
_CLAIMS: dict[str, float] = {}
_CLAIM_TTL = 300.0

# Ile wideo naprzód grzejemy w tle (żeby anotator nie czekał na pobranie z Drive).
PREFETCH_AHEAD = 5
# Osobny zamek per plik: dwa RÓŻNE wideo mogą pobierać się równolegle, ale to samo
# — tylko raz. Globalny zamek serializował wszystko i zabijał prefetch.
_locks_guard = Lock()
_dl_locks: dict[str, Lock] = {}
_prefetch_pool = ThreadPoolExecutor(max_workers=2)
_prefetching: set[str] = set()
# Klient Drive (httplib2) NIE jest bezpieczny wątkowo — współdzielony między
# wątkami się wywala (segfault). Każde sięgnięcie do sieci serializujemy.
_drive_lock = Lock()


def _lock_for(fid: str) -> Lock:
    with _locks_guard:
        return _dl_locks.setdefault(fid, Lock())


def _prefetch(fid: str) -> None:
    """Zleca pobranie wideo do cache w tle (bez duplikatów, bez blokowania /next)."""
    with _locks_guard:
        if (CACHE / f"{fid}.mp4").is_file() or fid in _prefetching:
            return
        _prefetching.add(fid)

    def _job() -> None:
        try:
            _ensure_cached(fid)
        finally:
            with _locks_guard:
                _prefetching.discard(fid)

    _prefetch_pool.submit(_job)

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
                emo = MERGE.get(r.get("emotion"), r.get("emotion"))
                if emo:
                    now[emo] += 1
                if r.get("pool") == "otobrane" and emo:
                    otob[emo] += 1
                elif r.get("pool") == "surowe":
                    surowe_done += 1
    for path in LABELS.glob("select_skip_*.txt"):
        done.update(x.strip() for x in path.open(encoding="utf-8") if x.strip())
    return done, now, otob, surowe_done


def _rated() -> dict[str, dict]:
    """fid -> {name, raters:[kto]} po WSZYSTKICH ocenach (pierwsze + podwójne)."""
    out: dict[str, dict] = {}
    for path in list(LABELS.glob("select_*.jsonl")) + list(LABELS.glob("double_*.jsonl")):
        for line in path.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("emotion"):
                    d = out.setdefault(r["fid"], {"name": r.get("video"), "raters": []})
                    if r.get("annotator") not in d["raters"]:
                        d["raters"].append(r.get("annotator"))
    return out


@app.get("/next")
def next_video(pool: str, annotator: str, emotion: str | None = None) -> JSONResponse:
    """Wydaje następne wideo z kolejki (Odebrane można zawęzić do jednej emocji)."""
    now = time.time()
    # Tryb podwójnej oceny: wideo ocenione już przez KOGOŚ INNEGO, na ślepo.
    if pool == "double":
        rated = _rated()
        with _LOCK:
            for k in [k for k, t in _CLAIMS.items() if now - t > _CLAIM_TTL]:
                del _CLAIMS[k]
            payload = {"total": len(rated), "pool": "double"}
            for fid, info in rated.items():
                if annotator in info["raters"] or fid in _CLAIMS:
                    continue
                _CLAIMS[fid] = now
                payload.update({"fid": fid, "name": info["name"], "emotion": None})
                return JSONResponse(payload)
            return JSONResponse(payload)

    items = POOLS.get(pool, [])
    if pool == "otobrane" and emotion:
        items = [it for it in items if it["emotion"] == emotion]
    done, *_ = _scan()
    with _LOCK:
        for k in [k for k, t in _CLAIMS.items() if now - t > _CLAIM_TTL]:
            del _CLAIMS[k]
        payload = {"total": len(items), "pool": pool}
        picked: dict | None = None
        ahead: list[str] = []
        for it in items:
            if it["fid"] in done or it["fid"] in _CLAIMS:
                continue
            if picked is None:
                _CLAIMS[it["fid"]] = now
                picked = it
                continue
            ahead.append(it["fid"])
            if len(ahead) >= PREFETCH_AHEAD:
                break
    if picked:
        payload.update({"fid": picked["fid"], "name": picked["name"],
                        "emotion": picked["emotion"]})
        if ahead:
            payload["prefetch"] = ahead[0]
        for f in ahead:  # grzejemy w tle następne wideo, zanim anotator do nich dojdzie
            _prefetch(f)
    return JSONResponse(payload)


def _video_codec(path: Path) -> str:
    """Kodek wideo pliku (np. h264, av1, vp9) — przez ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1",
             str(path)],
            capture_output=True, text=True, timeout=30,
        )
        return r.stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


_VIDEO_EXT = (".mov", ".mp4", ".webm", ".m4v", ".mkv", ".avi")


def _unwrap_if_zip(raw: Path, fid: str) -> Path | None:
    """Część pobrań z Envato to ZIP z wideo w środku (a nie sam plik). Rozpakowuje
    wideo ze środka; jeśli to nie ZIP — zwraca raw bez zmian. None gdy ZIP bez wideo.
    """
    with raw.open("rb") as fh:
        if fh.read(4) != b"PK\x03\x04":
            return raw
    try:
        with zipfile.ZipFile(raw) as z:
            vids = [n for n in z.namelist() if n.lower().endswith(_VIDEO_EXT)]
            if not vids:
                return None
            inner = CACHE / f"{fid}.inner"
            with z.open(vids[0]) as s, inner.open("wb") as d:
                shutil.copyfileobj(s, d)
    except Exception:  # noqa: BLE001
        return None
    raw.unlink(missing_ok=True)
    return inner


def _ensure_cached(fid: str) -> Path | None:
    """Pobiera plik z Drive po ID (raz) do cache; transkoduje do H.264 jeśli trzeba.

    Safari/iOS nie odtwarza AV1/VP9 — dlatego wszystko, co nie jest H.264,
    przekodowujemy, żeby grało na telefonach zespołu.
    """
    dst = CACHE / f"{fid}.mp4"
    if dst.is_file():
        return dst
    with _lock_for(fid):
        if dst.is_file():
            return dst
        raw = CACHE / f"{fid}.raw"
        # Drive bywa zrywa połączenie (SSL EOF) — bez ponowienia dawało to 404
        # i przekreślony player na telefonie. Trzy próby wystarczają.
        for attempt in range(3):
            try:
                with _drive_lock:
                    _drive.download_file(fid, raw)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"DL FAIL {fid} próba {attempt}: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
                raw.unlink(missing_ok=True)
                if attempt == 2:
                    return None
                time.sleep(1.5 * (attempt + 1))
        # Rozpakuj, jeśli Envato oddało wideo w ZIP-ie zamiast samego pliku.
        src = _unwrap_if_zip(raw, fid)
        if src is None:
            print(f"UNZIP FAIL {fid}: ZIP bez wideo", file=sys.stderr, flush=True)
            raw.unlink(missing_ok=True)
            return None
        # Zawsze przekodowujemy do H.264 720p bez dźwięku: pliki z Envato to h264
        # 100-200 MB, a przez tunel cloudflare taki plik ładuje się w nieskończoność.
        # Do wyboru momentu (start/end) 720p w zupełności wystarcza.
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(src), "-vf", "scale='min(1280,iw)':-2",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-an",
                 "-movflags", "+faststart", str(dst)],
                capture_output=True, timeout=300, check=True,
            )
        except Exception as exc:  # noqa: BLE001
            err = getattr(exc, "stderr", b"")
            tail = err[-300:].decode("utf-8", "replace") if isinstance(err, bytes) else ""
            print(f"FFMPEG FAIL {fid}: {type(exc).__name__}: {exc} | {tail}",
                  file=sys.stderr, flush=True)
            src.unlink(missing_ok=True)
            dst.unlink(missing_ok=True)
            return None
        src.unlink(missing_ok=True)
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


TARGET = 250  # cel: tyle wideo na każdą emocję


def _totals() -> Counter:
    """Łącznie ręcznie oznaczonych wideo na emocję: stare (video_*) + nowe (select_*)."""
    tot: Counter = Counter()
    old: dict[str, str] = {}
    for p in LABELS.glob("video_*.jsonl"):
        for line in p.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("emotion"):
                    old[r["video"]] = r["emotion"]
    new: dict[str, str] = {}
    for p in LABELS.glob("select_*.jsonl"):
        for line in p.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r.get("emotion"):
                    new[r["fid"]] = r["emotion"]
    for e in list(old.values()) + list(new.values()):
        tot[MERGE.get(e, e)] += 1
    return tot


@app.get("/stats")
def stats() -> JSONResponse:
    tot = _totals()
    double = sum(1 for p in LABELS.glob("double_*.jsonl")
                 for line in p.open(encoding="utf-8") if line.strip())
    return JSONResponse({
        "totals": {e: tot.get(e, 0) for e in EMOTIONS},
        "target": TARGET,
        "double": double,
    })


@app.post("/save")
async def save(payload: dict) -> JSONResponse:
    LABELS.mkdir(parents=True, exist_ok=True)
    annotator = payload["annotator"]
    if payload.get("skip"):
        with (LABELS / f"select_skip_{annotator}.txt").open("a", encoding="utf-8") as h:
            h.write(payload["fid"] + "\n")
        return JSONResponse({"ok": True, "skipped": True})
    # Tryb podwójnej oceny -> osobny plik double_<kto>.jsonl (sama emocja)
    if payload.get("pool") == "double":
        rec = {"fid": payload["fid"], "video": payload.get("name"),
               "emotion": payload["emotion"], "annotator": annotator, "pool": "double",
               "timestamp": datetime.now(timezone.utc).isoformat()}
        with (LABELS / f"double_{annotator}.jsonl").open("a", encoding="utf-8") as h:
            h.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return JSONResponse({"ok": True})
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
const OTOB=[["neutral","нейтр."],["sad","грусть"],["happy","радость"],["angry","злость"],["surprise","удивление"],["fearful","страх"]];
let kto=new URLSearchParams(location.search).get("kto");
let pool="otobrane",otobEmo="neutral",cur=null,emo=null,startT=null,endT=null;
const vid=document.getElementById("vid"),bar=document.getElementById("bar");
function fmt(t){return t==null?"—":t.toFixed(2)+"с";}
function poolbar(){const b=document.getElementById("poolbar");b.innerHTML="";
 [["otobrane","✅ Отобранное"],["surowe","🎞 Сырьё"],["double","🔁 Двойная"]].forEach(p=>{const x=document.createElement("button");x.className="pool"+(pool===p[0]?" sel":"");x.textContent=p[1];x.onclick=()=>{pool=p[0];poolbar();reset();load();stats();};b.appendChild(x);});
 if(pool==="otobrane"){b.appendChild(document.createElement("br"));OTOB.forEach(e=>{const x=document.createElement("button");x.className="emo"+(otobEmo===e[0]?" sel":"");x.textContent=e[1];x.onclick=()=>{otobEmo=e[0];poolbar();reset();load();stats();};b.appendChild(x);});}}
function reset(){emo=null;startT=null;endT=null;}
function updateBar(){
 bar.innerHTML="";
 if(pool!=="double"){
  const b1=document.createElement("button");b1.className="mk"+(startT!=null?" set":"");b1.textContent="① начало "+fmt(startT);b1.onclick=()=>{startT=vid.currentTime;updateBar();};bar.appendChild(b1);
  const b2=document.createElement("button");b2.className="mk"+(endT!=null?" set":"");b2.textContent="② конец "+fmt(endT);b2.onclick=()=>{endT=vid.currentTime;updateBar();};bar.appendChild(b2);
  const rs=document.createElement("button");rs.textContent="✕ сброс";rs.onclick=()=>{startT=null;endT=null;updateBar();};bar.appendChild(rs);
  bar.appendChild(document.createElement("br"));
 }
 if(cur&&cur.emotion===null){EMO.forEach(e=>{const b=document.createElement("button");b.className="emo"+(emo===e[0]?" sel":"");b.textContent=e[1];b.onclick=()=>{emo=e[0];updateBar();};bar.appendChild(b);});bar.appendChild(document.createElement("br"));}
 const sk=document.createElement("button");sk.textContent="skip";sk.onclick=()=>save(true);bar.appendChild(sk);
 const need_emo=cur&&cur.emotion===null;
 const ok=pool==="double"?!!emo:((need_emo?emo:true)&&startT!=null&&endT!=null);
 const s=document.createElement("button");s.id="save";s.textContent="Сохранить →";s.disabled=!ok;s.onclick=()=>save(false);bar.appendChild(s);
}
async function save(skip){if(!cur)return;
 const body=skip?{annotator:kto,fid:cur.fid,skip:true}:(pool==="double"?{annotator:kto,fid:cur.fid,name:cur.name,pool:"double",emotion:emo}:{annotator:kto,fid:cur.fid,name:cur.name,pool:pool,emotion:cur.emotion||emo,start_time:startT,end_time:endT});
 await fetch("/save",{method:"POST",headers:{"Content-Type":"application/json","ngrok-skip-browser-warning":"1"},body:JSON.stringify(body)});
 reset();load();stats();}
async function load(){const u="/next?pool="+pool+"&annotator="+kto+(pool==="otobrane"?"&emotion="+otobEmo:"");const r=await fetch(u,{headers:{"ngrok-skip-browser-warning":"1"}});const d=await r.json();
 if(!d.fid){vid.style.display="none";document.getElementById("hdr").innerHTML="Готово в этом режиме.";bar.innerHTML="";return;}
 vid.style.display="";cur=d;
 document.getElementById("hdr").innerHTML=kto+" · "+(pool==="double"?"<b>🔁 оцени эмоцию вслепую</b>":(pool==="otobrane"?("эмоция: <b>"+d.emotion+"</b>"):"<b>выбери эмоцию</b>"))+" · "+d.name.slice(0,40);
 vid.src="/video?fid="+d.fid;vid.load();vid.play().catch(()=>{});updateBar();
 if(d.prefetch)fetch("/warm?fid="+d.prefetch,{headers:{"ngrok-skip-browser-warning":"1"}}).catch(()=>{});}
async function stats(){const r=await fetch("/stats",{headers:{"ngrok-skip-browser-warning":"1"}});const d=await r.json();
 const t=d.totals||{};const T=d.target||250;let h="";
 for(const e of EMO){const n=t[e[0]]||0;h+="<span class='"+(n>=T?"done":"need")+"'>"+e[0]+" "+n+"/"+T+"</span>&nbsp;&nbsp; ";}
 if(pool==="double")h+="<span style='color:#7bf'>🔁 вторых оценок: "+(d.double||0)+"</span>";
 document.getElementById("prev").innerHTML="";
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
