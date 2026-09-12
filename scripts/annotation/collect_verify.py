#!/usr/bin/env python3
"""
Zbieranie wideo z YouTube z ręczną weryfikacją i wgrywaniem na Google Drive.

Człowiek wpisuje zapytanie (np. "growling dog"), przegląda kandydatów w
osadzonym odtwarzaczu YouTube (bez pobierania — szybko) i decyduje
✓ bierzemy / ✗ mijamy. Wzięte wideo pobierane jest przez yt-dlp i wgrywane do
folderu `DataSet_<emocja>` na Drive, a link źródłowy + licencja zapisywane.

- pokazujemy tylko krótkie klipy (≤ MAX_DURATION s),
- deduplikacja po ID YouTube działa GLOBALNIE: pomijamy też to, co już mamy
  (stary zbiór — youtube_existing.json), więc nie ściągamy dubli.
"""

import json
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
import yt_dlp
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from scripts.download.tiktok.config import (
    GDRIVE_CREDENTIALS_PATH,
    GDRIVE_FOLDER_ID,
    GDRIVE_TOKEN_PATH,
)
from scripts.download.tiktok.drive_uploader import GoogleDriveUploader

REPO = Path(__file__).resolve().parent.parent.parent
STATE_DIR = REPO / "data" / "collection" / "new"
STATE_DIR.mkdir(parents=True, exist_ok=True)
SEEN_FILE = STATE_DIR / "seen_yt.json"
EXISTING_FILE = STATE_DIR / "youtube_existing.json"
TMP = Path("/tmp/yt_dl")
TMP.mkdir(exist_ok=True)
PORT = 8002

EMOTIONS = ["neutral", "sad", "happy", "surprise", "angry", "fearful"]
EMO_PL = {"neutral": "нейтрально", "sad": "грусть", "happy": "радость",
          "surprise": "удивление", "angry": "злость", "fearful": "страх"}
MAX_DURATION = 45  # sekundy — dłuższych klipów nie pokazujemy
SEARCH_N = 50      # ilu kandydatów pobrać na zapytanie

# Pobieranie z YouTube (jak w download_videos.py): node do ekstrakcji, cookies z Chrome
JS_RUNTIMES = {"node": {"path": None}}
COOKIES_FROM_BROWSER: tuple = ("chrome", None, None, None)
FORMAT = "best[height<=720][ext=mp4][acodec!=none][vcodec!=none]/18/best[ext=mp4]/best"

socket.setdefaulttimeout(180)
app = FastAPI()
_LOCK = threading.Lock()
_DRIVE_LOCK = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=1)  # httplib2 (Drive) NIE jest wątkowo bezpieczny
_UPLOAD_RETRIES = 3
_inflight = 0

_seen: set[str] = set()
if SEEN_FILE.is_file():
    _seen |= set(json.loads(SEEN_FILE.read_text()))
if EXISTING_FILE.is_file():
    _seen |= set(json.loads(EXISTING_FILE.read_text()))

_folder_ids: dict[str, str] = {}
_drive = GoogleDriveUploader(GDRIVE_CREDENTIALS_PATH, GDRIVE_TOKEN_PATH, GDRIVE_FOLDER_ID)
_drive.authenticate()


def _save_seen() -> None:
    SEEN_FILE.write_text(json.dumps(sorted(_seen)))


def _folder_for(emotion: str) -> str:
    """ID folderu DataSet_<emocja> na Drive (tworzy raz). Woła się pod _DRIVE_LOCK."""
    if emotion not in _folder_ids:
        _folder_ids[emotion] = _drive.ensure_folder(f"DataSet_{emotion}", GDRIVE_FOLDER_ID)
    return _folder_ids[emotion]


def _kept_counts() -> dict[str, int]:
    counts = {e: 0 for e in EMOTIONS}
    for e in EMOTIONS:
        path = STATE_DIR / f"{e}.jsonl"
        if path.is_file():
            counts[e] = sum(1 for line in path.open() if line.strip())
    return counts


@app.get("/search")
def search(q: str, emotion: str) -> JSONResponse:
    """Kandydaci z YouTube dla zapytania: krótkie klipy, bez już widzianych."""
    try:
        proc = subprocess.run(
            ["yt-dlp", "--flat-playlist", "--dump-json", f"ytsearch{SEARCH_N}:{q}"],
            capture_output=True, text=True, timeout=90,
        )
    except subprocess.TimeoutExpired:
        return JSONResponse({"error": "timeout"}, status_code=504)
    out = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        try:
            v = json.loads(line)
        except json.JSONDecodeError:
            continue
        vid = v.get("id")
        dur = v.get("duration")
        if not vid or vid in _seen:
            continue
        if dur is not None and dur > MAX_DURATION:
            continue
        out.append({
            "id": vid,
            "title": (v.get("title") or "")[:90],
            "duration": dur,
            "uploader": v.get("uploader") or v.get("channel"),
            "url": f"https://www.youtube.com/watch?v={vid}",
        })
    return JSONResponse({"videos": out})


def _download(vid: str) -> tuple[Path | None, str | None]:
    """Pobiera wideo YouTube do TMP. Zwraca (ścieżka, licencja) lub (None, None)."""
    opts = {
        "format": FORMAT,
        "outtmpl": str(TMP / "yt_%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "js_runtimes": JS_RUNTIMES,
        "cookiesfrombrowser": COOKIES_FROM_BROWSER,
        "remote_components": ["ejs:github"],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=True)
    path = TMP / f"yt_{vid}.mp4"
    return (path if path.is_file() else None), info.get("license")


def _fetch_and_upload(video: dict, emotion: str) -> None:
    """W jednym wątku: pobiera z YouTube i wgrywa na Drive; zwraca wideo przy porażce."""
    global _inflight
    vid = video["id"]
    local = TMP / f"yt_{vid}.mp4"
    try:
        for attempt in range(1, _UPLOAD_RETRIES + 1):
            try:
                path, license_ = _download(vid)
                if path is None:
                    raise RuntimeError("pobranie nie dało pliku mp4")
                with _DRIVE_LOCK:
                    _drive.upload_file(
                        path, remote_name=f"yt_{vid}.mp4",
                        folder_id=_folder_for(emotion), resumable=False,
                    )
                record = {
                    "id": vid,
                    "source_video": f"yt_{vid}",
                    "emotion": emotion,
                    "platform": "YouTube",
                    "license": license_ or "YouTube — fragment badawczy (TDM)",
                    "link": video["url"],
                    "query": video.get("query"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                with _LOCK:
                    with (STATE_DIR / f"{emotion}.jsonl").open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                return
            except Exception as exc:  # noqa: BLE001 — ponawiamy, na końcu zwracamy wideo
                print(f"  ! {vid} próba {attempt}/{_UPLOAD_RETRIES}: {exc}", file=sys.stderr)
                local.unlink(missing_ok=True)
                time.sleep(2 * attempt)
        with _LOCK:
            _seen.discard(vid)
            _save_seen()
        print(f"  ! {vid} nie wgrane po {_UPLOAD_RETRIES} próbach — zwrócone", file=sys.stderr)
    finally:
        local.unlink(missing_ok=True)
        with _LOCK:
            _inflight -= 1


@app.post("/keep")
async def keep(payload: dict) -> JSONResponse:
    """Bierzemy wideo: dedup + pobranie/wgranie na Drive w tle."""
    global _inflight
    vid = str(payload["id"])
    emotion = payload["emotion"]
    with _LOCK:
        if vid in _seen:
            return JSONResponse({"ok": True, "dup": True})
        _seen.add(vid)
        _save_seen()
        _inflight += 1
    _pool.submit(_fetch_and_upload, payload, emotion)
    return JSONResponse({"ok": True})


@app.post("/skip")
async def skip(payload: dict) -> JSONResponse:
    """Mijamy wideo: zapamiętujemy ID, by nie wróciło."""
    with _LOCK:
        _seen.add(str(payload["id"]))
        _save_seen()
    return JSONResponse({"ok": True})


@app.get("/stats")
def stats() -> JSONResponse:
    return JSONResponse({"counts": _kept_counts(), "inflight": _inflight})


HTML = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>YouTube — zbieranie</title><style>
body{background:#111;color:#eee;font-family:system-ui;margin:0;padding-bottom:150px;text-align:center}
#stats{padding:6px;font-size:13px;color:#bbb}
#top{padding:8px;position:sticky;top:0;background:#111;z-index:2}
input{font-size:16px;padding:8px;border-radius:8px;border:1px solid #333;background:#1b1b1b;color:#eee;width:60%}
#pl{width:97vw;max-width:760px;aspect-ratio:16/9;margin-top:6px;border:0;border-radius:8px;background:#000}
.bar{position:fixed;bottom:0;left:0;right:0;background:#1b1b1b;padding:10px}
button{font-size:16px;margin:3px;padding:10px 14px;border:0;border-radius:8px;background:#2a2a2a;color:#eee}
.emo.sel{background:#3a6;color:#000}
#keep{background:#2a7;color:#000;font-weight:bold;font-size:20px;padding:12px 26px}
#skip{font-size:20px;padding:12px 26px}
.done{color:#5c6}.need{color:#e88}
a{color:#7bf}
</style></head><body>
<div id="stats"></div>
<div id="top">
 <div id="emobar"></div>
 <input id="q" placeholder="запрос, напр. growling dog" onkeydown="if(event.key==='Enter')startSearch()">
 <button onclick="startSearch()">искать</button>
</div>
<div id="hdr" style="font-size:13px;color:#999;padding:4px"></div>
<iframe id="pl" allow="autoplay" allowfullscreen></iframe>
<div style="font-size:11px;color:#666">клавиши: k — забрать · j — мимо (пробел не жми — он для плеера)</div>
<div class="bar" id="bar"></div>
<script>
const EMO=%EMO%;
let emotion="angry",q="",queue=[],cur=null;
const pl=document.getElementById("pl");
function emobar(){const b=document.getElementById("emobar");b.innerHTML="";EMO.forEach(e=>{const x=document.createElement("button");x.className="emo"+(emotion===e[0]?" sel":"");x.textContent=e[1];x.onclick=()=>{emotion=e[0];emobar();stats();};b.appendChild(x);});}
async function startSearch(){q=document.getElementById("q").value.trim();if(!q)return;document.getElementById("hdr").innerHTML="ищу…";const r=await fetch("/search?emotion="+emotion+"&q="+encodeURIComponent(q));const d=await r.json();queue=d.videos||[];next();}
function next(){
 if(queue.length===0){document.getElementById("hdr").innerHTML="Кандидаты кончились. Введи другой запрос.";pl.src="";cur=null;return;}
 cur=queue.shift();pl.src="https://www.youtube.com/embed/"+cur.id+"?autoplay=1&mute=1&rel=0";
 document.getElementById("hdr").innerHTML='<b>'+emotion+'</b> · '+(cur.duration||"?")+'с · '+(cur.title||"")+' · <a href="'+cur.url+'" target="_blank">открыть</a> · в очереди: '+queue.length;
}
function decide(keep){if(!cur)return;const v=cur;v.query=q;v.emotion=emotion;
 fetch(keep?"/keep":"/skip",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(v)}).then(()=>stats());
 next();}
async function stats(){const r=await fetch("/stats");const d=await r.json();const c=d.counts||{};let h="";for(const e of EMO){const n=c[e[0]]||0;h+="<span class='"+(n>=150?"done":"need")+"'>"+e[0]+" "+n+"</span>&nbsp;&nbsp; ";}if(d.inflight>0)h+="<span style='color:#fc0'>⬆ загружается: "+d.inflight+"</span>";document.getElementById("stats").innerHTML=h;}
document.addEventListener("keydown",e=>{if(e.target.tagName==="INPUT")return;const k=e.key.toLowerCase();if(k==="k"){decide(true);}else if(k==="j"){decide(false);}});
function mkbar(){const b=document.getElementById("bar");b.innerHTML="";
 const s=document.createElement("button");s.id="skip";s.textContent="✗ мимо (j)";s.onclick=()=>decide(false);b.appendChild(s);
 const k=document.createElement("button");k.id="keep";k.textContent="✓ забрать (k)";k.onclick=()=>decide(true);b.appendChild(k);}
emobar();mkbar();stats();setInterval(stats,3000);
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    emo_js = json.dumps([[e, f"{e} · {EMO_PL[e]}"] for e in EMOTIONS])
    return HTML.replace("%EMO%", emo_js)


if __name__ == "__main__":
    print(f"Zbieranie YouTube: http://localhost:{PORT}/", file=sys.stderr)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
