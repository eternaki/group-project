# Design: Wsparcie wideo w Demo App

**Data:** 2025-01-23
**Status:** Zatwierdzony
**Autor:** Claude + Danylo Lohachov

---

## Podsumowanie

Dodanie obsługi wideo do aplikacji demo Streamlit z analizą w czasie rzeczywistym, wsparciem YouTube URL i eksportem wyników.

---

## Wymagania

| Parametr | Wybór |
|----------|-------|
| Tryb | Czas rzeczywisty |
| Źródło | Upload + URL (YouTube) |
| FPS | Konfigurowalny (0.5-5, domyślnie 2) |
| Wyświetlanie | Side-by-side (oryginał / anotowany) |
| Eksport | COCO JSON + anotowane wideo MP4 |

---

## Architektura

```
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit App                          │
├─────────────────────────────────────────────────────────────┤
│  Sidebar                │        Main Area                  │
│  ├─ Confidence slider   │  ┌─────────────┬─────────────┐   │
│  ├─ FPS slider (0.5-5)  │  │  Original   │  Annotated  │   │
│  ├─ Visualization opts  │  │   Frame     │    Frame    │   │
│  └─ Source selector     │  └─────────────┴─────────────┘   │
│     (Upload/URL)        │  ┌─────────────────────────────┐ │
│                         │  │  Progress bar + controls    │ │
│                         │  │  [Play/Pause] [Stop]        │ │
│                         │  └─────────────────────────────┘ │
│                         │  ┌─────────────────────────────┐ │
│                         │  │  Export: [JSON] [Video]     │ │
│                         │  └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Komponenty

### 1. VideoProcessor (`packages/pipeline/video.py`)

```python
@dataclass
class VideoInfo:
    """Informacje o wideo."""
    path: Path
    width: int
    height: int
    fps: float
    duration: float
    frame_count: int

class VideoProcessor:
    """Przetwarzanie wideo dla aplikacji demo."""

    def __init__(self, target_fps: float = 2.0):
        self.target_fps = target_fps

    def get_info(self, video_path: Path) -> VideoInfo:
        """Pobiera metadane wideo."""

    def extract_frames(self, video_path: Path) -> Iterator[tuple[int, np.ndarray]]:
        """
        Ekstrahuje klatki z zadanym FPS.

        Yields:
            (frame_number, frame_image)
        """

    def create_annotated_video(
        self,
        original_path: Path,
        frames: list[tuple[int, np.ndarray]],
        output_path: Path,
    ) -> Path:
        """
        Tworzy wideo z anotowanych klatek.
        Używa FFmpeg do zachowania oryginalnego audio.
        """
```

### 2. YouTubeDownloader (`packages/pipeline/downloader.py`)

```python
@dataclass
class DownloadResult:
    """Wynik pobierania wideo."""
    success: bool
    path: Optional[Path]
    title: str
    duration: float
    error: Optional[str] = None

class YouTubeDownloader:
    """Pobieranie wideo po URL."""

    def __init__(self, output_dir: Path, max_duration: int = 30):
        self.output_dir = output_dir
        self.max_duration = max_duration

    def download(self, url: str) -> DownloadResult:
        """
        Pobiera wideo po URL.

        Obsługuje:
        - YouTube (youtube.com, youtu.be)
        - Bezpośrednie linki do wideo (.mp4, .mov)
        """

    def get_video_info(self, url: str) -> Optional[dict]:
        """Pobiera informacje bez ściągania (do walidacji)."""
```

---

## UI Komponenty

### Taby w głównym obszarze

```python
tab_image, tab_video = st.tabs(["📸 Obraz", "🎬 Wideo"])
```

### Sidebar (rozszerzony)

```python
# Istniejące ustawienia
confidence = st.slider("Próg pewności", 0.1, 0.9, 0.3)
show_bbox = st.checkbox("Bounding Boxes", True)
show_keypoints = st.checkbox("Keypoints", True)

# Nowe ustawienia dla wideo
st.divider()
st.subheader("Ustawienia wideo")
fps = st.slider("Klatki do analizy (FPS)", 0.5, 5.0, 2.0, 0.5)
```

### Wybór źródła wideo

```python
source = st.radio("Źródło wideo", ["Upload pliku", "URL (YouTube)"])

if source == "Upload pliku":
    video_file = st.file_uploader("Wybierz wideo", type=["mp4", "mov", "avi"])
else:
    video_url = st.text_input("URL wideo", placeholder="https://youtube.com/watch?v=...")
```

### Ograniczenia

- Maksymalny czas trwania: 30 sekund
- Maksymalny rozmiar pliku: 100 MB
- Obsługiwane formaty: MP4, MOV, AVI, WebM

---

## Logika przetwarzania

```python
def process_video_realtime(video_path: Path, pipeline: InferencePipeline, fps: float):
    """Przetwarza wideo i aktualizuje UI w czasie rzeczywistym."""

    processor = VideoProcessor(target_fps=fps)
    info = processor.get_info(video_path)

    # Kontenery dla UI
    col_orig, col_annot = st.columns(2)
    placeholder_orig = col_orig.empty()
    placeholder_annot = col_annot.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Akumulacja wyników
    all_results: list[FrameResult] = []
    annotated_frames: list[tuple[int, np.ndarray]] = []

    total_frames = int(info.duration * fps)

    for i, (frame_num, frame) in enumerate(processor.extract_frames(video_path)):
        # Przetwarzanie klatki
        result = pipeline.process_frame(frame, frame_id=f"frame_{frame_num}")
        all_results.append(result)

        # Wizualizacja
        annotated = visualize_results(frame, result)
        annotated_frames.append((frame_num, annotated))

        # Aktualizacja UI
        placeholder_orig.image(frame, channels="BGR", caption="Oryginał")
        placeholder_annot.image(annotated, channels="BGR", caption="Anotowany")
        progress_bar.progress((i + 1) / total_frames)
        status_text.text(f"Klatka {i + 1} / {total_frames}")

    return all_results, annotated_frames
```

---

## Eksport

### COCO JSON

```python
def export_video_to_coco(
    results: list[FrameResult],
    video_info: VideoInfo,
    video_name: str,
) -> str:
    """Eksportuje wszystkie klatki do pojedynczego COCO JSON."""

    dataset = COCODataset(
        description=f"Dog FACS - anotacje wideo: {video_name}"
    )

    for i, result in enumerate(results):
        image_id = dataset.add_image(
            file_name=f"{video_name}_frame_{i:04d}.jpg",
            width=result.width,
            height=result.height,
            video_id=video_name,
            frame_number=i,
        )

        for ann in result.annotations:
            dataset.add_annotation_from_dog(image_id, ann)

    return json.dumps(dataset.to_dict(), indent=2, ensure_ascii=False)
```

### Anotowane wideo

```python
def export_annotated_video(
    original_path: Path,
    annotated_frames: list[tuple[int, np.ndarray]],
    output_path: Path,
    original_fps: float,
) -> Path:
    """Tworzy wideo z anotacjami, zachowując oryginalny dźwięk."""

    # 1. Zapisz klatki do tymczasowego pliku (bez dźwięku)
    temp_video = output_path.with_suffix('.temp.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = annotated_frames[0][1].shape[:2]
    writer = cv2.VideoWriter(str(temp_video), fourcc, original_fps, (w, h))

    for _, frame in annotated_frames:
        writer.write(frame)
    writer.release()

    # 2. Dodaj dźwięk z oryginału przez FFmpeg
    cmd = [
        'ffmpeg', '-y',
        '-i', str(temp_video),
        '-i', str(original_path),
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-map', '0:v:0',
        '-map', '1:a:0?',
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True)
    temp_video.unlink()

    return output_path
```

---

## Struktura plików

### Nowe pliki

```
packages/pipeline/
├── __init__.py          # Dodać eksporty
├── video.py             # VideoProcessor, VideoInfo
└── downloader.py        # YouTubeDownloader, DownloadResult

apps/demo/
└── app.py               # Rozszerzyć o taby Video
```

### Zmiany w istniejących plikach

```python
# packages/pipeline/__init__.py
from .video import VideoProcessor, VideoInfo
from .downloader import YouTubeDownloader, DownloadResult

__all__ = [
    # istniejące...
    "VideoProcessor",
    "VideoInfo",
    "YouTubeDownloader",
    "DownloadResult",
]
```

---

## Zależności

| Pakiet | Użycie | Status |
|--------|--------|--------|
| `opencv-python` | Odczyt/zapis wideo | ✅ Jest |
| `yt-dlp` | Pobieranie YouTube | ✅ Jest (optional) |
| `streamlit` | UI | ✅ Jest |
| `ffmpeg` | Montaż wideo z audio | ⚠️ Pakiet systemowy |

### Wymaganie FFmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu
apt install ffmpeg

# Windows
# https://ffmpeg.org/download.html
```

---

## Plan implementacji

| # | Zadanie | Plik | Zależy od |
|---|---------|------|-----------|
| 1 | VideoProcessor klasa | `packages/pipeline/video.py` | - |
| 2 | YouTubeDownloader klasa | `packages/pipeline/downloader.py` | - |
| 3 | Eksporty w `__init__.py` | `packages/pipeline/__init__.py` | 1, 2 |
| 4 | Taby w UI (Image/Video) | `apps/demo/app.py` | - |
| 5 | Sidebar ustawienia wideo | `apps/demo/app.py` | 4 |
| 6 | Upload + URL input | `apps/demo/app.py` | 4, 2 |
| 7 | Realtime przetwarzanie | `apps/demo/app.py` | 1, 3 |
| 8 | Side-by-side wyświetlanie | `apps/demo/app.py` | 7 |
| 9 | Eksport COCO JSON | `apps/demo/app.py` | 7 |
| 10 | Eksport wideo z anotacjami | `apps/demo/app.py` | 1, 7 |
| 11 | Aktualizacja dokumentacji | `README.md`, story 7.3 | 10 |

---

## Powiązane dokumenty

- [Story 7.3: Video Upload](../sprints/7-demo-application/stories/7.3-video-upload.md)
- [Sprint 7: Demo Application](../sprints/7-demo-application/SPRINT.md)
