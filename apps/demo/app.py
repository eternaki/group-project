"""
Dog FACS Demo Application.

Aplikacja Streamlit do demonstracji pipeline klasyfikacji emocji psów.
Pozwala na:
- Upload obrazów (JPG, PNG)
- Upload wideo (MP4, MOV, AVI) lub URL (YouTube)
- Analizę przez wszystkie 4 modele
- Wizualizację wyników w czasie rzeczywistym
- Eksport anotacji do COCO JSON i anotowanego wideo
"""

import json
import sys
import tempfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image

# Dodaj root projektu do PYTHONPATH
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from packages.pipeline import (
    InferencePipeline,
    PipelineConfig,
    FrameResult,
    VideoProcessor,
    VideoInfo,
    YouTubeDownloader,
    DownloadResult,
)
from packages.data import COCODataset


# Konfiguracja strony
st.set_page_config(
    page_title="Dog FACS Demo",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_pipeline(confidence_threshold: float = 0.3) -> InferencePipeline:
    """
    Ładuje pipeline z cache'owaniem.

    Args:
        confidence_threshold: Próg pewności detekcji

    Returns:
        Załadowany InferencePipeline
    """
    config = PipelineConfig(
        bbox_weights=project_root / "models" / "yolov8m.pt",
        breed_weights=project_root / "models" / "breed.pt",
        keypoints_weights=project_root / "models" / "keypoints_best.pt",
        emotion_weights=project_root / "models" / "emotion.pt",
        breeds_json=project_root / "packages" / "models" / "breeds.json",
        device="cpu",
        confidence_threshold=confidence_threshold,
        max_dogs=10,
    )

    pipeline = InferencePipeline(config)
    pipeline.load()
    return pipeline


def visualize_results(
    image: np.ndarray,
    result: FrameResult,
    show_bbox: bool = True,
    show_keypoints: bool = True,
    show_labels: bool = True,
) -> np.ndarray:
    """
    Wizualizuje wyniki na obrazie.

    Args:
        image: Oryginalny obraz
        result: Wynik przetwarzania
        show_bbox: Czy rysować bounding boxy
        show_keypoints: Czy rysować keypoints
        show_labels: Czy rysować etykiety

    Returns:
        Obraz z narysowanymi anotacjami
    """
    from PIL import Image as PILImage, ImageDraw

    # Kolory dla różnych psów
    colors = [
        (255, 0, 0),      # czerwony
        (0, 255, 0),      # zielony
        (0, 0, 255),      # niebieski
        (255, 255, 0),    # żółty
        (255, 0, 255),    # magenta
        (0, 255, 255),    # cyan
        (255, 128, 0),    # pomarańczowy
        (128, 0, 255),    # fioletowy
        (0, 255, 128),    # turkusowy
        (255, 128, 128),  # różowy
    ]

    # Konwertuj do PIL
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    pil_image = PILImage.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    for ann in result.annotations:
        color = colors[ann.dog_id % len(colors)]
        x, y, w, h = ann.bbox

        # Rysuj bbox
        if show_bbox:
            draw.rectangle(
                [(x, y), (x + w, y + h)],
                outline=color,
                width=3,
            )

        # Rysuj keypoints
        if show_keypoints and ann.keypoints:
            for kp in ann.keypoints.keypoints:
                if kp.visibility > 0.3:
                    kp_x = x + kp.x
                    kp_y = y + kp.y
                    draw.ellipse(
                        [(kp_x - 3, kp_y - 3), (kp_x + 3, kp_y + 3)],
                        fill=color,
                        outline=(255, 255, 255),
                    )

        # Rysuj etykietę
        if show_labels:
            label_parts = [f"Dog {ann.dog_id}"]

            if ann.breed:
                label_parts.append(ann.breed.class_name)

            if ann.emotion:
                label_parts.append(ann.emotion.emotion.upper())

            label = " | ".join(label_parts)

            # Tło etykiety
            try:
                text_bbox = draw.textbbox((x, y - 25), label)
                draw.rectangle(
                    [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2],
                    fill=color,
                )
                draw.text((x, y - 25), label, fill=(255, 255, 255))
            except Exception:
                # Fallback dla starszych wersji Pillow
                draw.text((x, y - 20), label, fill=color)

    return np.array(pil_image)


def export_to_coco(result: FrameResult, filename: str) -> str:
    """
    Eksportuje wyniki do formatu COCO JSON.

    Args:
        result: Wynik przetwarzania
        filename: Nazwa pliku obrazu

    Returns:
        JSON string
    """
    dataset = COCODataset()

    # Dodaj obraz
    image_id = dataset.add_image(
        file_name=filename,
        width=result.width,
        height=result.height,
    )

    # Dodaj anotacje
    for ann in result.annotations:
        dataset.add_annotation_from_dog(image_id, ann)

    return json.dumps(dataset.to_dict(), indent=2, ensure_ascii=False)


def export_video_to_coco(
    results: list[FrameResult],
    video_name: str,
) -> str:
    """
    Eksportuje wszystkie klatki wideo do COCO JSON.

    Args:
        results: Lista wyników dla każdej klatki
        video_name: Nazwa wideo

    Returns:
        JSON string
    """
    dataset = COCODataset()

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


def render_dog_details(result: FrameResult):
    """
    Renderuje szczegóły każdego psa.

    Args:
        result: Wynik przetwarzania
    """
    if not result.annotations:
        st.info("Nie wykryto psów na obrazie.")
        return

    st.subheader(f"Wykryto {len(result.annotations)} psów")

    for ann in result.annotations:
        with st.expander(f"🐕 Pies #{ann.dog_id}", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Detekcja**")
                st.write(f"Confidence: {ann.bbox_confidence:.1%}")
                st.write(f"BBox: {ann.bbox}")

            with col2:
                st.markdown("**Rasa**")
                if ann.breed:
                    st.write(f"**{ann.breed.class_name}**")
                    st.progress(ann.breed.confidence)
                    st.caption(f"{ann.breed.confidence:.1%}")

                    # Top-3 rasy
                    if ann.breed.top_k and len(ann.breed.top_k) > 1:
                        st.markdown("*Inne możliwe rasy:*")
                        for _, name, conf in ann.breed.top_k[1:4]:
                            st.caption(f"  • {name}: {conf:.1%}")
                else:
                    st.write("Brak danych")

            with col3:
                st.markdown("**Emocja**")
                if ann.emotion:
                    emotion_emoji = {
                        "happy": "😊",
                        "sad": "😢",
                        "angry": "😠",
                        "relaxed": "😌",
                    }
                    emoji = emotion_emoji.get(ann.emotion.emotion, "🐕")
                    st.write(f"**{emoji} {ann.emotion.emotion.upper()}**")
                    st.progress(ann.emotion.confidence)
                    st.caption(f"{ann.emotion.confidence:.1%}")

                    # Prawdopodobieństwa wszystkich emocji
                    if ann.emotion.probabilities:
                        st.markdown("*Wszystkie emocje:*")
                        for emo, prob in sorted(
                            ann.emotion.probabilities.items(),
                            key=lambda x: x[1],
                            reverse=True
                        ):
                            st.caption(f"  • {emo}: {prob:.1%}")
                else:
                    st.write("Brak danych")


def process_video_realtime(
    video_path: Path,
    pipeline: InferencePipeline,
    fps: float,
    show_bbox: bool,
    show_keypoints: bool,
    show_labels: bool,
) -> tuple[list[FrameResult], list[tuple[int, np.ndarray]]]:
    """
    Przetwarza wideo i aktualizuje UI w czasie rzeczywistym.

    Args:
        video_path: Ścieżka do pliku wideo
        pipeline: Załadowany pipeline
        fps: Częstotliwość próbkowania klatek
        show_bbox: Czy pokazywać bounding boxy
        show_keypoints: Czy pokazywać keypoints
        show_labels: Czy pokazywać etykiety

    Returns:
        Tuple (lista wyników, lista anotowanych klatek)
    """
    processor = VideoProcessor(fps_sample=fps)
    info = processor.get_video_info(video_path)

    # Kontenery dla UI
    col_orig, col_annot = st.columns(2)
    with col_orig:
        st.markdown("**Oryginał**")
        placeholder_orig = st.empty()
    with col_annot:
        st.markdown("**Anotowany**")
        placeholder_annot = st.empty()

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Akumulacja wyników
    all_results: list[FrameResult] = []
    annotated_frames: list[tuple[int, np.ndarray]] = []

    total_frames = processor.count_estimated_frames(video_path)
    if total_frames == 0:
        total_frames = 1  # Unikaj dzielenia przez zero

    for i, (frame_num, frame_rgb) in enumerate(processor.extract_frames(video_path)):
        # Przetwarzanie klatki
        result = pipeline.process_frame(frame_rgb, frame_id=f"frame_{frame_num}")
        all_results.append(result)

        # Wizualizacja
        annotated = visualize_results(
            frame_rgb, result, show_bbox, show_keypoints, show_labels
        )
        annotated_frames.append((frame_num, annotated))

        # Aktualizacja UI
        placeholder_orig.image(frame_rgb, use_container_width=True)
        placeholder_annot.image(annotated, use_container_width=True)
        progress = min((i + 1) / total_frames, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Klatka {i + 1} / {total_frames} | Wykryto psów: {len(result.annotations)}")

    progress_bar.progress(1.0)
    status_text.text(f"Zakończono! Przetworzono {len(all_results)} klatek.")

    return all_results, annotated_frames


def render_video_stats(results: list[FrameResult]):
    """
    Renderuje statystyki dla całego wideo.

    Args:
        results: Lista wyników dla każdej klatki
    """
    if not results:
        return

    total_detections = sum(len(r.annotations) for r in results)
    frames_with_dogs = sum(1 for r in results if r.annotations)

    # Zbierz wszystkie emocje
    all_emotions = []
    all_breeds = []
    for r in results:
        for ann in r.annotations:
            if ann.emotion:
                all_emotions.append(ann.emotion.emotion)
            if ann.breed:
                all_breeds.append(ann.breed.class_name)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Klatki z psami", f"{frames_with_dogs}/{len(results)}")

    with col2:
        st.metric("Łączne detekcje", total_detections)

    with col3:
        if all_emotions:
            most_common = max(set(all_emotions), key=all_emotions.count)
            st.metric("Dominująca emocja", most_common.upper())
        else:
            st.metric("Dominująca emocja", "-")

    with col4:
        unique_breeds = len(set(all_breeds))
        st.metric("Unikalne rasy", unique_breeds)


def render_image_tab(confidence: float, show_bbox: bool, show_keypoints: bool, show_labels: bool):
    """Renderuje zakładkę obrazów."""
    st.header("📸 Upload Obrazu")

    uploaded_file = st.file_uploader(
        "Wybierz obraz z psem",
        type=["jpg", "jpeg", "png"],
        help="Obsługiwane formaty: JPG, JPEG, PNG. Maksymalny rozmiar: 10MB.",
        key="image_uploader",
    )

    if uploaded_file is not None:
        # Walidacja rozmiaru
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error("Plik za duży! Maksymalny rozmiar to 10MB.")
            return

        # Wczytaj obraz
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # Pokaż oryginalny obraz
        st.image(image, caption="Wczytany obraz", use_container_width=True)

        # Przycisk analizy
        if st.button("🔍 Analizuj", type="primary", use_container_width=True, key="analyze_image"):
            with st.spinner("Ładowanie pipeline..."):
                pipeline = load_pipeline(confidence)

            with st.spinner("Przetwarzanie obrazu..."):
                result = pipeline.process_frame(image_np, frame_id=uploaded_file.name)

            # Wizualizacja
            st.header("📊 Wyniki")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Oryginalny obraz")
                st.image(image_np, use_container_width=True)

            with col2:
                st.subheader("Anotowany obraz")
                annotated = visualize_results(
                    image_np, result, show_bbox, show_keypoints, show_labels
                )
                st.image(annotated, use_container_width=True)

            # Szczegóły
            st.divider()
            render_dog_details(result)

            # Eksport
            st.divider()
            st.header("💾 Eksport")

            col1, col2 = st.columns(2)

            with col1:
                # Eksport JSON
                json_str = export_to_coco(result, uploaded_file.name)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                st.download_button(
                    label="📥 Pobierz COCO JSON",
                    data=json_str,
                    file_name=f"annotations_{timestamp}.json",
                    mime="application/json",
                    use_container_width=True,
                )

            with col2:
                # Eksport obrazu
                img_buffer = BytesIO()
                Image.fromarray(annotated).save(img_buffer, format="JPEG", quality=95)

                st.download_button(
                    label="📥 Pobierz anotowany obraz",
                    data=img_buffer.getvalue(),
                    file_name=f"annotated_{timestamp}.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )

            # Statystyki
            st.divider()
            st.header("📈 Statystyki")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Wykryte psy", len(result.annotations))

            with col2:
                if result.annotations:
                    breeds = set(
                        ann.breed.class_name
                        for ann in result.annotations
                        if ann.breed
                    )
                    st.metric("Unikalne rasy", len(breeds))
                else:
                    st.metric("Unikalne rasy", 0)

            with col3:
                if result.annotations:
                    emotions = [
                        ann.emotion.emotion
                        for ann in result.annotations
                        if ann.emotion
                    ]
                    most_common = max(set(emotions), key=emotions.count) if emotions else "-"
                    st.metric("Dominująca emocja", most_common.upper() if most_common != "-" else "-")
                else:
                    st.metric("Dominująca emocja", "-")

            with col4:
                if result.annotations:
                    avg_conf = np.mean([
                        ann.bbox_confidence for ann in result.annotations
                    ])
                    st.metric("Śr. confidence", f"{avg_conf:.1%}")
                else:
                    st.metric("Śr. confidence", "-")

    else:
        # Placeholder
        st.info("👆 Wgraj obraz, aby rozpocząć analizę.")

        # Przykładowe zdjęcia
        st.subheader("Przykładowe obrazy testowe")

        test_images = [
            project_root / "test" / "ShihTzu-original.jpeg",
            project_root / "test" / "spruce-pets-200-types-of-dogs-45a7bd12aacf458cb2e77b841c41abe7.jpg",
        ]

        cols = st.columns(2)
        for i, img_path in enumerate(test_images):
            if img_path.exists():
                with cols[i]:
                    st.image(
                        str(img_path),
                        caption=img_path.name,
                        use_container_width=True,
                    )


def render_video_tab(
    confidence: float,
    fps: float,
    show_bbox: bool,
    show_keypoints: bool,
    show_labels: bool,
):
    """Renderuje zakładkę wideo."""
    st.header("🎬 Analiza Wideo")

    # Inicjalizacja session_state dla wideo
    if "video_path" not in st.session_state:
        st.session_state.video_path = None
    if "video_name" not in st.session_state:
        st.session_state.video_name = "video"

    # Wybór źródła
    source = st.radio(
        "Źródło wideo",
        ["Upload pliku", "URL (YouTube)"],
        horizontal=True,
        key="video_source",
    )

    if source == "Upload pliku":
        uploaded_file = st.file_uploader(
            "Wybierz wideo",
            type=["mp4", "mov", "avi", "webm"],
            help="Obsługiwane formaty: MP4, MOV, AVI, WebM. Maks. 100MB, 30 sekund.",
            key="video_uploader",
        )

        if uploaded_file is not None:
            # Walidacja rozmiaru
            if uploaded_file.size > 100 * 1024 * 1024:
                st.error("Plik za duży! Maksymalny rozmiar to 100MB.")
                return

            # Zapisz do pliku tymczasowego tylko jeśli to nowy plik
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.get("video_file_id") != file_id:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                    f.write(uploaded_file.read())
                    st.session_state.video_path = Path(f.name)
                    st.session_state.video_name = uploaded_file.name
                    st.session_state.video_file_id = file_id

    else:  # URL
        video_url = st.text_input(
            "URL wideo",
            placeholder="https://youtube.com/watch?v=... lub bezpośredni link do MP4",
            key="video_url",
        )

        if video_url:
            downloader = YouTubeDownloader(max_duration=30)

            # Sprawdź info przed pobraniem
            with st.spinner("Sprawdzanie wideo..."):
                info = downloader.get_video_info(video_url)

            if info and "error" not in info:
                st.info(f"**{info.get('title', 'Unknown')}** | Czas: {info.get('duration', 0):.0f}s")

                if info.get("duration", 0) > 30:
                    st.error("Wideo za długie! Maksymalny czas to 30 sekund.")
                    return

                if st.button("📥 Pobierz wideo", key="download_video"):
                    with st.spinner("Pobieranie wideo..."):
                        result = downloader.download(video_url)

                    if result.success and result.path:
                        st.session_state.video_path = result.path
                        st.session_state.video_name = result.title
                        st.success(f"Pobrano: {result.title}")
                    else:
                        st.error(f"Błąd pobierania: {result.error}")
                        return
            elif info and "error" in info:
                st.error(f"Błąd: {info['error']}")
                return
            else:
                st.warning("Nie można pobrać informacji o wideo. Sprawdź URL.")
                return

    # Przetwarzanie wideo
    video_path = st.session_state.video_path
    video_name = st.session_state.video_name

    if video_path and video_path.exists():
        # Pokaż informacje o wideo
        processor = VideoProcessor(fps_sample=fps)
        try:
            video_info = processor.get_video_info(video_path)

            st.markdown(f"""
            **Informacje o wideo:**
            - Rozdzielczość: {video_info.width}x{video_info.height}
            - FPS: {video_info.fps:.1f}
            - Czas trwania: {video_info.duration:.1f}s
            - Klatki do analizy: ~{processor.count_estimated_frames(video_path)}
            """)

            # Walidacja czasu trwania
            if video_info.duration > 30:
                st.error("Wideo za długie! Maksymalny czas to 30 sekund.")
                return

        except Exception as e:
            st.error(f"Błąd odczytu wideo: {e}")
            return

        # Pokaż preview wideo
        st.video(str(video_path))

        # Przycisk analizy
        if st.button("🔍 Analizuj wideo", type="primary", use_container_width=True, key="analyze_video"):
            with st.spinner("Ładowanie pipeline..."):
                pipeline = load_pipeline(confidence)

            st.header("📊 Analiza w czasie rzeczywistym")

            # Przetwarzanie
            results, annotated_frames = process_video_realtime(
                video_path,
                pipeline,
                fps,
                show_bbox,
                show_keypoints,
                show_labels,
            )

            # Statystyki
            st.divider()
            st.header("📈 Statystyki wideo")
            render_video_stats(results)

            # Eksport
            st.divider()
            st.header("💾 Eksport")

            col1, col2 = st.columns(2)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            with col1:
                # Eksport COCO JSON
                json_str = export_video_to_coco(results, video_name)

                st.download_button(
                    label="📥 Pobierz COCO JSON",
                    data=json_str,
                    file_name=f"{video_name}_annotations_{timestamp}.json",
                    mime="application/json",
                    use_container_width=True,
                )

            with col2:
                # Eksport anotowanego wideo
                if annotated_frames:
                    with st.spinner("Tworzenie anotowanego wideo..."):
                        output_video_path = Path(tempfile.gettempdir()) / f"{video_name}_annotated_{timestamp}.mp4"

                        try:
                            processor.create_annotated_video(
                                video_path,
                                annotated_frames,
                                output_video_path,
                                include_audio=True,
                            )

                            if output_video_path.exists():
                                with open(output_video_path, "rb") as f:
                                    video_bytes = f.read()

                                st.download_button(
                                    label="📥 Pobierz anotowane wideo",
                                    data=video_bytes,
                                    file_name=f"{video_name}_annotated_{timestamp}.mp4",
                                    mime="video/mp4",
                                    use_container_width=True,
                                )

                                # Pokaż preview anotowanego wideo
                                st.subheader("Preview anotowanego wideo")
                                st.video(str(output_video_path))

                        except Exception as e:
                            st.warning(f"Nie można utworzyć wideo: {e}")
                            st.info("Możesz pobrać anotacje jako JSON.")

    else:
        st.info("👆 Wgraj wideo lub podaj URL, aby rozpocząć analizę.")


def main():
    """Główna funkcja aplikacji."""

    # Header
    st.title("🐕 Dog FACS - Analiza Emocji Psów")
    st.markdown(
        "Automatyczna anotacja psów: detekcja, klasyfikacja rasy, "
        "punkty kluczowe i klasyfikacja emocji."
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Ustawienia")

        confidence = st.slider(
            "Próg pewności detekcji",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.05,
            help="Minimalny próg pewności dla detekcji psów"
        )

        st.divider()

        st.subheader("Wizualizacja")
        show_bbox = st.checkbox("Bounding Boxes", value=True)
        show_keypoints = st.checkbox("Keypoints", value=True)
        show_labels = st.checkbox("Etykiety", value=True)

        st.divider()

        st.subheader("Ustawienia wideo")
        fps = st.slider(
            "Klatki do analizy (FPS)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.5,
            help="Liczba klatek na sekundę do analizy"
        )

        st.divider()

        st.subheader("O aplikacji")
        st.markdown(
            """
            **Dog FACS Dataset**

            Projekt grupowy PG WETI

            Pipeline AI:
            - YOLOv8m (detekcja)
            - EfficientNet-B4 (rasy)
            - SimpleBaseline (keypoints)
            - EfficientNet-B0 (emocje)
            """
        )

    # Taby
    tab_image, tab_video = st.tabs(["📸 Obraz", "🎬 Wideo"])

    with tab_image:
        render_image_tab(confidence, show_bbox, show_keypoints, show_labels)

    with tab_video:
        render_video_tab(confidence, fps, show_bbox, show_keypoints, show_labels)


if __name__ == "__main__":
    main()
