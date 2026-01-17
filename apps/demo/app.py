"""
Dog FACS Demo Application.

Aplikacja Streamlit do demonstracji pipeline klasyfikacji emocji psów.
Pozwala na:
- Upload obrazów (JPG, PNG)
- Analizę przez wszystkie 4 modele
- Wizualizację wyników
- Eksport anotacji do COCO JSON
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# Dodaj root projektu do PYTHONPATH
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from packages.pipeline import InferencePipeline, PipelineConfig, FrameResult
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
    from PIL import Image as PILImage, ImageDraw, ImageFont

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

    # Główna treść
    st.header("📸 Upload Obrazu")

    uploaded_file = st.file_uploader(
        "Wybierz obraz z psem",
        type=["jpg", "jpeg", "png"],
        help="Obsługiwane formaty: JPG, JPEG, PNG. Maksymalny rozmiar: 10MB."
    )

    if uploaded_file is not None:
        # Walidacja rozmiaru
        if uploaded_file.size > 10 * 1024 * 1024:
            st.error("❌ Plik za duży! Maksymalny rozmiar to 10MB.")
            return

        # Wczytaj obraz
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # Pokaż oryginalny obraz
        st.image(image, caption="Wczytany obraz", use_container_width=True)

        # Przycisk analizy
        if st.button("🔍 Analizuj", type="primary", use_container_width=True):
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
                from io import BytesIO
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
                    st.metric("Dominująca emocja", most_common.upper())
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


if __name__ == "__main__":
    main()
