#!/usr/bin/env python3
"""
Aplikacja Streamlit do manualnej weryfikacji anotacji.

Funkcje:
- Przegląd anotacji z wizualizacją
- Akceptacja/odrzucenie/korekta anotacji
- Śledzenie postępu weryfikacji
- Eksport korekt do JSON

Użycie:
    streamlit run apps/verification/app.py
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st

# Konfiguracja strony
st.set_page_config(
    page_title="Dog FACS - Weryfikacja Anotacji",
    page_icon="🐕",
    layout="wide",
)


# ============================================================================
# Klasy pomocnicze
# ============================================================================


class VerificationSession:
    """Sesja weryfikacji anotacji."""

    def __init__(
        self,
        annotations_path: Path,
        frames_dir: Path,
        sample_ids_path: Optional[Path] = None,
    ) -> None:
        """
        Inicjalizuje sesję weryfikacji.

        Args:
            annotations_path: Ścieżka do pliku COCO JSON
            frames_dir: Katalog z klatkami
            sample_ids_path: Opcjonalnie - lista ID do weryfikacji
        """
        self.annotations_path = Path(annotations_path)
        self.frames_dir = Path(frames_dir)
        self.sample_ids_path = Path(sample_ids_path) if sample_ids_path else None

        self.coco_data: dict = {}
        self.images: list[dict] = []
        self.annotations: dict[int, list[dict]] = {}  # image_id -> annotations
        self.sample_ids: set[int] = set()

        self.corrections: dict[int, dict] = {}  # annotation_id -> correction
        self.verified_ids: set[int] = set()

        self._load_data()

    def _load_data(self) -> None:
        """Wczytuje dane anotacji."""
        if not self.annotations_path.exists():
            return

        with open(self.annotations_path, encoding="utf-8") as f:
            self.coco_data = json.load(f)

        self.images = self.coco_data.get("images", [])

        # Grupuj anotacje po image_id
        for ann in self.coco_data.get("annotations", []):
            image_id = ann.get("image_id")
            if image_id not in self.annotations:
                self.annotations[image_id] = []
            self.annotations[image_id].append(ann)

        # Wczytaj sample IDs jeśli podano
        if self.sample_ids_path and self.sample_ids_path.exists():
            with open(self.sample_ids_path, encoding="utf-8") as f:
                data = json.load(f)
                self.sample_ids = set(data.get("image_ids", []))

            # Filtruj obrazy
            if self.sample_ids:
                self.images = [img for img in self.images if img["id"] in self.sample_ids]

    def get_image_by_index(self, index: int) -> Optional[dict]:
        """Pobiera obraz po indeksie."""
        if 0 <= index < len(self.images):
            return self.images[index]
        return None

    def get_annotations_for_image(self, image_id: int) -> list[dict]:
        """Pobiera anotacje dla obrazu."""
        return self.annotations.get(image_id, [])

    def add_correction(
        self,
        annotation_id: int,
        correction_type: str,
        corrected_data: Optional[dict] = None,
        note: str = "",
    ) -> None:
        """
        Dodaje korektę anotacji.

        Args:
            annotation_id: ID anotacji
            correction_type: Typ korekty (accept, reject, correct)
            corrected_data: Poprawione dane (opcjonalnie)
            note: Notatka
        """
        self.corrections[annotation_id] = {
            "annotation_id": annotation_id,
            "type": correction_type,
            "corrected_data": corrected_data,
            "note": note,
            "timestamp": datetime.now().isoformat(),
            "verified_by": st.session_state.get("annotator_name", "unknown"),
        }

    def mark_verified(self, image_id: int) -> None:
        """Oznacza obraz jako zweryfikowany."""
        self.verified_ids.add(image_id)

    def save_corrections(self, output_path: Path) -> None:
        """Zapisuje korekty do pliku."""
        data = {
            "generated": datetime.now().isoformat(),
            "annotator": st.session_state.get("annotator_name", "unknown"),
            "total_corrections": len(self.corrections),
            "verified_images": len(self.verified_ids),
            "corrections": list(self.corrections.values()),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_progress(self) -> dict:
        """Zwraca statystyki postępu."""
        total = len(self.images)
        verified = len(self.verified_ids)

        return {
            "total": total,
            "verified": verified,
            "remaining": total - verified,
            "percent": round(verified / max(total, 1) * 100, 1),
            "corrections": len(self.corrections),
        }


# ============================================================================
# Funkcje UI
# ============================================================================


def draw_bbox_on_image(image, bbox: list, color: tuple = (0, 255, 0), thickness: int = 2):
    """Rysuje bounding box na obrazie."""
    import cv2

    x, y, w, h = [int(v) for v in bbox[:4]]
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    return image


def draw_keypoints_on_image(image, keypoints: list, color: tuple = (255, 0, 0), radius: int = 3):
    """Rysuje keypoints na obrazie."""
    import cv2

    for i in range(0, len(keypoints) - 2, 3):
        x, y, v = keypoints[i], keypoints[i + 1], keypoints[i + 2]
        if v > 0:
            cv2.circle(image, (int(x), int(y)), radius, color, -1)
    return image


def visualize_annotation(image, annotation: dict):
    """Wizualizuje anotację na obrazie."""
    import cv2

    result = image.copy()

    # BBox
    bbox = annotation.get("bbox", [])
    if bbox:
        confidence = annotation.get("score", annotation.get("confidence", 0))

        # Kolor zależny od confidence
        if confidence >= 0.7:
            color = (0, 255, 0)
        elif confidence >= 0.5:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

        result = draw_bbox_on_image(result, bbox, color)

        # Etykieta
        breed = annotation.get("breed", {})
        breed_name = breed.get("name", "") if isinstance(breed, dict) else str(breed)

        emotion = annotation.get("emotion", {})
        emotion_name = emotion.get("name", "") if isinstance(emotion, dict) else str(emotion)

        label = f"{breed_name[:15]} | {emotion_name} ({confidence:.2f})"

        x, y = int(bbox[0]), int(bbox[1])
        cv2.putText(result, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Keypoints
    keypoints = annotation.get("keypoints", [])
    if keypoints:
        result = draw_keypoints_on_image(result, keypoints)

    return result


def render_sidebar():
    """Renderuje sidebar z konfiguracją."""
    st.sidebar.title("Konfiguracja")

    # Nazwa annotatora
    annotator_name = st.sidebar.text_input(
        "Twoje imię",
        value=st.session_state.get("annotator_name", ""),
        key="annotator_input",
    )
    if annotator_name:
        st.session_state["annotator_name"] = annotator_name

    st.sidebar.divider()

    # Ścieżki
    annotations_path = st.sidebar.text_input(
        "Ścieżka do anotacji COCO",
        value="data/annotations/annotations.json",
    )

    frames_dir = st.sidebar.text_input(
        "Katalog z klatkami",
        value="data/frames",
    )

    sample_ids_path = st.sidebar.text_input(
        "Plik z sample IDs (opcjonalnie)",
        value="",
    )

    # Przycisk ładowania
    if st.sidebar.button("Załaduj dane"):
        try:
            session = VerificationSession(
                annotations_path=Path(annotations_path),
                frames_dir=Path(frames_dir),
                sample_ids_path=Path(sample_ids_path) if sample_ids_path else None,
            )
            st.session_state["session"] = session
            st.session_state["current_index"] = 0
            st.sidebar.success(f"Załadowano {len(session.images)} obrazów")
        except Exception as e:
            st.sidebar.error(f"Błąd: {e}")

    st.sidebar.divider()

    # Postęp
    if "session" in st.session_state:
        session = st.session_state["session"]
        progress = session.get_progress()

        st.sidebar.subheader("Postęp")
        st.sidebar.progress(progress["percent"] / 100)
        st.sidebar.write(f"Zweryfikowano: {progress['verified']} / {progress['total']}")
        st.sidebar.write(f"Korekty: {progress['corrections']}")

        # Eksport korekt
        if st.sidebar.button("Eksportuj korekty"):
            output_path = Path("data/verification/corrections.json")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            session.save_corrections(output_path)
            st.sidebar.success(f"Zapisano do {output_path}")


def render_navigation():
    """Renderuje nawigację między obrazami."""
    if "session" not in st.session_state:
        return

    session = st.session_state["session"]
    current_index = st.session_state.get("current_index", 0)
    total = len(session.images)

    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

    with col1:
        if st.button("⏮ Pierwszy"):
            st.session_state["current_index"] = 0

    with col2:
        if st.button("◀ Poprzedni"):
            st.session_state["current_index"] = max(0, current_index - 1)

    with col3:
        new_index = st.number_input(
            "Obraz",
            min_value=1,
            max_value=max(total, 1),
            value=current_index + 1,
            key="nav_index",
        )
        if new_index - 1 != current_index:
            st.session_state["current_index"] = new_index - 1

    with col4:
        if st.button("Następny ▶"):
            st.session_state["current_index"] = min(total - 1, current_index + 1)

    with col5:
        if st.button("Ostatni ⏭"):
            st.session_state["current_index"] = total - 1

    st.write(f"Obraz {current_index + 1} z {total}")


def render_annotation_editor(annotation: dict, annotation_idx: int):
    """Renderuje edytor pojedynczej anotacji."""
    ann_id = annotation.get("id", annotation_idx)

    with st.expander(f"Anotacja #{ann_id}", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**BBox:**", annotation.get("bbox", []))

            breed = annotation.get("breed", {})
            if isinstance(breed, dict):
                st.write(f"**Rasa:** {breed.get('name', 'N/A')} ({breed.get('confidence', 0):.2f})")
            else:
                st.write(f"**Rasa:** {breed}")

            emotion = annotation.get("emotion", {})
            if isinstance(emotion, dict):
                st.write(f"**Emocja:** {emotion.get('name', 'N/A')} ({emotion.get('confidence', 0):.2f})")
            else:
                st.write(f"**Emocja:** {emotion}")

            confidence = annotation.get("score", annotation.get("confidence", 0))
            st.write(f"**Confidence:** {confidence:.3f}")

        with col2:
            # Korekta emocji
            emotions = ["happy", "sad", "angry", "relaxed", "fearful", "neutral"]
            current_emotion = emotion.get("name", "") if isinstance(emotion, dict) else str(emotion)

            corrected_emotion = st.selectbox(
                "Korekta emocji",
                options=["(bez zmian)"] + emotions,
                key=f"emotion_{ann_id}",
            )

            # Notatka
            note = st.text_input("Notatka", key=f"note_{ann_id}")

        # Przyciski akcji
        action_col1, action_col2, action_col3 = st.columns(3)

        with action_col1:
            if st.button("✅ Akceptuj", key=f"accept_{ann_id}"):
                session = st.session_state["session"]
                session.add_correction(ann_id, "accept", note=note)
                st.success("Zaakceptowano")

        with action_col2:
            if st.button("✏️ Popraw", key=f"correct_{ann_id}"):
                session = st.session_state["session"]
                corrected_data = {}

                if corrected_emotion != "(bez zmian)":
                    corrected_data["emotion"] = corrected_emotion

                session.add_correction(ann_id, "correct", corrected_data, note)
                st.info("Zapisano korektę")

        with action_col3:
            if st.button("❌ Odrzuć", key=f"reject_{ann_id}"):
                session = st.session_state["session"]
                session.add_correction(ann_id, "reject", note=note)
                st.warning("Odrzucono")


def render_main_content():
    """Renderuje główną zawartość - obraz i anotacje."""
    if "session" not in st.session_state:
        st.info("Załaduj dane w panelu bocznym, aby rozpocząć weryfikację.")
        return

    session = st.session_state["session"]
    current_index = st.session_state.get("current_index", 0)

    # Pobierz obraz
    image_info = session.get_image_by_index(current_index)
    if image_info is None:
        st.warning("Brak obrazu do wyświetlenia")
        return

    image_id = image_info["id"]
    file_name = image_info.get("file_name", "")

    # Wczytaj obraz
    image_path = session.frames_dir / file_name

    try:
        import cv2

        image = cv2.imread(str(image_path))
        if image is None:
            st.error(f"Nie można wczytać obrazu: {image_path}")
            return

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        st.error(f"Błąd wczytywania obrazu: {e}")
        return

    # Pobierz anotacje
    annotations = session.get_annotations_for_image(image_id)

    # Informacje o obrazie
    st.subheader(f"Obraz: {file_name}")

    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.write(f"**ID:** {image_id}")
    with col_info2:
        st.write(f"**Rozmiar:** {image_info.get('width', 0)} x {image_info.get('height', 0)}")
    with col_info3:
        st.write(f"**Anotacji:** {len(annotations)}")

    # Wyświetl obraz z anotacjami
    col_img, col_ann = st.columns([2, 1])

    with col_img:
        # Wizualizacja
        vis_image = image_rgb.copy()
        for ann in annotations:
            vis_image = visualize_annotation(vis_image, ann)

        st.image(vis_image, caption="Obraz z anotacjami", use_container_width=True)

    with col_ann:
        st.subheader("Anotacje")

        if not annotations:
            st.write("Brak anotacji dla tego obrazu")
        else:
            for idx, ann in enumerate(annotations):
                render_annotation_editor(ann, idx)

    # Przycisk zatwierdzenia całego obrazu
    st.divider()

    col_final1, col_final2 = st.columns(2)

    with col_final1:
        if st.button("✅ Zatwierdź obraz i przejdź dalej", type="primary"):
            session.mark_verified(image_id)

            # Przejdź do następnego
            if current_index < len(session.images) - 1:
                st.session_state["current_index"] = current_index + 1
                st.rerun()

    with col_final2:
        verified = image_id in session.verified_ids
        if verified:
            st.success("Ten obraz został zweryfikowany")


# ============================================================================
# Main
# ============================================================================


def main():
    """Główna funkcja aplikacji."""
    st.title("🐕 Dog FACS - Weryfikacja Anotacji")

    # Sidebar
    render_sidebar()

    # Nawigacja
    render_navigation()

    st.divider()

    # Główna zawartość
    render_main_content()


if __name__ == "__main__":
    main()
