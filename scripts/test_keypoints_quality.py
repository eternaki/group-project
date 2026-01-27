"""
Skrypt do testowania jakości modeli keypoints.

Porównuje różne wagi modelu na obrazach testowych.
"""

import sys
from pathlib import Path

# Dodaj project root do path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from packages.models import KeypointsConfig, KeypointsModel, BBoxConfig, BBoxModel
from packages.data.schemas import KEYPOINT_NAMES, NUM_KEYPOINTS


def test_keypoints_model(
    image_path: str,
    weights_path: str,
    output_path: str,
) -> dict:
    """
    Testuje model keypoints na pojedynczym obrazie.

    Returns:
        Słownik z metrykami jakości
    """
    # Załaduj obraz
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Nie można wczytać obrazu: {image_path}")

    print(f"\n{'='*60}")
    print(f"Test: {Path(weights_path).name}")
    print(f"Obraz: {Path(image_path).name}")
    print(f"{'='*60}")

    # Załaduj model bbox do detekcji psa
    bbox_config = BBoxConfig(
        weights_path=project_root / "models" / "yolov8m.pt",
        device="cpu",
    )
    bbox_model = BBoxModel(bbox_config)
    bbox_model.load()

    # Wykryj psy
    detections = bbox_model.filter_dogs_only(image)

    if not detections:
        print("  Nie wykryto psa na obrazie!")
        return {"error": "no_dog_detected"}

    print(f"  Wykryto {len(detections)} psa/psów")

    # Weź pierwszego psa (największy bbox)
    detection = detections[0]
    x, y, w, h = detection.bbox
    print(f"  BBox: x={x}, y={y}, w={w}, h={h}, conf={detection.confidence:.2f}")

    # Crop
    height, width = image.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, width - x)
    h = min(h, height - y)
    cropped = image[y:y+h, x:x+w]

    # Załaduj model keypoints
    kp_config = KeypointsConfig(
        weights_path=Path(weights_path),
        device="cpu",
        use_tta=True,  # Test-Time Augmentation
    )
    kp_model = KeypointsModel(kp_config)
    kp_model.load()

    # Predykcja keypoints
    prediction = kp_model.predict(cropped)

    # Statystyki
    visibilities = [kp.visibility for kp in prediction.keypoints]
    avg_visibility = np.mean(visibilities)
    max_visibility = np.max(visibilities)
    min_visibility = np.min(visibilities)
    visible_count = sum(1 for v in visibilities if v > 0.3)

    print(f"\n  === Keypoints Statistics ===")
    print(f"  Średnia visibility: {avg_visibility:.3f}")
    print(f"  Min visibility: {min_visibility:.3f}")
    print(f"  Max visibility: {max_visibility:.3f}")
    print(f"  Wykryte keypoints (vis > 0.3): {visible_count}/{NUM_KEYPOINTS}")
    print(f"  Overall confidence: {prediction.confidence:.3f}")

    # Szczegóły dla każdego keypoint
    print(f"\n  === Szczegóły Keypoints ===")
    for i, kp in enumerate(prediction.keypoints):
        status = "OK" if kp.visibility > 0.3 else "LOW" if kp.visibility > 0.1 else "BAD"
        print(f"  [{i:2d}] {KEYPOINT_NAMES[i]:20s}: vis={kp.visibility:.3f} [{status}]")

    # Wizualizacja na pełnym obrazie
    result_image = image.copy()

    # Rysuj bbox
    cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Rysuj keypoints (transformując współrzędne z cropa na pełny obraz)
    for i, kp in enumerate(prediction.keypoints):
        if kp.visibility > 0.1:  # Rysuj nawet nisko-pewne
            # Transformuj z cropa na pełny obraz
            px = int(kp.x + x)
            py = int(kp.y + y)

            # Kolor zależy od visibility
            if kp.visibility > 0.5:
                color = (0, 255, 0)  # Zielony - wysoka pewność
                radius = 5
            elif kp.visibility > 0.3:
                color = (0, 255, 255)  # Żółty - średnia pewność
                radius = 4
            else:
                color = (0, 0, 255)  # Czerwony - niska pewność
                radius = 3

            cv2.circle(result_image, (px, py), radius, color, -1)
            cv2.circle(result_image, (px, py), radius, (255, 255, 255), 1)

            # Opcjonalnie: numer keypoint
            cv2.putText(
                result_image, str(i), (px + 5, py - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1
            )

    # Dodaj tekst z metrykami
    cv2.putText(
        result_image,
        f"Model: {Path(weights_path).stem}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
    )
    cv2.putText(
        result_image,
        f"Visible: {visible_count}/{NUM_KEYPOINTS}, Avg: {avg_visibility:.2f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )

    # Legenda
    cv2.putText(result_image, "High conf (>0.5)", (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(result_image, "Med conf (0.3-0.5)", (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(result_image, "Low conf (<0.3)", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Zapisz wynik
    cv2.imwrite(output_path, result_image)
    print(f"\n  Zapisano wizualizację: {output_path}")

    return {
        "weights": Path(weights_path).name,
        "avg_visibility": avg_visibility,
        "visible_count": visible_count,
        "confidence": prediction.confidence,
        "keypoints": [(kp.x, kp.y, kp.visibility) for kp in prediction.keypoints],
    }


def main():
    """Główna funkcja testowa."""

    # Ścieżki
    models_dir = project_root / "models"
    output_dir = project_root / "test" / "keypoints_quality"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Obrazy testowe - użyj istniejących
    test_images = [
        project_root / "test" / "spruce-pets-200-types-of-dogs-45a7bd12aacf458cb2e77b841c41abe7.jpg",
    ]

    # Znajdź dodatkowe obrazy в static/frames
    frames_dir = project_root / "apps" / "webapp" / "backend" / "apps" / "webapp" / "backend" / "static" / "frames"
    if frames_dir.exists():
        for session_dir in frames_dir.iterdir():
            if session_dir.is_dir():
                frames = list(session_dir.glob("*.jpg"))
                if frames:
                    test_images.append(frames[0])  # Weź pierwszy frame z każdej sesji
                    if len(test_images) >= 5:
                        break

    # Modele do porównania
    weight_files = [
        models_dir / "keypoints_dogflw.pt",
        models_dir / "keypoints_best.pt",
    ]

    # Filtruj tylko istniejące
    weight_files = [w for w in weight_files if w.exists()]
    test_images = [i for i in test_images if i.exists()]

    print(f"Znaleziono {len(weight_files)} modeli do testowania")
    print(f"Znaleziono {len(test_images)} obrazów testowych")

    if not weight_files:
        print("BŁĄD: Brak plików z wagami modelu!")
        return

    if not test_images:
        print("BŁĄD: Brak obrazów testowych!")
        return

    # Testuj każdą kombinację
    results = []

    for img_path in test_images[:3]:  # Ogranicz do 3 obrazów
        for weights_path in weight_files:
            output_name = f"{img_path.stem}_{weights_path.stem}.jpg"
            output_path = output_dir / output_name

            try:
                result = test_keypoints_model(
                    str(img_path),
                    str(weights_path),
                    str(output_path),
                )
                result["image"] = img_path.name
                results.append(result)
            except Exception as e:
                print(f"BŁĄD: {e}")
                results.append({"error": str(e), "weights": weights_path.name, "image": img_path.name})

    # Podsumowanie
    print("\n" + "="*60)
    print("PODSUMOWANIE")
    print("="*60)

    for r in results:
        if "error" in r:
            print(f"  {r['weights']:25s} | {r['image']:30s} | ERROR: {r['error']}")
        else:
            print(f"  {r['weights']:25s} | {r['image']:30s} | "
                  f"vis={r['avg_visibility']:.2f}, detected={r['visible_count']}/{NUM_KEYPOINTS}")

    print(f"\nWyniki zapisane w: {output_dir}")


if __name__ == "__main__":
    main()
