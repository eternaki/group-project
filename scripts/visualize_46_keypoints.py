"""
Wizualizacja wszystkich 46 keypoints z DogFLW.

Porównuje 46 vs 20 keypoints.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import torch
from torchvision import transforms

from packages.models import BBoxConfig, BBoxModel
from packages.models.keypoints import SimpleBaselineModel
from packages.data.schemas import (
    NUM_KEYPOINTS,
    NUM_KEYPOINTS_DOGFLW,
    KEYPOINT_NAMES,
    PROJECT_TO_DOGFLW_MAPPING,
)


# Nazwy wszystkich 46 keypoints w DogFLW
DOGFLW_KEYPOINT_NAMES = [
    # 0-1: Oczy
    "left_eye", "right_eye",
    # 2-13: Kontur twarzy (12 punktów)
    "contour_0", "contour_1", "contour_2", "contour_3",
    "contour_4", "contour_5", "contour_6", "contour_7",
    "contour_8", "contour_9", "contour_10", "contour_11",
    # 14-19: Nos (6 punktów)
    "nose_tip", "nose_1", "nose_2", "nose_3", "nose_4", "nose_5",
    # 20-31: Usta (12 punktów)
    "mouth_left", "mouth_1", "mouth_top", "mouth_3",
    "mouth_right", "mouth_5", "mouth_bottom", "mouth_7",
    "mouth_8", "mouth_9", "mouth_10", "mouth_11",
    # 32-45: Uszy i inne (14 punktów)
    "left_ear_base", "left_ear_1", "left_ear_tip", "left_ear_3",
    "right_ear_base", "right_ear_1", "right_ear_tip", "right_ear_3",
    "forehead", "forehead_1", "left_brow", "left_brow_1",
    "right_brow", "right_brow_1",
]


def get_dogflw_color(idx: int) -> tuple:
    """Kolor dla keypoint na podstawie grupy."""
    if idx < 2:  # Oczy
        return (0, 255, 0)  # Zielony
    elif idx < 14:  # Kontur
        return (255, 165, 0)  # Pomarańczowy
    elif idx < 20:  # Nos
        return (0, 0, 255)  # Niebieski
    elif idx < 32:  # Usta
        return (255, 255, 0)  # Żółty
    else:  # Uszy i inne
        return (255, 0, 255)  # Magenta


def visualize_46_keypoints(image_path: str, output_dir: str):
    """Wizualizuj wszystkie 46 keypoints."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Załaduj obraz
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Nie można wczytać: {image_path}")

    print(f"Obraz: {Path(image_path).name}")
    print(f"Rozmiar: {image.shape[1]}x{image.shape[0]}")

    # Wykryj psa
    bbox_config = BBoxConfig(
        weights_path=project_root / "models" / "yolov8m.pt",
        device="cpu",
    )
    bbox_model = BBoxModel(bbox_config)
    bbox_model.load()

    detections = bbox_model.filter_dogs_only(image)
    if not detections:
        print("Nie wykryto psa!")
        return

    detection = detections[0]
    x, y, w, h = detection.bbox
    print(f"BBox: x={x}, y={y}, w={w}, h={h}")

    # Crop
    height, width = image.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, width - x)
    h = min(h, height - y)
    cropped = image[y:y+h, x:x+w]

    # Załaduj model keypoints (raw 46 keypoints)
    device = torch.device("cpu")
    model = SimpleBaselineModel(num_keypoints=NUM_KEYPOINTS_DOGFLW, backbone="resnet34")

    weights_path = project_root / "models" / "keypoints_dogflw.pt"
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print("Model załadowany")

    # Preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tensor = transform(cropped).unsqueeze(0).to(device)

    # Predykcja
    with torch.no_grad():
        heatmaps = model(tensor)

    # Dekoduj wszystkie 46 keypoints
    hm = heatmaps[0].cpu().numpy()
    hm_h, hm_w = hm.shape[1], hm.shape[2]
    scale_x = w / hm_w
    scale_y = h / hm_h

    all_keypoints = []
    for k in range(NUM_KEYPOINTS_DOGFLW):
        heatmap = hm[k]
        max_val = heatmap.max()
        max_idx = heatmap.argmax()
        y_hm = max_idx // hm_w
        x_hm = max_idx % hm_w

        # Transformuj na pełny obraz
        kp_x = x + x_hm * scale_x
        kp_y = y + y_hm * scale_y

        all_keypoints.append((kp_x, kp_y, float(max_val)))

    # === Wizualizacja 1: Wszystkie 46 keypoints ===
    img_46 = image.copy()
    cv2.rectangle(img_46, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for i, (kp_x, kp_y, vis) in enumerate(all_keypoints):
        if vis > 0.1:
            color = get_dogflw_color(i)
            # BGR dla OpenCV
            color_bgr = (color[2], color[1], color[0])

            radius = 4 if vis > 0.3 else 2
            cv2.circle(img_46, (int(kp_x), int(kp_y)), radius, color_bgr, -1)
            cv2.circle(img_46, (int(kp_x), int(kp_y)), radius, (255, 255, 255), 1)

            # Numer keypoint
            cv2.putText(img_46, str(i), (int(kp_x)+5, int(kp_y)-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # Legenda
    cv2.putText(img_46, "46 KEYPOINTS (DogFLW)", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img_46, "Green=Eyes, Orange=Contour, Blue=Nose", (10, 55),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img_46, "Yellow=Mouth, Magenta=Ears", (10, 75),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    output_46 = output_dir / "keypoints_46.jpg"
    cv2.imwrite(str(output_46), img_46)
    print(f"Zapisano: {output_46}")

    # === Wizualizacja 2: Tylko 20 keypoints (projekt) ===
    img_20 = image.copy()
    cv2.rectangle(img_20, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for project_idx in range(NUM_KEYPOINTS):
        dogflw_idx = PROJECT_TO_DOGFLW_MAPPING[project_idx]
        kp_x, kp_y, vis = all_keypoints[dogflw_idx]

        if vis > 0.1:
            color = get_dogflw_color(dogflw_idx)
            color_bgr = (color[2], color[1], color[0])

            radius = 5 if vis > 0.3 else 3
            cv2.circle(img_20, (int(kp_x), int(kp_y)), radius, color_bgr, -1)
            cv2.circle(img_20, (int(kp_x), int(kp_y)), radius, (255, 255, 255), 1)

            # Nazwa keypoint
            name = KEYPOINT_NAMES[project_idx][:8]  # Skróć nazwę
            cv2.putText(img_20, f"{project_idx}:{name}", (int(kp_x)+5, int(kp_y)-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    cv2.putText(img_20, "20 KEYPOINTS (Current Project)", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    output_20 = output_dir / "keypoints_20.jpg"
    cv2.imwrite(str(output_20), img_20)
    print(f"Zapisano: {output_20}")

    # === Porównanie side-by-side ===
    # Zmniejsz oba obrazy do tej samej wysokości
    target_h = 600
    scale = target_h / image.shape[0]
    new_w = int(image.shape[1] * scale)

    img_46_resized = cv2.resize(img_46, (new_w, target_h))
    img_20_resized = cv2.resize(img_20, (new_w, target_h))

    comparison = np.hstack([img_46_resized, img_20_resized])

    output_compare = output_dir / "comparison_46_vs_20.jpg"
    cv2.imwrite(str(output_compare), comparison)
    print(f"Zapisano porównanie: {output_compare}")

    # Statystyki
    print("\n" + "="*60)
    print("STATYSTYKI")
    print("="*60)

    visible_46 = sum(1 for _, _, v in all_keypoints if v > 0.3)
    avg_vis_46 = np.mean([v for _, _, v in all_keypoints])

    visible_20 = sum(1 for i in range(NUM_KEYPOINTS)
                     if all_keypoints[PROJECT_TO_DOGFLW_MAPPING[i]][2] > 0.3)
    avg_vis_20 = np.mean([all_keypoints[PROJECT_TO_DOGFLW_MAPPING[i]][2]
                          for i in range(NUM_KEYPOINTS)])

    print(f"46 keypoints: {visible_46}/46 visible, avg={avg_vis_46:.3f}")
    print(f"20 keypoints: {visible_20}/20 visible, avg={avg_vis_20:.3f}")

    # Pokaż które keypoints tracimy
    print("\n--- Utracone keypoints (26 punktów) ---")
    used_dogflw = set(PROJECT_TO_DOGFLW_MAPPING.values())
    for i in range(NUM_KEYPOINTS_DOGFLW):
        if i not in used_dogflw:
            kp_x, kp_y, vis = all_keypoints[i]
            name = DOGFLW_KEYPOINT_NAMES[i] if i < len(DOGFLW_KEYPOINT_NAMES) else f"kp_{i}"
            status = "OK" if vis > 0.3 else "LOW"
            print(f"  [{i:2d}] {name:15s}: vis={vis:.3f} [{status}]")


def main():
    # Użyj istniejącego obrazu testowego lub podaj własny
    test_images = [
        project_root / "test" / "spruce-pets-200-types-of-dogs-45a7bd12aacf458cb2e77b841c41abe7.jpg",
    ]

    output_dir = project_root / "test" / "keypoints_46_comparison"

    for img_path in test_images:
        if img_path.exists():
            print(f"\n{'='*60}")
            print(f"Przetwarzanie: {img_path.name}")
            print(f"{'='*60}\n")
            visualize_46_keypoints(str(img_path), str(output_dir))
            break
    else:
        print("Brak obrazów testowych!")
        print("Podaj ścieżkę do obrazu jako argument:")
        print("  python scripts/visualize_46_keypoints.py <ścieżka_do_obrazu>")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        output_dir = project_root / "test" / "keypoints_46_comparison"
        visualize_46_keypoints(img_path, str(output_dir))
    else:
        main()
