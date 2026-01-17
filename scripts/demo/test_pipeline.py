#!/usr/bin/env python3
"""
Демонстрационный скрипт Pipeline: Dog Detection → Breed Classification.

Использование:
    python scripts/demo/test_pipeline.py <путь_к_изображению>
    python scripts/demo/test_pipeline.py test/ShihTzu-original.jpeg

Выходные файлы сохраняются рядом с исходным изображением:
    - {name}-with-bbox.jpg   - фото с рамкой детекции
    - {name}-crop-{i}.jpg    - вырезанные собаки
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import timm
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from ultralytics import YOLO


# Пути к ресурсам
PROJECT_ROOT = Path(__file__).parent.parent.parent
BREEDS_PATH = PROJECT_ROOT / "packages" / "models" / "breeds.json"
BREED_WEIGHTS = PROJECT_ROOT / "models" / "breed.pt"  # Обученные веса
YOLO_WEIGHTS = "yolov8n.pt"  # Pretrained, скачается автоматически


def load_image(image_path: Path) -> tuple[Image.Image, np.ndarray]:
    """Загружает изображение."""
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    return image, image_np


def detect_dogs(image_np: np.ndarray, yolo_model: YOLO) -> list[dict]:
    """
    Детектирует собак на изображении.

    Returns:
        Список словарей с bbox и confidence
    """
    results = yolo_model(image_np, verbose=False)

    dogs = []
    for result in results:
        boxes = result.boxes
        for i, cls in enumerate(boxes.cls):
            if int(cls) == 16:  # dog class в COCO
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                dogs.append({
                    "bbox": bbox,  # x1, y1, x2, y2
                    "confidence": conf
                })

    return dogs


def draw_bboxes(
    image: Image.Image,
    dogs: list[dict],
    output_path: Path
) -> None:
    """Рисует bounding boxes на изображении и сохраняет."""
    image_with_bbox = image.copy()
    draw = ImageDraw.Draw(image_with_bbox)

    # Толщина линии пропорционально размеру
    line_width = max(5, min(image.size) // 200)
    font_size = max(30, min(image.size) // 30)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except OSError:
        font = ImageFont.load_default()

    for i, dog in enumerate(dogs):
        x1, y1, x2, y2 = map(int, dog["bbox"])
        conf = dog["confidence"]

        # Рамка
        draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=line_width)

        # Подпись
        label = f"Dog {i+1}: {conf*100:.1f}%"
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        padding = 10

        # Фон для текста
        draw.rectangle(
            [(x1, y1 - text_h - padding * 2), (x1 + text_w + padding * 2, y1)],
            fill=(0, 255, 0)
        )
        draw.text((x1 + padding, y1 - text_h - padding), label, fill=(0, 0, 0), font=font)

    image_with_bbox.save(output_path, quality=95)


def crop_dogs(
    image: Image.Image,
    dogs: list[dict],
    output_dir: Path,
    base_name: str
) -> list[tuple[Image.Image, Path]]:
    """Вырезает собак из изображения и сохраняет crops."""
    crops = []

    for i, dog in enumerate(dogs):
        x1, y1, x2, y2 = map(int, dog["bbox"])
        crop = image.crop((x1, y1, x2, y2))
        crop_path = output_dir / f"{base_name}-crop-{i+1}.jpg"
        crop.save(crop_path, quality=95)
        crops.append((crop, crop_path))

    return crops


def classify_breeds(
    crops: list[tuple[Image.Image, Path]],
    breeds: dict[str, str],
    model: torch.nn.Module,
    transform: transforms.Compose,
    top_k: int = 5
) -> list[list[tuple[str, float]]]:
    """Классифицирует породы для каждого crop."""
    results = []

    for crop, _ in crops:
        tensor = transform(crop).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        topk_idx = torch.topk(probs, top_k).indices.tolist()
        topk_probs = torch.topk(probs, top_k).values.tolist()

        predictions = [
            (breeds.get(str(idx), f"Unknown_{idx}"), prob)
            for idx, prob in zip(topk_idx, topk_probs)
        ]
        results.append(predictions)

    return results


def main() -> None:
    """Основная функция pipeline."""
    parser = argparse.ArgumentParser(
        description="Тест Pipeline: Dog Detection → Breed Classification"
    )
    parser.add_argument("image", type=Path, help="Путь к изображению")
    parser.add_argument("--top-k", type=int, default=5, help="Количество топ пород")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"❌ Файл не найден: {args.image}")
        sys.exit(1)

    print("=" * 60)
    print("PIPELINE: Dog Detection → Breed Classification")
    print("=" * 60)

    # Загрузка изображения
    print(f"\n📷 Загрузка: {args.image}")
    image, image_np = load_image(args.image)
    print(f"   Размер: {image.size[0]}x{image.size[1]}")

    # Определяем выходные пути
    output_dir = args.image.parent
    base_name = args.image.stem

    # STEP 1: Dog Detection
    print("\n" + "-" * 60)
    print("STEP 1: Dog Detection (YOLOv8)")
    print("-" * 60)

    yolo = YOLO(YOLO_WEIGHTS)
    dogs = detect_dogs(image_np, yolo)

    if not dogs:
        print("⚠️  Собаки не найдены на изображении")
        sys.exit(0)

    print(f"✓ Найдено собак: {len(dogs)}")
    for i, dog in enumerate(dogs):
        x1, y1, x2, y2 = map(int, dog["bbox"])
        print(f"   Dog {i+1}: bbox=({x1}, {y1}, {x2}, {y2}), conf={dog['confidence']:.2f}")

    # STEP 2: Сохранение фото с BBox
    print("\n" + "-" * 60)
    print("STEP 2: Сохранение фото с BBox рамкой")
    print("-" * 60)

    bbox_path = output_dir / f"{base_name}-with-bbox.jpg"
    draw_bboxes(image, dogs, bbox_path)
    print(f"✓ Сохранено: {bbox_path}")

    # STEP 3: Crop собак
    print("\n" + "-" * 60)
    print("STEP 3: Вырезание собак (crop)")
    print("-" * 60)

    crops = crop_dogs(image, dogs, output_dir, base_name)
    for crop, path in crops:
        print(f"✓ Сохранено: {path} ({crop.size[0]}x{crop.size[1]})")

    # STEP 4: Breed Classification
    print("\n" + "-" * 60)
    print("STEP 4: Breed Classification (EfficientNet-B4)")
    print("-" * 60)

    # Загружаем маппинг пород
    if not BREEDS_PATH.exists():
        print(f"⚠️  Файл пород не найден: {BREEDS_PATH}")
        sys.exit(1)

    with open(BREEDS_PATH) as f:
        breeds = json.load(f)
    print(f"✓ Загружено {len(breeds)} пород")

    # Загружаем модель
    num_classes = len(breeds)
    if BREED_WEIGHTS.exists():
        model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=num_classes)
        model.load_state_dict(torch.load(BREED_WEIGHTS, map_location="cpu"))
        print(f"✓ EfficientNet-B4 загружен (обученный: {BREED_WEIGHTS})")
    else:
        model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=num_classes)
        print("✓ EfficientNet-B4 загружен (pretrained ImageNet)")
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Классификация
    breed_results = classify_breeds(crops, breeds, model, transform, args.top_k)

    for i, predictions in enumerate(breed_results):
        print(f"\n🐕 Dog {i+1} - Top-{args.top_k} Predictions:")
        for rank, (breed_name, prob) in enumerate(predictions, 1):
            print(f"   {rank}. {breed_name}: {prob*100:.2f}%")

    # Итог
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТ")
    print("=" * 60)
    print(f"\n📁 Сохранённые файлы:")
    print(f"   {bbox_path.name} - фото с рамкой детекции")
    for _, path in crops:
        print(f"   {path.name} - crop для классификации")

    if BREED_WEIGHTS.exists():
        print(f"\n✅ Модель обучена на Stanford Dogs Dataset")
    else:
        print(f"\n💡 Примечание: модель использует ImageNet pretrained веса.")
        print("   Для лучших результатов обучи модель на Stanford Dogs.")


if __name__ == "__main__":
    main()
