#!/usr/bin/env python3
"""
Тест модели keypoints на реальном фото.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

# Добавляем путь к packages
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from packages.models.keypoints import KeypointsConfig, KeypointsModel


def draw_keypoints(image: Image.Image, keypoints: list, threshold: float = 0.3) -> Image.Image:
    """Рисует keypoints на изображении."""
    draw = ImageDraw.Draw(image)

    # Цвета для разных групп keypoints
    colors = {
        "eyes": (0, 255, 0),      # Зелёный
        "nose": (255, 0, 0),       # Красный
        "ears": (0, 0, 255),       # Синий
        "mouth": (255, 255, 0),    # Жёлтый
        "other": (255, 0, 255),    # Пурпурный
    }

    for i, kp in enumerate(keypoints):
        if kp.visibility > threshold:
            # Определяем цвет по индексу
            if i < 2:
                color = colors["eyes"]
            elif i == 2:
                color = colors["nose"]
            elif i < 7:
                color = colors["ears"]
            elif i < 11:
                color = colors["mouth"]
            else:
                color = colors["other"]

            x, y = int(kp.x), int(kp.y)
            r = 4
            draw.ellipse([(x-r, y-r), (x+r, y+r)], fill=color, outline=(255, 255, 255))
            draw.text((x+5, y-5), str(i), fill=(255, 255, 255))

    return image


def main():
    # Пути
    image_path = Path("test/ShihTzu-original.jpeg")
    weights_path = Path("models/keypoints_best.pt")
    output_path = Path("test/ShihTzu-keypoints.jpg")

    if not image_path.exists():
        print(f"Фото не найдено: {image_path}")
        return

    if not weights_path.exists():
        print(f"Веса не найдены: {weights_path}")
        return

    print("=" * 50)
    print("Тест модели Keypoints")
    print("=" * 50)

    # Загружаем модель
    print(f"\nЗагрузка модели: {weights_path}")
    config = KeypointsConfig(weights_path=str(weights_path))
    model = KeypointsModel(config)
    model.load()
    print("Модель загружена!")

    # Загружаем изображение
    print(f"\nЗагрузка фото: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    print(f"Размер: {image.size}")

    # Предсказание
    print("\nПредсказание keypoints...")
    prediction = model.predict(image_np)

    # Результаты
    print(f"\n{'='*50}")
    print("РЕЗУЛЬТАТЫ")
    print(f"{'='*50}")
    print(f"Обнаружено keypoints: {prediction.num_detected} / 46")
    print(f"Средняя уверенность: {prediction.confidence:.2%}")

    # Показываем топ keypoints
    print(f"\nТоп-10 keypoints по уверенности:")
    kp_with_idx = [(i, kp) for i, kp in enumerate(prediction.keypoints)]
    kp_sorted = sorted(kp_with_idx, key=lambda x: x[1].visibility, reverse=True)

    for i, (idx, kp) in enumerate(kp_sorted[:10]):
        print(f"  {i+1}. Keypoint {idx}: ({kp.x:.1f}, {kp.y:.1f}) - {kp.visibility:.2%}")

    # Рисуем keypoints
    print(f"\nСохранение визуализации: {output_path}")
    result_image = draw_keypoints(image.copy(), prediction.keypoints)
    result_image.save(output_path, quality=95)

    print(f"\n{'='*50}")
    print("Готово!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
