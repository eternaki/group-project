#!/usr/bin/env python3
"""
Простий test runner для перевірки базової функціональності.
Запуск: python test_runner.py
"""

import sys
from pathlib import Path

# Додаємо project root до sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("DOGFACS DATASET GENERATOR - QUICK TEST SUITE")
print("=" * 70)

test_results = []

# Test 1: Перевірка імпортів
print("\n[Test 1/6] Перевірка імпортів модулів...")
try:
    from packages.models import (
        EmotionPrediction,
        EmotionRule,
        DogFACSRuleEngine,
        EMOTION_RULES,
        EMOTION_CLASSES,
        classify_emotion_from_delta_aus,
    )
    from packages.models.delta_action_units import (
        DeltaActionUnit,
        DeltaActionUnitsExtractor,
        ACTION_UNIT_NAMES,
    )
    from packages.pipeline.neutral_frame import (
        NeutralFrameDetector,
        estimate_head_pose,
    )
    from packages.pipeline.peak_selector import (
        PeakFrameSelector,
        compute_tfm,
    )
    print("  ✓ Всі імпорти успішні")
    test_results.append(("Imports", "PASS"))
except Exception as e:
    print(f"  ✗ Помилка імпорту: {e}")
    test_results.append(("Imports", "FAIL"))
    sys.exit(1)

# Test 2: Перевірка EMOTION_RULES
print("\n[Test 2/6] Перевірка EMOTION_RULES...")
try:
    assert len(EMOTION_RULES) == 6, f"Expected 6 rules, got {len(EMOTION_RULES)}"
    rule_emotions = {rule.emotion for rule in EMOTION_RULES}
    assert rule_emotions == set(EMOTION_CLASSES), "Rules don't cover all emotions"
    print(f"  ✓ {len(EMOTION_RULES)} emotion rules validated")
    print(f"    Emotions: {', '.join(sorted(EMOTION_CLASSES))}")
    test_results.append(("Emotion Rules", "PASS"))
except Exception as e:
    print(f"  ✗ {e}")
    test_results.append(("Emotion Rules", "FAIL"))

# Test 3: Перевірка ACTION_UNIT_NAMES
print("\n[Test 3/6] Перевірка ACTION_UNIT_NAMES...")
try:
    assert len(ACTION_UNIT_NAMES) == 12, f"Expected 12 AUs, got {len(ACTION_UNIT_NAMES)}"
    expected_aus = {
        "AU101", "AU102", "AU12", "AU115", "AU116", "AU117",
        "AU121", "EAD102", "EAD103", "AD19", "AD37", "AU26",
    }
    assert set(ACTION_UNIT_NAMES) == expected_aus, "AU names mismatch"
    print(f"  ✓ {len(ACTION_UNIT_NAMES)} Action Units validated")
    test_results.append(("Action Units", "PASS"))
except Exception as e:
    print(f"  ✗ {e}")
    test_results.append(("Action Units", "FAIL"))

# Test 4: Тест DogFACSRuleEngine
print("\n[Test 4/6] Тест DogFACSRuleEngine...")
try:
    import numpy as np

    # Створимо mock delta AUs для happy emotion
    delta_aus = {
        "AU101": DeltaActionUnit("AU101", 1.05, 0.05, False, 0.9),
        "AU102": DeltaActionUnit("AU102", 1.03, 0.03, False, 0.9),
        "AU12": DeltaActionUnit("AU12", 1.25, 0.25, True, 0.9),  # Smile
        "AU115": DeltaActionUnit("AU115", 1.00, 0.00, False, 0.9),
        "AU116": DeltaActionUnit("AU116", 1.00, 0.00, False, 0.9),
        "AU117": DeltaActionUnit("AU117", 1.00, 0.00, False, 0.9),
        "AU121": DeltaActionUnit("AU121", 1.00, 0.00, False, 0.9),
        "EAD102": DeltaActionUnit("EAD102", 1.15, 0.15, True, 0.9),  # Ears forward
        "EAD103": DeltaActionUnit("EAD103", 1.00, 0.00, False, 0.9),
        "AD19": DeltaActionUnit("AD19", 1.00, 0.00, False, 0.9),
        "AD37": DeltaActionUnit("AD37", 1.00, 0.00, False, 0.9),
        "AU26": DeltaActionUnit("AU26", 1.05, 0.05, False, 0.9),
    }

    engine = DogFACSRuleEngine()
    prediction = engine.classify(delta_aus)

    assert prediction.emotion == "happy", f"Expected 'happy', got '{prediction.emotion}'"
    assert prediction.confidence > 0.7, f"Low confidence: {prediction.confidence}"
    assert prediction.rule_applied is not None, "No rule applied"

    print(f"  ✓ Emotion classified: {prediction.emotion}")
    print(f"    Confidence: {prediction.confidence:.2f}")
    print(f"    Rule: {prediction.rule_applied}")
    test_results.append(("Rule Engine", "PASS"))
except Exception as e:
    print(f"  ✗ {e}")
    test_results.append(("Rule Engine", "FAIL"))

# Test 5: Тест NeutralFrameDetector
print("\n[Test 5/6] Тест NeutralFrameDetector...")
try:
    import numpy as np

    detector = NeutralFrameDetector()

    # Mock frames та keypoints
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(10)]

    # Простий mock keypoints (20 × 3 = 60 values)
    mock_kp = np.array([
        [100, 150, 0.95], [200, 150, 0.95], [150, 200, 0.95],
        [80, 100, 0.9], [220, 100, 0.9], [70, 50, 0.85],
        [230, 50, 0.85], [120, 220, 0.9], [180, 220, 0.9],
        [150, 210, 0.9], [150, 230, 0.9], [150, 250, 0.85],
        [150, 120, 0.8], [90, 150, 0.9], [210, 150, 0.9],
        [110, 150, 0.9], [190, 150, 0.9], [150, 180, 0.85],
        [100, 190, 0.8], [200, 190, 0.8],
    ], dtype=np.float32).flatten()

    keypoints_list = [mock_kp.copy() for _ in range(10)]
    head_poses = [estimate_head_pose(kp) for kp in keypoints_list]

    neutral_idx = detector.detect_auto(frames, keypoints_list, head_poses)

    assert 0 <= neutral_idx < len(frames), f"Invalid neutral_idx: {neutral_idx}"

    print(f"  ✓ Neutral frame detected: frame {neutral_idx}")
    test_results.append(("Neutral Detection", "PASS"))
except Exception as e:
    print(f"  ✗ {e}")
    test_results.append(("Neutral Detection", "FAIL"))

# Test 6: Тест DeltaActionUnitsExtractor
print("\n[Test 6/6] Тест DeltaActionUnitsExtractor...")
try:
    import numpy as np

    # Mock neutral keypoints
    neutral_kp = np.array([
        [100, 150, 0.95], [200, 150, 0.95], [150, 200, 0.95],
        [80, 100, 0.9], [220, 100, 0.9], [70, 50, 0.85],
        [230, 50, 0.85], [120, 220, 0.9], [180, 220, 0.9],
        [150, 210, 0.9], [150, 230, 0.9], [150, 250, 0.85],
        [150, 120, 0.8], [90, 150, 0.9], [210, 150, 0.9],
        [110, 150, 0.9], [190, 150, 0.9], [150, 180, 0.85],
        [100, 190, 0.8], [200, 190, 0.8],
    ], dtype=np.float32).flatten()

    extractor = DeltaActionUnitsExtractor(neutral_kp)
    delta_aus = extractor.extract(neutral_kp)  # neutral vs neutral

    assert len(delta_aus) == 12, f"Expected 12 AUs, got {len(delta_aus)}"

    # Перевірка що neutral vs neutral дає ratio ~1.0
    for au_name, au in delta_aus.items():
        assert 0.95 <= au.ratio <= 1.05, f"{au_name} ratio not ~1.0: {au.ratio}"
        assert au.is_active is False, f"{au_name} should not be active"

    print(f"  ✓ Delta AU extracted: {len(delta_aus)} AUs")
    print(f"    All ratios ~1.0 (neutral baseline)")
    test_results.append(("Delta AU", "PASS"))
except Exception as e:
    print(f"  ✗ {e}")
    test_results.append(("Delta AU", "FAIL"))

# Summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)

passed = sum(1 for _, status in test_results if status == "PASS")
failed = sum(1 for _, status in test_results if status == "FAIL")

for test_name, status in test_results:
    symbol = "✓" if status == "PASS" else "✗"
    print(f"  {symbol} {test_name:<25} {status}")

print(f"\nTotal: {passed}/{len(test_results)} passed")

if failed > 0:
    print("\n⚠ Some tests failed. Please check the errors above.")
    sys.exit(1)
else:
    print("\n✅ All tests passed!")
    sys.exit(0)
