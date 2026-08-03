"""Testy trekowania psów między klatkami."""

import numpy as np

from packages.models.bbox import Detection
from packages.pipeline.dog_tracker import DogTracker


def _frame_with_patch(color: tuple[int, int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    """Buduje czarną klatkę 400x400 z jednokolorowym prostokątem w miejscu psa."""
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    x, y, w, h = box
    frame[y : y + h, x : x + w] = color
    return frame


def _detection(box: tuple[int, int, int, int]) -> Detection:
    """Buduje detekcję psa o zadanym bboxie."""
    return Detection(bbox=box, confidence=0.9, class_id=0, class_name="dog")


class TestDogTracker:
    """Przypisywanie stabilnych track_id."""

    def test_ten_sam_pies_zachowuje_id_miedzy_klatkami(self):
        box_a = (100, 100, 60, 60)
        box_b = (108, 104, 60, 60)  # niewielkie przesunięcie
        tracker = DogTracker()

        ids_first = tracker.update(_frame_with_patch((0, 0, 200), box_a), [_detection(box_a)])
        ids_second = tracker.update(_frame_with_patch((0, 0, 200), box_b), [_detection(box_b)])

        assert ids_first == ids_second

    def test_dwa_psy_dostaja_rozne_id(self):
        box_left = (30, 30, 50, 50)
        box_right = (300, 300, 50, 50)
        frame = _frame_with_patch((0, 0, 200), box_left)
        frame[300:350, 300:350] = (0, 200, 0)
        tracker = DogTracker()

        ids = tracker.update(frame, [_detection(box_left), _detection(box_right)])

        assert ids[0] != ids[1]

    def test_psy_o_tym_samym_polozeniu_ale_innym_kolorze_nie_zlewaja_sie(self):
        """Kolor rozstrzyga, gdy geometria jest myląca (pies zniknął, inny wszedł w to miejsce)."""
        box = (100, 100, 60, 60)
        tracker = DogTracker()

        first = tracker.update(_frame_with_patch((0, 0, 200), box), [_detection(box)])
        second = tracker.update(_frame_with_patch((0, 200, 0), box), [_detection(box)])

        assert first != second

    def test_krotka_przerwa_wraca_do_tego_samego_treku(self):
        box = (100, 100, 60, 60)
        frame = _frame_with_patch((0, 0, 200), box)
        tracker = DogTracker(max_gap_frames=3)

        first = tracker.update(frame, [_detection(box)])
        tracker.update(np.zeros((400, 400, 3), dtype=np.uint8), [])
        again = tracker.update(frame, [_detection(box)])

        assert first == again

    def test_dluga_przerwa_tworzy_nowy_trek(self):
        box = (100, 100, 60, 60)
        frame = _frame_with_patch((0, 0, 200), box)
        tracker = DogTracker(max_gap_frames=2)

        first = tracker.update(frame, [_detection(box)])
        for _ in range(3):
            tracker.update(np.zeros((400, 400, 3), dtype=np.uint8), [])
        again = tracker.update(frame, [_detection(box)])

        assert first != again
