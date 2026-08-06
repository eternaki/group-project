"""
Testy dla temporalnej agregacji delta AU (TemporalAUBuffer, TemporalProcessor).

Główny nacisk na filtr pozy głowy: progi podane w konstruktorze muszą realnie
docierać do estymatora pozy. Wcześniej `TemporalProcessor` przekazywał jeden
próg pozycyjnie do `validate_head_pose`, przez co skonfigurowany próg był
martwy — klatki filtrowała wyłącznie wartość domyślna w `estimate_head_pose`.
Dlatego testy sprawdzają obie strony: zaostrzenie progu ma odrzucać, a jego
poluzowanie ma przepuszczać.

Uruchomienie:
    pytest tests/test_pipeline/test_temporal_processor.py -v
"""

from packages.models.delta_action_units import ACTION_UNIT_NAMES
from packages.models.head_pose import (
    DEFAULT_MAX_ROLL,
    DEFAULT_MAX_YAW_ASYMMETRY,
    estimate_head_pose,
)
from packages.pipeline.temporal_processor import TemporalAUBuffer, TemporalProcessor
from tests.test_pipeline.kp_fixtures import (
    make_frontal_kp,
    make_low_visibility_kp,
    make_tilted_kp,
    make_turned_kp,
)

# Przesunięcia dobrane pod progi frontalności (patrz head_pose.py). Metryka
# nasyca się przy dużych przesunięciach (~0.38 to maksimum dla tej fikstury),
# dlatego testy jawnie asertują samą wartość — zmiana geometrii fikstury ma
# oblać test wprost, a nie po cichu odwrócić jego wynik.
NOSE_SHIFT_SMALL_TURN = 20.0
NOSE_SHIFT_LARGE_TURN = 65.0
EYE_SHIFT_SMALL_TILT = -20.0
LOOSE_YAW_THRESHOLD = 0.9
STRICT_YAW_THRESHOLD = 0.05
STRICT_ROLL_THRESHOLD = 5.0


class TestTemporalProcessorPoseFilter:
    """Testy filtra pozy głowy w TemporalProcessor."""

    def test_accepts_frontal_frames(self) -> None:
        """Frontalne klatki trafiają do bufora, nic nie jest odrzucane."""
        processor = TemporalProcessor(neutral_keypoints=make_frontal_kp())

        for _ in range(5):
            processor.process_frame(make_frontal_kp())

        stats = processor.get_statistics()
        assert stats["accepted_frames"] == 5
        assert stats["rejected_head_pose"] == 0

    def test_rejects_strongly_turned_frame(self) -> None:
        """Asymetria powyżej domyślnego progu → klatka odrzucona przez pozę."""
        turned = make_turned_kp(NOSE_SHIFT_LARGE_TURN)
        assert abs(estimate_head_pose(turned).yaw_asymmetry) > DEFAULT_MAX_YAW_ASYMMETRY

        processor = TemporalProcessor(neutral_keypoints=make_frontal_kp())
        processor.process_frame(turned)

        stats = processor.get_statistics()
        assert stats["accepted_frames"] == 0
        assert stats["rejected_head_pose"] == 1

    def test_loosened_yaw_threshold_accepts_turned_frame(self) -> None:
        """Regresja: poluzowany próg musi dotrzeć do estymatora, nie tylko do walidacji.

        Klatka przekracza domyślne 0.35, więc `is_frontal` policzone na wartościach
        domyślnych odrzuciłoby ją mimo jawnie poluzowanego progu.
        """
        processor = TemporalProcessor(
            neutral_keypoints=make_frontal_kp(),
            max_yaw_asymmetry=LOOSE_YAW_THRESHOLD,
        )

        processor.process_frame(make_turned_kp(NOSE_SHIFT_LARGE_TURN))

        assert processor.get_statistics()["accepted_frames"] == 1

    def test_tightened_yaw_threshold_rejects_small_turn(self) -> None:
        """Zaostrzony próg odrzuca obrót, który mieści się w wartości domyślnej."""
        turned = make_turned_kp(NOSE_SHIFT_SMALL_TURN)
        assert abs(estimate_head_pose(turned).yaw_asymmetry) < DEFAULT_MAX_YAW_ASYMMETRY

        processor = TemporalProcessor(
            neutral_keypoints=make_frontal_kp(),
            max_yaw_asymmetry=STRICT_YAW_THRESHOLD,
        )
        processor.process_frame(turned)

        stats = processor.get_statistics()
        assert stats["accepted_frames"] == 0
        assert stats["rejected_head_pose"] == 1

    def test_tightened_roll_threshold_rejects_small_tilt(self) -> None:
        """Regresja: próg przechylenia z konstruktora też musi docierać do estymatora.

        `validate_head_pose` nie sprawdza rolla, więc jedyną drogą tego progu
        jest `estimate_head_pose`.
        """
        tilted = make_tilted_kp(EYE_SHIFT_SMALL_TILT)
        assert abs(estimate_head_pose(tilted).roll) < DEFAULT_MAX_ROLL

        default_processor = TemporalProcessor(neutral_keypoints=make_frontal_kp())
        default_processor.process_frame(tilted)
        assert default_processor.get_statistics()["accepted_frames"] == 1

        strict_processor = TemporalProcessor(
            neutral_keypoints=make_frontal_kp(),
            max_roll=STRICT_ROLL_THRESHOLD,
        )
        strict_processor.process_frame(tilted)
        assert strict_processor.get_statistics()["rejected_head_pose"] == 1

    def test_rejects_low_visibility_frame(self) -> None:
        """Widoczność poniżej progu → odrzucenie przed estymacją pozy."""
        processor = TemporalProcessor(neutral_keypoints=make_frontal_kp())

        processor.process_frame(make_low_visibility_kp())

        stats = processor.get_statistics()
        assert stats["rejected_visibility"] == 1
        assert stats["rejected_head_pose"] == 0

    def test_reset_clears_statistics(self) -> None:
        """reset() zeruje liczniki i bufor."""
        processor = TemporalProcessor(neutral_keypoints=make_frontal_kp())
        for _ in range(3):
            processor.process_frame(make_frontal_kp())

        processor.reset()

        assert processor.get_statistics()["total_frames"] == 0
        assert processor.process_frame(make_frontal_kp()) is None


class TestTemporalAUBuffer:
    """Testy bufora agregującego ratio AU."""

    def test_does_not_aggregate_below_min_frames(self) -> None:
        """Poniżej min_frames bufor nie zwraca wyniku."""
        buffer = TemporalAUBuffer(window_size=10, min_frames=3)
        buffer.add_frame({name: 1.0 for name in ACTION_UNIT_NAMES})

        assert buffer.is_ready() is False
        assert buffer.get_aggregated() is None

    def test_averages_ratios_over_window(self) -> None:
        """Średnia ważona ratio przy równych wagach to zwykła średnia."""
        buffer = TemporalAUBuffer(window_size=10, min_frames=2)
        buffer.add_frame({name: 1.0 for name in ACTION_UNIT_NAMES})
        buffer.add_frame({name: 1.4 for name in ACTION_UNIT_NAMES})

        result = buffer.get_aggregated()

        assert result is not None
        assert result.num_frames == 2
        assert abs(result.values[ACTION_UNIT_NAMES[0]] - 1.2) < 1e-6

    def test_window_drops_oldest_frames(self) -> None:
        """Bufor trzyma najwyżej window_size klatek."""
        buffer = TemporalAUBuffer(window_size=3, min_frames=1)
        for value in (1.0, 1.0, 2.0, 2.0):
            buffer.add_frame({name: value for name in ACTION_UNIT_NAMES})

        result = buffer.get_aggregated()

        assert result is not None
        assert result.num_frames == 3

    def test_feature_vector_has_one_value_per_au(self) -> None:
        """to_feature_vector zwraca jedną wartość na AU."""
        buffer = TemporalAUBuffer(window_size=5, min_frames=1)
        buffer.add_frame({name: 1.0 for name in ACTION_UNIT_NAMES})

        result = buffer.get_aggregated()

        assert result is not None
        assert result.to_feature_vector().shape == (len(ACTION_UNIT_NAMES),)
