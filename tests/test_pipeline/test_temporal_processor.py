"""
Testy dla temporalnej agregacji delta AU (TemporalAUBuffer, TemporalProcessor).

Główny nacisk na filtr pozy głowy: progi podane w konstruktorze muszą realnie
docierać do estymatora pozy. Wcześniej `TemporalProcessor` przekazywał jeden
próg pozycyjnie do `validate_head_pose`, przez co skonfigurowany próg był
martwy — klatki filtrowała wyłącznie domyślna wartość w `estimate_head_pose`.

Uruchomienie:
    pytest tests/test_pipeline/test_temporal_processor.py -v
"""

import numpy as np

from packages.data.schemas import KP, NUM_KEYPOINTS
from packages.models.delta_action_units import ACTION_UNIT_NAMES
from packages.pipeline.temporal_processor import TemporalAUBuffer, TemporalProcessor
from tests.test_pipeline.kp_fixtures import make_frontal_kp, make_turned_kp

# Przesunięcia nosa dobrane pod progi frontalności (patrz head_pose.py):
# 20 px → asymetria ~0.19 (poniżej domyślnych 0.35), 60 px → ~0.38 (powyżej).
SHIFT_LEKKI_OBROT = 20.0
SHIFT_MOCNY_OBROT = 60.0


def make_tilted_kp(dy: float) -> np.ndarray:
    """Tworzy twarz przechyloną — prawe oko przesunięte w pionie o `dy`."""
    kp = make_frontal_kp().reshape(NUM_KEYPOINTS, 3)
    kp[KP.RIGHT_EYE_INNER, 1] += dy
    kp[KP.RIGHT_EYE_OUTER, 1] += dy
    return kp.flatten()


def make_low_visibility_kp() -> np.ndarray:
    """Tworzy klatkę z niską widocznością wszystkich punktów."""
    kp = make_frontal_kp().reshape(NUM_KEYPOINTS, 3)
    kp[:, 2] = 0.1
    return kp.flatten()


class TestTemporalProcessorPoseFilter:
    """Testy filtra pozy głowy w TemporalProcessor."""

    def test_akceptuje_klatki_frontalne(self) -> None:
        """Frontalne klatki trafiają do bufora, nic nie jest odrzucane."""
        neutral = make_frontal_kp()
        processor = TemporalProcessor(neutral_keypoints=neutral)

        for _ in range(5):
            processor.process_frame(make_frontal_kp())

        stats = processor.get_statistics()
        assert stats["accepted_frames"] == 5
        assert stats["rejected_head_pose"] == 0

    def test_odrzuca_mocno_obrocona_klatke(self) -> None:
        """Asymetria powyżej domyślnego progu → klatka odrzucona przez pozę."""
        processor = TemporalProcessor(neutral_keypoints=make_frontal_kp())

        processor.process_frame(make_turned_kp(SHIFT_MOCNY_OBROT))

        stats = processor.get_statistics()
        assert stats["accepted_frames"] == 0
        assert stats["rejected_head_pose"] == 1

    def test_zaostrzony_prog_yaw_odrzuca_lekki_obrot(self) -> None:
        """Regresja: próg z konstruktora musi działać, nie tylko wartość domyślna.

        Lekki obrót (asymetria ~0.19) mieści się w domyślnych 0.35, więc przy
        martwym parametrze klatka byłaby zaakceptowana.
        """
        processor = TemporalProcessor(
            neutral_keypoints=make_frontal_kp(),
            max_yaw_asymmetry=0.05,
        )

        processor.process_frame(make_turned_kp(SHIFT_LEKKI_OBROT))

        stats = processor.get_statistics()
        assert stats["accepted_frames"] == 0
        assert stats["rejected_head_pose"] == 1

    def test_zaostrzony_prog_roll_odrzuca_lekkie_przechylenie(self) -> None:
        """Regresja: próg przechylenia z konstruktora też musi docierać do estymatora."""
        tilted = make_tilted_kp(-20.0)

        domyslny = TemporalProcessor(neutral_keypoints=make_frontal_kp())
        domyslny.process_frame(tilted)
        assert domyslny.get_statistics()["accepted_frames"] == 1

        zaostrzony = TemporalProcessor(
            neutral_keypoints=make_frontal_kp(),
            max_roll=5.0,
        )
        zaostrzony.process_frame(tilted)
        assert zaostrzony.get_statistics()["rejected_head_pose"] == 1

    def test_odrzuca_klatke_o_niskiej_widocznosci(self) -> None:
        """Widoczność poniżej progu → odrzucenie przed estymacją pozy."""
        processor = TemporalProcessor(neutral_keypoints=make_frontal_kp())

        processor.process_frame(make_low_visibility_kp())

        stats = processor.get_statistics()
        assert stats["rejected_visibility"] == 1
        assert stats["rejected_head_pose"] == 0

    def test_reset_czysci_statystyki(self) -> None:
        """reset() zeruje liczniki i bufor."""
        processor = TemporalProcessor(neutral_keypoints=make_frontal_kp())
        for _ in range(3):
            processor.process_frame(make_frontal_kp())

        processor.reset()

        assert processor.get_statistics()["total_frames"] == 0
        assert processor.process_frame(make_frontal_kp()) is None


class TestTemporalAUBuffer:
    """Testy bufora agregującego ratio AU."""

    def test_nie_agreguje_przed_min_frames(self) -> None:
        """Poniżej min_frames bufor nie zwraca wyniku."""
        buffer = TemporalAUBuffer(window_size=10, min_frames=3)
        buffer.add_frame({name: 1.0 for name in ACTION_UNIT_NAMES})

        assert buffer.is_ready() is False
        assert buffer.get_aggregated() is None

    def test_usrednia_ratio_z_okna(self) -> None:
        """Średnia ważona ratio przy równych wagach to zwykła średnia."""
        buffer = TemporalAUBuffer(window_size=10, min_frames=2)
        buffer.add_frame({name: 1.0 for name in ACTION_UNIT_NAMES})
        buffer.add_frame({name: 1.4 for name in ACTION_UNIT_NAMES})

        result = buffer.get_aggregated()

        assert result is not None
        assert result.num_frames == 2
        assert abs(result.values[ACTION_UNIT_NAMES[0]] - 1.2) < 1e-6

    def test_okno_odrzuca_najstarsze_klatki(self) -> None:
        """Bufor trzyma najwyżej window_size klatek."""
        buffer = TemporalAUBuffer(window_size=3, min_frames=1)
        for value in (1.0, 1.0, 2.0, 2.0):
            buffer.add_frame({name: value for name in ACTION_UNIT_NAMES})

        result = buffer.get_aggregated()

        assert result is not None
        assert result.num_frames == 3

    def test_wektor_cech_ma_dlugosc_liczby_au(self) -> None:
        """to_feature_vector zwraca jedną wartość na AU."""
        buffer = TemporalAUBuffer(window_size=5, min_frames=1)
        buffer.add_frame({name: 1.0 for name in ACTION_UNIT_NAMES})

        result = buffer.get_aggregated()

        assert result is not None
        assert result.to_feature_vector().shape == (len(ACTION_UNIT_NAMES),)
