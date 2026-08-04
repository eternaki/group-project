"""
Testy zapisu treku do zbioru COCO przez anotację wsadową.

Sprawdzają, że do zbioru trafia klatka neutralna każdego treku (baza AU — bez niej
żadne ratio nie ma odniesienia), że peaki wskazują właśnie ją, że trek odrzucony nie
udaje próbki treningowej oraz że anotacja niesie zmierzony szum AU razem z liczbą prób.
"""

from pathlib import Path

import numpy as np
import pytest

from packages.data.coco import COCODataset, au_signal_above_noise
from packages.data.schemas import NUM_KEYPOINTS
from packages.models.delta_action_units import DeltaActionUnit
from packages.models.head_pose import HeadPose
from packages.pipeline.track_processing import TrackFrame, TrackResult, rejected_track
from scripts.annotation.batch_annotate import BatchAnnotator, BatchConfig, VideoContext

FRAME_SIZE: int = 64
NEUTRAL_IDX: int = 0
PEAK_IDX: int = 2
PEAK_RATIO: float = 1.40
REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def _au(name: str, ratio: float) -> DeltaActionUnit:
    """Buduje pojedynczy pomiar AU o zadanym ratio."""
    return DeltaActionUnit(
        name=name,
        ratio=ratio,
        delta=ratio - 1.0,
        is_active=ratio > 1.15,
        confidence=0.8,
    )


def _track_frame(frame_idx: int, ratio: float = 1.0) -> TrackFrame:
    """Buduje klatkę treku z jednym AU o zadanym ratio."""
    keypoints = np.zeros(NUM_KEYPOINTS * 3, dtype=float)
    keypoints[0::3] = np.arange(NUM_KEYPOINTS, dtype=float)
    keypoints[1::3] = np.arange(NUM_KEYPOINTS, dtype=float) * 0.5
    keypoints[2::3] = 0.8
    return TrackFrame(
        frame_idx=frame_idx,
        keypoints=keypoints,
        face_box=(0.0, 0.0, 120.0, 120.0),
        head_pose=HeadPose(yaw_asymmetry=0.0, roll=0.0, is_frontal=True, confidence=0.9),
        delta_aus={"AU101": _au("AU101", ratio)},
    )


def _track(**overrides) -> TrackResult:
    """Buduje przyjęty trek: klatka neutralna 0, peak 2."""
    defaults = {
        "track_id": 3,
        "neutral_frame_idx": NEUTRAL_IDX,
        "frames": [_track_frame(0), _track_frame(1, 1.05), _track_frame(PEAK_IDX, PEAK_RATIO)],
        "peak_indices": [PEAK_IDX],
        "au_noise": {"AU101": 0.10},
        "au_sample_count": {"AU101": 3},
    }
    defaults.update(overrides)
    return TrackResult(**defaults)


@pytest.fixture
def annotator(tmp_path) -> BatchAnnotator:
    """Anotator zapisujący do katalogu tymczasowego, bez modeli AI."""
    config = BatchConfig(
        output_dir=tmp_path / "annotations",
        frames_dir=tmp_path / "frames",
        progress_file=tmp_path / "annotations" / "progress.json",
        # Ścieżka bezwzględna: wynik testu nie może zależeć od katalogu uruchomienia
        mean_shape_file=REPO_ROOT / "models" / "dogflw_mean_shape.json",
    )
    instance = BatchAnnotator(config)
    instance.coco_dataset = COCODataset()
    return instance


@pytest.fixture
def context(tmp_path) -> VideoContext:
    """Kontekst wideo z trzema pustymi klatkami."""
    frames_dir = tmp_path / "frames" / "happy" / "vid"
    frames_dir.mkdir(parents=True, exist_ok=True)
    return VideoContext(
        video_id="vid",
        emotion="happy",
        frames=[np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8) for _ in range(3)],
        frame_numbers=[0, 10, 20],
        frames_dir=frames_dir,
    )


def _save(annotator: BatchAnnotator, context: VideoContext, track: TrackResult) -> list[dict]:
    """Zapisuje trek i zwraca powstałe anotacje."""
    stats = {"frames_processed": 0, "detections": 0, "low_confidence": 0}
    annotator._save_track_frames(track, context, stats)
    return annotator.coco_dataset.to_dict()["annotations"]


class TestZapisTreku:
    """Klatka neutralna i peaki jednego psa."""

    def test_zapisuje_klatke_neutralna_obok_peaku(self, annotator, context):
        annotations = _save(annotator, context, _track())

        assert [ann["frame_role"] for ann in annotations] == ["neutral", "peak"]

    def test_peak_wskazuje_klatke_neutralna_treku(self, annotator, context):
        neutral, peak = _save(annotator, context, _track())

        assert peak["neutral_frame_id"] == neutral["image_id"]

    def test_klatka_neutralna_jest_wlasna_baza(self, annotator, context):
        """Baza AU klatki neutralnej to ona sama — join po neutral_frame_id ma działać."""
        neutral = _save(annotator, context, _track())[0]

        assert neutral["neutral_frame_id"] == neutral["image_id"]

    def test_klatka_neutralna_bedaca_peakiem_zapisana_raz(self, annotator, context):
        """Podwójny wpis podwoiłby wagę tej samej klatki w treningu."""
        annotations = _save(annotator, context, _track(peak_indices=[NEUTRAL_IDX, PEAK_IDX]))

        assert len(annotations) == 2

    def test_kazda_anotacja_niesie_track_id(self, annotator, context):
        annotations = _save(annotator, context, _track())

        assert all(ann["track_id"] == 3 for ann in annotations)

    def test_odrzucony_trek_nie_trafia_do_zbioru(self, annotator, context):
        """Anotacja w COCO jest z definicji próbką treningową — odrzucony trek nią nie jest."""
        track = rejected_track(3, [_track_frame(0)], "za mała morda: 30 px < 64 px")

        assert _save(annotator, context, track) == []


class TestSzumWZapisie:
    """Szum AU trafia do zbioru razem z liczbą prób i porównaniem z sygnałem."""

    def test_anotacja_niesie_szum_i_liczbe_prob(self, annotator, context):
        peak = _save(annotator, context, _track())[1]

        assert peak["au_noise"] == {"AU101": 0.10}
        assert peak["au_sample_count"] == {"AU101": 3}

    def test_sygnal_powyzej_szumu_jest_rozpoznawalny(self, annotator, context):
        peak = _save(annotator, context, _track())[1]

        assert au_signal_above_noise(peak["au_analysis"]["AU101"]) is True

    def test_au_bez_zmierzonego_szumu_zostaje_nieznane(self, annotator, context):
        """AU o zbyt małej liczbie pomiarów nie ma klucza w au_noise — to brak wiedzy."""
        peak = _save(annotator, context, _track(au_noise={}, au_sample_count={"AU101": 1}))[1]

        assert peak["au_analysis"]["AU101"]["snr"] is None
        assert au_signal_above_noise(peak["au_analysis"]["AU101"]) is None


class TestKanonicznyKsztalt:
    """procrustes_keypoints — planowane wejście sieci AU."""

    def test_zapisuje_kanoniczny_ksztalt(self, annotator, context):
        peak = _save(annotator, context, _track())[1]

        assert len(peak["procrustes_keypoints"]) == NUM_KEYPOINTS * 3

    def test_brak_pliku_ksztaltu_nie_przerywa_anotacji(self, annotator, context, tmp_path):
        annotator.config.mean_shape_file = tmp_path / "nie-ma-takiego.json"

        peak = _save(annotator, context, _track())[1]

        assert "procrustes_keypoints" not in peak
        assert peak["au_analysis"]["AU101"]["ratio"] == PEAK_RATIO
