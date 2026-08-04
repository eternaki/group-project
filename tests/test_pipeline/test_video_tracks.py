"""
Testy przetwarzania wideo na treki psów (`process_video_for_dataset`).

Testy nie wymagają wag modeli — detektor psa i model keypoints są podmienione
na atrapy o znanej geometrii. Dzięki temu sprawdzamy to, co naprawdę jest tu
trudne: rozdzielenie psów na treki, własną bazę AU każdego psa, dotarcie progów
z konfiguracji do miejsc decyzji i wygładzanie drgań punktów.
"""

from typing import Optional

import numpy as np
import pytest

from packages.data.schemas import KP, NUM_KEYPOINTS, Keypoint
from packages.models.bbox import Detection
from packages.models.keypoints import KeypointsPrediction
from packages.pipeline.inference import (
    InferencePipeline,
    PipelineConfig,
    VideoDatasetConfig,
)
from tests.test_pipeline.kp_fixtures import make_frontal_kp

# Psy w kadrze: lewy mniejszy, prawy większy (różny rozmiar cropu = różna geometria
# mordy w atrapie modelu keypoints — patrz `_FakeKeypointsModel`)
LEFT_DOG_BOX: tuple[int, int, int, int] = (50, 100, 200, 200)
RIGHT_DOG_BOX: tuple[int, int, int, int] = (360, 100, 260, 260)

FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 480
DEFAULT_FRAMES: int = 6

# Próg szerokości cropu rozdzielający geometrię obu psów w atrapie
_WIDE_CROP_PX: int = 250


def _frame_with_two_dogs() -> np.ndarray:
    """Klatka z dwoma wyraźnie różnymi kolorystycznie psami."""
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    for box, color in ((LEFT_DOG_BOX, (200, 30, 30)), (RIGHT_DOG_BOX, (30, 200, 30))):
        x, y, w, h = box
        frame[y : y + h, x : x + w] = color
    return frame


class _FakeBBoxModel:
    """Atrapa detektora psów — zwraca stałe boksy w każdej klatce."""

    def __init__(self, boxes: list[tuple[int, int, int, int]]) -> None:
        self.boxes = boxes

    def predict(self, image: np.ndarray) -> list[Detection]:
        """Zwraca detekcje o zadanych boksach."""
        return [
            Detection(bbox=box, confidence=0.9, class_id=0, class_name="dog")
            for box in self.boxes
        ]


class _FakeKeypointsModel:
    """
    Atrapa modelu keypoints — stały układ punktów w układzie cropu.

    Geometria zależy od szerokości cropu: pies z większym boksem dostaje opuszczoną
    żuchwę. Dzięki temu widać, czy każdy trek liczy AU względem WŁASNEJ klatki
    neutralnej — przy wspólnej bazie ratio odjechałoby od 1.0.

    Drganie (`jitter`) dotyczy wyłącznie czubka nosa. Drganie całej mordy byłoby
    wspólne dla wszystkich punktów, więc skasowałoby się przy normalizacji boksem
    mordy i nie testowałoby wygładzania.
    """

    def __init__(self, jitter: float = 0.0) -> None:
        self.jitter = jitter
        self.calls = 0

    def predict(self, crop: np.ndarray) -> KeypointsPrediction:
        """Zwraca keypoints w układzie cropu."""
        keypoints = make_frontal_kp().reshape(NUM_KEYPOINTS, 3).copy()
        if crop.shape[1] >= _WIDE_CROP_PX:
            keypoints[KP.LOWER_LIP_CENTER, 1] += 20.0
            keypoints[KP.CHIN, 1] += 20.0
        keypoints[KP.NOSE_TIP, 0] += self.jitter * (1.0 if self.calls % 2 else -1.0)
        self.calls += 1

        return KeypointsPrediction(
            keypoints=[
                Keypoint(x=float(x), y=float(y), visibility=float(v))
                for x, y, v in keypoints
            ],
            confidence=float(np.mean(keypoints[:, 2])),
            num_detected=NUM_KEYPOINTS,
        )


def _pipeline(
    jitter: float = 0.0,
    boxes: Optional[list[tuple[int, int, int, int]]] = None,
) -> InferencePipeline:
    """Buduje pipeline z atrapami modeli (bez wag, bez detektora mordy)."""
    pipeline = InferencePipeline(
        PipelineConfig(device="cpu", keypoints_two_pass=False, use_face_detector=False)
    )
    default_boxes = [LEFT_DOG_BOX, RIGHT_DOG_BOX]
    pipeline.bbox_model = _FakeBBoxModel(default_boxes if boxes is None else boxes)
    pipeline.keypoints_model = _FakeKeypointsModel(jitter)
    pipeline._models_loaded = True
    return pipeline


def _config(**overrides) -> VideoDatasetConfig:
    """Konfiguracja testowa — filtr ostrości wyłączony (klatki są syntetyczne)."""
    settings: dict = {"num_peaks": 3, "min_sharpness": 0.0}
    settings.update(overrides)
    return VideoDatasetConfig(**settings)


def _run(
    pipeline: InferencePipeline,
    frames: int = DEFAULT_FRAMES,
    **overrides,
) -> dict:
    """Uruchamia pipeline na zadanej liczbie identycznych klatek."""
    return pipeline.process_video_for_dataset(
        [_frame_with_two_dogs() for _ in range(frames)],
        config=_config(**overrides),
    )


class TestStrukturaWyniku:
    """Kontrakt zwracanej struktury."""

    def test_zwraca_treki_zamiast_jednego_psa(self) -> None:
        result = _run(_pipeline())

        assert set(result) == {"tracks", "rejected_tracks", "total_frames"}
        assert result["total_frames"] == DEFAULT_FRAMES

    def test_dwa_psy_daja_dwa_osobne_treki(self) -> None:
        result = _run(_pipeline())

        track_ids = {track.track_id for track in result["tracks"]}
        assert len(result["tracks"]) == 2
        assert len(track_ids) == 2

    def test_klatka_neutralna_i_peaki_to_numery_klatek_wideo(self) -> None:
        result = _run(_pipeline())

        for track in result["tracks"]:
            frame_indices = {frame.frame_idx for frame in track.frames}
            assert track.neutral_frame_idx in frame_indices
            assert set(track.peak_indices) <= frame_indices

    def test_liczba_peakow_nie_przekracza_zamowionej(self) -> None:
        result = _run(_pipeline(), num_peaks=2)

        for track in result["tracks"]:
            assert len(track.peak_indices) <= 2

    def test_szum_au_liczony_jest_dla_przyjetego_treku(self) -> None:
        result = _run(_pipeline(jitter=2.0))

        for track in result["tracks"]:
            assert track.au_noise, "przyjęty trek musi mieć zmierzony szum AU"
            assert set(track.au_noise) <= set(track.au_sample_count)


class TestOsobnaBazaKazdegoPsa:
    """Każdy pies ma własną klatkę neutralną — bazy nie wolno współdzielić."""

    def test_ratio_au_wynosi_jeden_dla_nieruchomego_psa(self) -> None:
        result = _run(_pipeline())
        assert result["tracks"], "oba psy powinny dać godne treki"

        for track in result["tracks"]:
            for frame in track.frames:
                for au in frame.delta_aus.values():
                    assert au.ratio == pytest.approx(1.0, abs=1e-6), (
                        f"AU {au.name} treku {track.track_id} liczone względem obcej bazy"
                    )

    def test_kazdy_trek_ma_klatke_neutralna_ze_swoich_klatek(self) -> None:
        result = _run(_pipeline())

        neutral_frames = [
            next(
                frame
                for frame in track.frames
                if frame.frame_idx == track.neutral_frame_idx
            )
            for track in result["tracks"]
        ]
        assert len(neutral_frames) == len(result["tracks"])


class TestProgiDocierajaDoDecyzji:
    """Progi z konfiguracji muszą docierać wszędzie tam, gdzie zapada decyzja."""

    def test_prog_pewnosci_dociera_do_bramki_godnosci_treku(self) -> None:
        """Przy progu wyższym niż pewność atrapy żaden trek nie może przejść."""
        result = _run(_pipeline(), min_keypoint_conf=0.99)

        assert not result["tracks"]
        assert result["rejected_tracks"]
        for track in result["rejected_tracks"]:
            assert "pewność" in track.rejected_reason

    def test_wskazana_recznie_klatka_neutralna_jest_uzywana(self) -> None:
        result = _run(_pipeline(), neutral_frame_idx=3)

        assert result["tracks"]
        for track in result["tracks"]:
            assert track.neutral_frame_idx == 3

    def test_niepoprawne_probkowanie_jest_odrzucane(self) -> None:
        with pytest.raises(ValueError, match="fps"):
            VideoDatasetConfig(fps=0.0)


class TestOdrzucaniaTrekow:
    """Odrzucenie musi być zapisane z powodem, nie ciche."""

    def test_zbyt_krotkie_wideo_nie_rzuca_wyjatkiem(self) -> None:
        result = _run(_pipeline(), frames=2)

        assert result["tracks"] == []
        assert len(result["rejected_tracks"]) == 2

    def test_kazde_odrzucenie_ma_powod(self) -> None:
        result = _run(_pipeline(), frames=2)

        for track in result["rejected_tracks"]:
            assert track.rejected_reason, "każde odrzucenie musi mieć powód"

    def test_wideo_bez_psow_daje_pusty_wynik(self) -> None:
        result = _run(_pipeline(boxes=[]))

        assert result["tracks"] == []
        assert result["rejected_tracks"] == []
        assert result["total_frames"] == DEFAULT_FRAMES


class TestWygladzania:
    """Filtr One Euro musi być podpięty osobno dla każdego treku."""

    def test_drganie_nosa_jest_tlumione(self) -> None:
        jitter = 2.0
        result = _run(_pipeline(jitter=jitter), frames=8)

        for track in result["tracks"]:
            nose_x = [
                frame.keypoints.reshape(NUM_KEYPOINTS, 3)[KP.NOSE_TIP, 0]
                for frame in track.frames[1:]
            ]
            spread = max(nose_x) - min(nose_x)
            assert spread < 2 * jitter, (
                f"drganie nietłumione: rozrzut {spread:.2f} px przy amplitudzie "
                f"{2 * jitter:.2f} px"
            )
