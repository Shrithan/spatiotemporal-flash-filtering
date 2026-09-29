
import numpy as np
import pytest

from flashfilter.temporal import global_change_score, global_change_scores, temporal_difference


def test_identical_frames():
    frame = np.full((10, 10), 100, dtype=np.float32)

    assert np.all(temporal_difference(frame, frame) == 0.0)
    assert global_change_score(frame, frame) == 0.0


def test_complete_brightness_change():
    black = np.zeros((10, 10), dtype=np.float32)
    white = np.ones((10, 10), dtype=np.float32)

    assert global_change_score(black, white) == 1.0


def test_localized_change():
    previous = np.zeros((10, 10), dtype=np.float32)
    current = previous.copy()

    current[:5, :5] = 0.8

    # Only 25% of pixels changed.
    assert global_change_score(previous, current) == pytest.approx(0.2)


def test_score_sequence_starts_at_zero():
    video = np.stack([np.zeros((2, 2)), np.ones((2, 2))]).astype(np.float32)
    assert np.allclose(global_change_scores(video), [0.0, 1.0])
