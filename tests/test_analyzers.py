import numpy as np

from flashfilter.analyzers import (
    AnalyzerConfig,
    analyze_frames,
    opposing_transition_mask,
    regular_pattern_mask,
    rolling_event_counts,
)
from flashfilter.filtering import AdaptiveFilterConfig, adaptive_event_filter, multianalyzer_filter


def test_opposing_transition_requires_direction_magnitude_and_overlap():
    previous = np.array([[0.2, 0.2], [0.2, 0.01]], dtype=np.float32)
    current = np.array([[-0.2, 0.2], [-0.01, -0.2]], dtype=np.float32)
    expected = np.array([[True, False], [False, False]])
    assert np.array_equal(opposing_transition_mask(previous, current, 0.1), expected)


def test_rolling_counts_use_one_second_timestamp_window():
    events = np.array([True, False, True, False, True, False, True])
    counts = rolling_event_counts(events, fps=4)
    assert counts.tolist() == [1, 1, 2, 2, 2, 2, 2]


def test_luminance_alternation_exceeds_count_but_monotonic_ramp_does_not():
    alternating = np.zeros((8, 8, 8, 3), dtype=np.float32)
    alternating[1::2] = 1
    result = analyze_frames(alternating, fps=8, config=AnalyzerConfig(min_area_ratio=.5))
    assert result.summary["luminance_flash_events"] == 6
    assert result.summary["luminance_limit_exceeded"] is True

    ramp = np.linspace(0, 1, 8, dtype=np.float32)[:, None, None, None] * np.ones((1, 8, 8, 3), np.float32)
    ramp_result = analyze_frames(ramp, fps=8, config=AnalyzerConfig(min_area_ratio=.5))
    assert ramp_result.summary["luminance_flash_events"] == 0


def test_red_analyzer_is_spatially_localized():
    frames = np.zeros((5, 16, 16, 3), dtype=np.float32)
    frames[1::2, 4:12, 4:12, 0] = 1
    result = analyze_frames(frames, fps=10, config=AnalyzerConfig(min_area_ratio=.1))
    assert result.summary["red_flash_events"] == 3
    assert np.all(result.red_flash_masks[2, 4:12, 4:12])
    assert not np.any(result.red_flash_masks[2, :4])


def test_regular_pattern_detector_accepts_stripes_and_rejects_flat_tile():
    striped = np.zeros((32, 32, 3), dtype=np.float32)
    for x in range(0, 32, 4):
        striped[:, x:x + 2] = 1
    config = AnalyzerConfig(pattern_tile_size=32, pattern_min_transitions=5)
    assert np.all(regular_pattern_mask(striped, config))
    assert not np.any(regular_pattern_mask(np.full_like(striped, .5), config))


def test_multianalyzer_filter_changes_only_local_flash_region():
    frames = np.full((4, 16, 16, 3), .1, dtype=np.float32)
    frames[1::2, 4:12, 4:12] = .9
    config = AnalyzerConfig(luminance_transition=.1, min_area_ratio=.1)
    result = multianalyzer_filter(frames, fps=8, blend=.5, feather_sigma=0, analyzer_config=config)
    assert np.allclose(result.frames[:, :4], frames[:, :4])
    assert np.any(np.abs(result.frames[2, 4:12, 4:12] - frames[2, 4:12, 4:12]) > 0)


def test_adaptive_filter_desaturates_only_detected_red_region():
    frames = np.zeros((4, 16, 16, 3), dtype=np.float32)
    frames[1::2, 4:12, 4:12, 0] = 1
    config = AnalyzerConfig(luminance_transition=2, red_transition=.1, min_area_ratio=.1)
    result = adaptive_event_filter(
        frames, 8, analyzer_config=config,
        filter_config=AdaptiveFilterConfig(desaturation=1, max_luminance_step=1, pattern_contrast_reduction=0, feather_sigma=0),
    )
    changed = result.frames[2, 4:12, 4:12]
    assert np.allclose(changed[..., 0], changed[..., 1])
    assert np.allclose(changed[..., 1], changed[..., 2])
    assert np.allclose(result.frames[:, :4], frames[:, :4])


def test_adaptive_filter_bounds_detected_luminance_reversal():
    frames = np.zeros((4, 8, 8, 3), dtype=np.float32)
    frames[1] = 1
    frames[3] = 1
    result = adaptive_event_filter(
        frames, 8, analyzer_config=AnalyzerConfig(min_area_ratio=.5),
        filter_config=AdaptiveFilterConfig(desaturation=0, max_luminance_step=.1, pattern_contrast_reduction=0, feather_sigma=0),
    )
    from flashfilter.luminance import linear_luminance
    assert np.max(np.abs(linear_luminance(result.frames[2]) - linear_luminance(result.frames[1]))) <= .1001


def test_adaptive_channel_switches_isolate_corrections():
    frames=np.zeros((4,16,16,3),np.float32)
    frames[1::2,4:12,4:12,0]=1
    analyzer=AnalyzerConfig(luminance_transition=2,red_transition=.1,min_area_ratio=.1)
    disabled=adaptive_event_filter(
        frames,8,analyzer_config=analyzer,
        filter_config=AdaptiveFilterConfig(enable_luminance=False,enable_red=False,enable_pattern=False),
    )
    assert np.allclose(disabled.frames,frames)
    assert not np.any(disabled.masks)


def test_static_pattern_correction_does_not_create_temporal_transient():
    striped=np.zeros((4,32,32,3),np.float32)
    for x in range(0,32,4): striped[:, :, x:x+2]=1
    result=adaptive_event_filter(striped,8)
    from flashfilter.metrics import temporal_metrics
    assert temporal_metrics(result.frames)["peak_activity"]==0
    assert np.any(np.abs(result.frames-striped)>0)
