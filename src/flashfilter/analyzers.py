"""Interpretable, standards-inspired analyzers for visual video events.

These routines implement computational proxies for research.  They are not a
complete implementation of WCAG, ITU-R BT.1702, ISO 9241-391, or a clinical
risk predictor.  In particular, display luminance and viewing geometry are
unknown for ordinary decoded video.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import cv2
import numpy as np

from .luminance import linear_luminance


@dataclass(frozen=True)
class AnalyzerConfig:
    """Configuration for separable luminance, red, and pattern analyzers."""

    luminance_transition: float = 0.10
    red_transition: float = 0.20
    saturated_red_ratio: float = 0.80
    min_area_ratio: float = 0.01
    flash_limit_per_second: int = 3
    pattern_tile_size: int = 64
    pattern_min_transitions: int = 5
    pattern_min_contrast: float = 0.25
    pattern_spacing_cv: float = 0.40
    pattern_min_duration_seconds: float = 0.50


@dataclass(frozen=True)
class AnalyzerResult:
    """Per-frame evidence and summary produced by :func:`analyze_frames`."""

    luminance_flash_masks: np.ndarray
    red_flash_masks: np.ndarray
    pattern_masks: np.ndarray
    luminance_flash_counts: np.ndarray
    red_flash_counts: np.ndarray
    summary: dict[str, float | int | bool]


def saturated_red_signal(rgb: np.ndarray, ratio_threshold: float = 0.80) -> np.ndarray:
    """Return a continuous red-dominance signal, zero outside saturated red.

    The ratio ``R / (R + G + B)`` is an intentionally simple chromatic proxy.
    It is exposed as a parameter and is not presented as an ISO/WCAG color
    definition.
    """
    array = np.asarray(rgb, dtype=np.float32)
    total = array.sum(axis=-1)
    ratio = array[..., 0] / np.maximum(total, 1e-6)
    dominance = np.maximum(array[..., 0] - np.maximum(array[..., 1], array[..., 2]), 0.0)
    return np.where(ratio >= ratio_threshold, dominance, 0.0).astype(np.float32)


def opposing_transition_mask(
    previous_signed: np.ndarray | None,
    current_signed: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Select pixels with consecutive, above-threshold opposing transitions.

    This expresses the pairwise flash concept: a large positive transition
    followed by a large negative transition, or vice versa, at the same pixel.
    """
    if previous_signed is None:
        return np.zeros_like(current_signed, dtype=bool)
    return (
        (previous_signed * current_signed < 0)
        & (np.abs(previous_signed) >= threshold)
        & (np.abs(current_signed) >= threshold)
    )


def rolling_event_counts(events: np.ndarray, fps: float) -> np.ndarray:
    """Count event frames in the trailing one-second timestamp window."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    flags = np.asarray(events, dtype=bool)
    counts = np.zeros(len(flags), dtype=np.int32)
    active: deque[float] = deque()
    for index, event in enumerate(flags):
        timestamp = index / fps
        while active and active[0] <= timestamp - 1.0:
            active.popleft()
        if event:
            active.append(timestamp)
        counts[index] = len(active)
    return counts


def _regular_profile(profile: np.ndarray, minimum_edges: int, contrast: float, max_cv: float) -> bool:
    """Recognize an alternating one-dimensional stripe profile."""
    differences = np.diff(np.asarray(profile, dtype=np.float32))
    positions = np.flatnonzero(np.abs(differences) >= contrast)
    if len(positions) < minimum_edges:
        return False
    signs = np.sign(differences[positions])
    if np.mean(signs[1:] != signs[:-1]) < 0.8:
        return False
    spacing = np.diff(positions).astype(np.float32)
    return bool(spacing.size and float(spacing.std() / max(spacing.mean(), 1e-6)) <= max_cv)


def regular_pattern_mask(frame: np.ndarray, config: AnalyzerConfig) -> np.ndarray:
    """Return tile-local masks for regular, high-contrast stripe patterns.

    Horizontal and vertical mean profiles make the rule transparent and cheap:
    a tile is selected when it contains at least ``pattern_min_transitions``
    strong, approximately equally spaced, alternating edges. Complexity is
    O(HW). Diagonal and curved patterns are an acknowledged limitation.
    """
    y = linear_luminance(np.asarray(frame, dtype=np.float32))
    height, width = y.shape
    size = max(8, int(config.pattern_tile_size))
    mask = np.zeros((height, width), dtype=bool)
    for y0 in range(0, height, size):
        for x0 in range(0, width, size):
            tile = y[y0:min(y0 + size, height), x0:min(x0 + size, width)]
            horizontal = _regular_profile(
                tile.mean(axis=0), config.pattern_min_transitions,
                config.pattern_min_contrast, config.pattern_spacing_cv,
            )
            vertical = _regular_profile(
                tile.mean(axis=1), config.pattern_min_transitions,
                config.pattern_min_contrast, config.pattern_spacing_cv,
            )
            if horizontal or vertical:
                mask[y0:min(y0 + size, height), x0:min(x0 + size, width)] = True
    return mask


def analyze_frames(
    frames: np.ndarray,
    fps: float,
    config: AnalyzerConfig = AnalyzerConfig(),
) -> AnalyzerResult:
    """Analyze decoded RGB frames without combining unlike evidence channels."""
    array = np.asarray(frames, dtype=np.float32)
    if array.ndim != 4 or array.shape[-1] != 3 or len(array) == 0:
        raise ValueError("frames must have shape (T, H, W, 3) with T > 0")
    if fps <= 0:
        raise ValueError("fps must be positive")

    luminance = linear_luminance(array)
    red = saturated_red_signal(array, config.saturated_red_ratio)
    shape = array.shape[:3]
    luminance_masks = np.zeros(shape, dtype=bool)
    red_masks = np.zeros(shape, dtype=bool)
    pattern_masks = np.zeros(shape, dtype=bool)
    previous_luminance_signed: np.ndarray | None = None
    previous_red_signed: np.ndarray | None = None

    for index in range(len(array)):
        pattern_masks[index] = regular_pattern_mask(array[index], config)
        if index == 0:
            continue
        luminance_signed = luminance[index] - luminance[index - 1]
        red_signed = red[index] - red[index - 1]
        luminance_masks[index] = opposing_transition_mask(
            previous_luminance_signed, luminance_signed, config.luminance_transition
        )
        red_masks[index] = opposing_transition_mask(
            previous_red_signed, red_signed, config.red_transition
        )
        previous_luminance_signed = luminance_signed
        previous_red_signed = red_signed

    luminance_areas = luminance_masks.mean(axis=(1, 2))
    red_areas = red_masks.mean(axis=(1, 2))
    luminance_events = luminance_areas >= config.min_area_ratio
    red_events = red_areas >= config.min_area_ratio
    luminance_counts = rolling_event_counts(luminance_events, fps)
    red_counts = rolling_event_counts(red_events, fps)

    required_pattern_frames = max(1, int(np.ceil(config.pattern_min_duration_seconds * fps)))
    pattern_present = pattern_masks.any(axis=(1, 2))
    run = longest = 0
    for present in pattern_present:
        run = run + 1 if present else 0
        longest = max(longest, run)

    summary: dict[str, float | int | bool] = {
        "frames": int(len(array)),
        "fps": float(fps),
        "luminance_flash_events": int(luminance_events.sum()),
        "red_flash_events": int(red_events.sum()),
        "peak_luminance_flashes_per_second": int(luminance_counts.max(initial=0)),
        "peak_red_flashes_per_second": int(red_counts.max(initial=0)),
        "peak_luminance_flash_area_ratio": float(luminance_areas.max(initial=0)),
        "peak_red_flash_area_ratio": float(red_areas.max(initial=0)),
        "pattern_longest_duration_seconds": float(longest / fps),
        "luminance_limit_exceeded": bool(np.any(luminance_counts > config.flash_limit_per_second)),
        "red_limit_exceeded": bool(np.any(red_counts > config.flash_limit_per_second)),
        "persistent_pattern_detected": bool(longest >= required_pattern_frames),
        "standards_compliance_claimed": False,
    }
    return AnalyzerResult(
        luminance_masks, red_masks, pattern_masks,
        luminance_counts, red_counts, summary,
    )


def feather_event_masks(
    result: AnalyzerResult,
    sigma: float = 1.5,
    min_area_ratio: float = 0.01,
) -> np.ndarray:
    """Combine qualifying temporal evidence into soft mitigation masks."""
    luminance = result.luminance_flash_masks.copy()
    red = result.red_flash_masks.copy()
    luminance[luminance.mean(axis=(1, 2)) < min_area_ratio] = False
    red[red.mean(axis=(1, 2)) < min_area_ratio] = False
    masks = np.logical_or(luminance, red).astype(np.float32)
    if sigma > 0:
        masks = np.stack([cv2.GaussianBlur(mask, (0, 0), sigma) for mask in masks])
    return np.clip(masks, 0, 1).astype(np.float32)
