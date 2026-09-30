"""Global and localized temporal blending baselines."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .luminance import brightness, linear_luminance, linear_to_srgb, srgb_to_linear
from .localization import LocalizationConfig, localize
from .temporal import global_change_score, temporal_change_maps

@dataclass(frozen=True)
class FilterResult:
    frames: np.ndarray
    masks: np.ndarray


@dataclass(frozen=True)
class AdaptiveFilterConfig:
    """Strengths for event-specific, spatially localized corrections.

    ``desaturation`` applies only to saturated-red flash evidence,
    ``max_luminance_step`` bounds relative-linear-luminance changes at general
    flash pixels, and ``pattern_contrast_reduction`` applies only to regular
    high-contrast pattern tiles. These are research parameters, not clinical
    safety thresholds.
    """

    desaturation: float = 0.90
    max_luminance_step: float = 0.08
    pattern_contrast_reduction: float = 0.35
    feather_sigma: float = 1.5
    enable_luminance: bool = True
    enable_red: bool = True
    enable_pattern: bool = True


def _soft_mask(mask: np.ndarray, sigma: float) -> np.ndarray:
    import cv2

    result = np.asarray(mask, dtype=np.float32)
    if sigma > 0:
        result = cv2.GaussianBlur(result, (0, 0), sigma)
    return np.clip(result, 0, 1)


def apply_event_specific_corrections(
    frame: np.ndarray,
    previous_output: np.ndarray,
    luminance_mask: np.ndarray,
    red_mask: np.ndarray,
    pattern_mask: np.ndarray,
    config: AdaptiveFilterConfig = AdaptiveFilterConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the smallest correction associated with each evidence channel.

    Red evidence is desaturated toward an equal-channel image with the same
    relative linear luminance. General-flash evidence is limited to a maximum
    luminance step relative to the previous output. Pattern evidence receives
    local contrast reduction around a Gaussian local mean. Soft masks avoid
    conspicuous region boundaries. Complexity is O(HW) per frame.
    """
    import cv2

    original = np.asarray(frame, dtype=np.float32)
    output = original.copy()
    zero = np.zeros_like(luminance_mask, dtype=np.float32)
    lum = _soft_mask(luminance_mask, config.feather_sigma) if config.enable_luminance else zero
    red = _soft_mask(red_mask, config.feather_sigma) if config.enable_red else zero
    pattern = _soft_mask(pattern_mask, config.feather_sigma) if config.enable_pattern else zero

    # Saturated-color events: remove chroma while preserving linear luminance.
    if np.any(red):
        gray = linear_to_srgb(np.repeat(linear_luminance(output)[..., None], 3, axis=-1))
        alpha = (np.clip(config.desaturation, 0, 1) * red)[..., None]
        output = (1 - alpha) * output + alpha * gray

    # General flashes: directly bound the temporal luminance envelope rather
    # than recursively blending frames (which produced motion trails).
    if np.any(lum):
        current_y = linear_luminance(output)
        previous_y = linear_luminance(previous_output)
        target_y = np.clip(
            current_y,
            previous_y - max(config.max_luminance_step, 0),
            previous_y + max(config.max_luminance_step, 0),
        )
        corrected_linear = np.clip(
            srgb_to_linear(output) + (target_y - current_y)[..., None], 0, 1
        )
        limited = linear_to_srgb(corrected_linear)
        output = (1 - lum[..., None]) * output + lum[..., None] * limited

    # Regular patterns: attenuate local contrast but retain the local mean.
    if np.any(pattern) and config.pattern_contrast_reduction > 0:
        local_mean = cv2.GaussianBlur(output, (0, 0), 3.0)
        reduced = local_mean + (output - local_mean) * (
            1 - np.clip(config.pattern_contrast_reduction, 0, 1)
        )
        output = (1 - pattern[..., None]) * output + pattern[..., None] * reduced

    combined = np.maximum.reduce((lum, red, pattern)).astype(np.float32)
    return np.clip(output, 0, 1).astype(np.float32), combined


def adaptive_event_filter(
    frames: np.ndarray,
    fps: float,
    *,
    filter_config: AdaptiveFilterConfig = AdaptiveFilterConfig(),
    analyzer_config=None,
) -> FilterResult:
    """Filter luminance, saturated-red, and pattern events by type."""
    from .analyzers import AnalyzerConfig, analyze_frames

    array = np.asarray(frames, dtype=np.float32)
    evidence = analyze_frames(
        array, fps, analyzer_config if analyzer_config is not None else AnalyzerConfig()
    )
    output = array.copy()
    masks = np.zeros(array.shape[:3], dtype=np.float32)
    empty = np.zeros(array.shape[1:3], dtype=bool)
    # Pattern evidence is spatial and available immediately. Applying it from
    # frame zero avoids manufacturing a one-frame temporal transient on an
    # otherwise static pattern sequence.
    output[0], masks[0] = apply_event_specific_corrections(
        array[0], array[0], empty, empty, evidence.pattern_masks[0], filter_config,
    )
    for index in range(1, len(array)):
        output[index], masks[index] = apply_event_specific_corrections(
            array[index], output[index - 1],
            evidence.luminance_flash_masks[index], evidence.red_flash_masks[index],
            evidence.pattern_masks[index], filter_config,
        )
    return FilterResult(output, masks)

def no_filter(frames: np.ndarray) -> FilterResult:
    array=np.asarray(frames,dtype=np.float32); return FilterResult(array.copy(),np.zeros(array.shape[:3],dtype=np.float32))

def global_filter(frames: np.ndarray, threshold: float=.12, blend: float=.65, luminance_mode: str="linear") -> FilterResult:
    """Blend the whole frame toward prior output when global D_t is high."""
    array=np.asarray(frames,dtype=np.float32); output=array.copy(); masks=np.zeros(array.shape[:3],dtype=np.float32); y=brightness(array,luminance_mode)
    for t in range(1,len(array)):
        if global_change_score(y[t-1],y[t]) >= threshold:
            masks[t]=1; output[t]=(1-blend)*array[t]+blend*output[t-1]
    return FilterResult(np.clip(output,0,1),masks)

def localized_filter(frames: np.ndarray, config: LocalizationConfig=LocalizationConfig(), blend: float=.65, *, cleanup: bool=True, soft: bool=True, luminance_mode: str="linear") -> FilterResult:
    """Blend only selected regions toward the prior filtered frame."""
    array=np.asarray(frames,dtype=np.float32); changes=temporal_change_maps(brightness(array,luminance_mode)); _,masks=localize(changes,config,cleanup=cleanup,soft=soft); output=array.copy()
    for t in range(1,len(array)):
        alpha=(blend*masks[t])[...,None]; output[t]=(1-alpha)*array[t]+alpha*output[t-1]
    return FilterResult(np.clip(output,0,1),masks)

def multianalyzer_filter(
    frames: np.ndarray,
    fps: float,
    *,
    blend: float = .65,
    feather_sigma: float = 1.5,
    analyzer_config=None,
) -> FilterResult:
    """Mitigate only regions supported by paired luminance or red evidence.

    Regular-pattern evidence is deliberately excluded: temporal blending does
    not remove a static pattern and would only introduce distortion. Pattern
    findings remain available through :func:`flashfilter.analyzers.analyze_frames`.
    """
    from .analyzers import AnalyzerConfig, analyze_frames, feather_event_masks

    array = np.asarray(frames, dtype=np.float32)
    config = analyzer_config if analyzer_config is not None else AnalyzerConfig()
    evidence = analyze_frames(array, fps, config)
    masks = feather_event_masks(evidence, feather_sigma, config.min_area_ratio)
    output = array.copy()
    for index in range(1, len(array)):
        alpha = (blend * masks[index])[..., None]
        output[index] = (1 - alpha) * array[index] + alpha * output[index - 1]
    return FilterResult(np.clip(output, 0, 1), masks)
