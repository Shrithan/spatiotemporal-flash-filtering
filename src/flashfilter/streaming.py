"""Constant-memory analysis and filtering for ordinary video files."""
from __future__ import annotations
from collections import deque
from pathlib import Path
import cv2
import numpy as np
from .localization import LocalizationConfig, block_activity_mask
from .luminance import brightness
from .enhanced import EnhancedConfig, EnhancedProcessor
from .framegen import FrameGenerationConfig, reconstruct_triplet
from .analyzers import AnalyzerConfig, opposing_transition_mask, regular_pattern_mask, saturated_red_signal
from .filtering import AdaptiveFilterConfig, apply_event_specific_corrections


def _open_capture(path: str | Path) -> cv2.VideoCapture:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Input does not exist: {source}")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not decode video: {source}")
    return capture


def analyze_video_stream(path: str | Path, pixel_threshold: float = .12, frame_threshold: float = .12) -> dict[str, float]:
    """Compute temporal metrics in O(HW) memory rather than loading all frames."""
    capture = _open_capture(path); previous = None; scores: list[float] = []; high_pixels = total_pixels = 0
    while True:
        ok, bgr = capture.read()
        if not ok: break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        current = brightness(rgb)
        if previous is not None:
            difference = np.abs(current - previous); scores.append(float(difference.mean()))
            high_pixels += int(np.count_nonzero(difference >= pixel_threshold)); total_pixels += difference.size
        previous = current
    capture.release()
    if previous is None: raise ValueError(f"Video contains no decodable frames: {path}")
    values = np.asarray(scores, dtype=np.float32)
    return {"mean_activity": float(values.mean()) if len(values) else 0., "peak_activity": float(values.max()) if len(values) else 0., "frames_exceeding_ratio": float(np.mean(values >= frame_threshold)) if len(values) else 0., "high_change_area_ratio": high_pixels / total_pixels if total_pixels else 0.}


def inspect_video_stream(path: str | Path, config: AnalyzerConfig = AnalyzerConfig()) -> dict[str, float | int | bool]:
    """Stream separate luminance, red, and regular-pattern evidence."""
    capture=_open_capture(path); fps=float(capture.get(cv2.CAP_PROP_FPS)) or 30.; index=0
    previous_y=previous_red=previous_lum_signed=previous_red_signed=None
    lum_times: deque[float]=deque(); red_times: deque[float]=deque()
    lum_events=red_events=peak_lum_count=peak_red_count=0
    peak_lum_area=peak_red_area=0.; pattern_run=longest_pattern_run=0
    while True:
        ok,bgr=capture.read()
        if not ok: break
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).astype(np.float32)/255
        y=brightness(rgb); red=saturated_red_signal(rgb,config.saturated_red_ratio)
        pattern_frame=rgb
        if max(rgb.shape[:2])>512:
            scale=512/max(rgb.shape[:2]); pattern_frame=cv2.resize(rgb,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        present=bool(regular_pattern_mask(pattern_frame,config).any())
        pattern_run=pattern_run+1 if present else 0; longest_pattern_run=max(longest_pattern_run,pattern_run)
        timestamp=index/fps
        while lum_times and lum_times[0] <= timestamp-1: lum_times.popleft()
        while red_times and red_times[0] <= timestamp-1: red_times.popleft()
        if previous_y is not None:
            lum_signed=y-previous_y; red_signed=red-previous_red
            lum_mask=opposing_transition_mask(previous_lum_signed,lum_signed,config.luminance_transition)
            red_mask=opposing_transition_mask(previous_red_signed,red_signed,config.red_transition)
            lum_area=float(lum_mask.mean()); red_area=float(red_mask.mean())
            peak_lum_area=max(peak_lum_area,lum_area); peak_red_area=max(peak_red_area,red_area)
            if lum_area>=config.min_area_ratio: lum_events+=1; lum_times.append(timestamp)
            if red_area>=config.min_area_ratio: red_events+=1; red_times.append(timestamp)
            peak_lum_count=max(peak_lum_count,len(lum_times)); peak_red_count=max(peak_red_count,len(red_times))
            previous_lum_signed,previous_red_signed=lum_signed,red_signed
        previous_y,previous_red=y,red; index+=1
    capture.release()
    if index==0: raise ValueError(f"Video contains no decodable frames: {path}")
    required=max(1,int(np.ceil(config.pattern_min_duration_seconds*fps)))
    return {"frames":index,"fps":fps,"luminance_flash_events":lum_events,"red_flash_events":red_events,"peak_luminance_flashes_per_second":peak_lum_count,"peak_red_flashes_per_second":peak_red_count,"peak_luminance_flash_area_ratio":peak_lum_area,"peak_red_flash_area_ratio":peak_red_area,"pattern_longest_duration_seconds":longest_pattern_run/fps,"luminance_limit_exceeded":peak_lum_count>config.flash_limit_per_second,"red_limit_exceeded":peak_red_count>config.flash_limit_per_second,"persistent_pattern_detected":longest_pattern_run>=required,"standards_compliance_claimed":False}


def filter_video_stream(
    input_path: str | Path,
    output_path: str | Path,
    method: str = "localized",
    config: LocalizationConfig = LocalizationConfig(),
    blend: float = .65,
    adaptive_config: AdaptiveFilterConfig = AdaptiveFilterConfig(),
) -> int:
    """Filter a video in constant memory and return the frame count.

    The localized path maintains only ``temporal_window`` change maps. Audio is
    not copied because OpenCV exposes video frames only.
    """
    if method not in {"global", "localized", "enhanced", "framegen", "multianalyzer", "adaptive"}: raise ValueError("unknown filtering method")
    capture = _open_capture(input_path); fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.; width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    destination = Path(output_path); destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened(): capture.release(); raise OSError(f"Could not create video writer: {destination}")
    previous_input_y = previous_output = None; history: deque[np.ndarray] = deque(maxlen=config.temporal_window); count = 0; enhanced=EnhancedProcessor(EnhancedConfig(transition_threshold=config.threshold)); triplet: deque[np.ndarray]=deque(maxlen=3)
    analyzer_config=AnalyzerConfig(luminance_transition=config.threshold)
    previous_luminance_signed=previous_red_signed=previous_red_signal=None
    kernel = np.ones((config.cleanup_kernel, config.cleanup_kernel), np.uint8)
    while True:
        ok, bgr = capture.read()
        if not ok: break
        original = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255; current_y = brightness(original)
        if method == "framegen":
            triplet.append(original)
            if len(triplet)<3:
                if count==0:
                    writer.write(bgr); count+=1
                continue
            output,_=reconstruct_triplet(triplet[0],triplet[1],triplet[2],FrameGenerationConfig(transition_threshold=config.threshold))
            encoded=cv2.cvtColor(np.round(np.clip(output,0,1)*255).astype(np.uint8),cv2.COLOR_RGB2BGR); writer.write(encoded); count+=1
            continue
        current_red_signal=saturated_red_signal(original,analyzer_config.saturated_red_ratio)
        if method == "enhanced":
            output,_,_=enhanced.process(original)
        elif method == "multianalyzer" and previous_input_y is not None:
            luminance_signed=current_y-previous_input_y
            red_signed=current_red_signal-previous_red_signal
            luminance_mask=opposing_transition_mask(previous_luminance_signed,luminance_signed,analyzer_config.luminance_transition)
            red_mask=opposing_transition_mask(previous_red_signed,red_signed,analyzer_config.red_transition)
            if float(luminance_mask.mean()) < analyzer_config.min_area_ratio: luminance_mask[:]=False
            if float(red_mask.mean()) < analyzer_config.min_area_ratio: red_mask[:]=False
            mask=np.logical_or(luminance_mask,red_mask).astype(np.float32)
            if config.feather_sigma>0: mask=cv2.GaussianBlur(mask,(0,0),config.feather_sigma)
            alpha=(blend*np.clip(mask,0,1))[...,None]; output=(1-alpha)*original+alpha*previous_output
            previous_luminance_signed,previous_red_signed=luminance_signed,red_signed
        elif method == "adaptive" and previous_input_y is not None:
            luminance_signed=current_y-previous_input_y
            red_signed=current_red_signal-previous_red_signal
            luminance_mask=opposing_transition_mask(previous_luminance_signed,luminance_signed,analyzer_config.luminance_transition)
            red_mask=opposing_transition_mask(previous_red_signed,red_signed,analyzer_config.red_transition)
            if float(luminance_mask.mean()) < analyzer_config.min_area_ratio: luminance_mask[:]=False
            if float(red_mask.mean()) < analyzer_config.min_area_ratio: red_mask[:]=False
            pattern_mask=regular_pattern_mask(original,analyzer_config)
            output,_=apply_event_specific_corrections(
                original,previous_output,luminance_mask,red_mask,pattern_mask,adaptive_config
            )
            previous_luminance_signed,previous_red_signed=luminance_signed,red_signed
        elif previous_input_y is None:
            output = original
        else:
            difference = np.abs(current_y - previous_input_y); history.append(difference)
            if method == "global":
                mask = np.ones((height, width), np.float32) if float(difference.mean()) >= config.threshold else np.zeros((height, width), np.float32)
            else:
                activity = np.mean(np.stack(history), axis=0); mask = block_activity_mask(activity, config.threshold, config.block_size)
                if config.cleanup_kernel > 1: mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                if config.feather_sigma > 0: mask = cv2.GaussianBlur(mask, (0, 0), config.feather_sigma)
            alpha = (blend * np.clip(mask, 0, 1))[..., None]; output = (1 - alpha) * original + alpha * previous_output
        encoded = cv2.cvtColor(np.round(np.clip(output, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR); writer.write(encoded)
        if method in {"multianalyzer","adaptive"} and previous_input_y is not None and previous_luminance_signed is None:
            previous_luminance_signed=current_y-previous_input_y
            previous_red_signed=current_red_signal-previous_red_signal
        previous_input_y, previous_output, previous_red_signal = current_y, output, current_red_signal; count += 1
    if method=="framegen" and len(triplet)>=2:
        final=cv2.cvtColor(np.round(np.clip(triplet[-1],0,1)*255).astype(np.uint8),cv2.COLOR_RGB2BGR); writer.write(final); count+=1
    capture.release(); writer.release()
    if count == 0: raise ValueError(f"Video contains no decodable frames: {input_path}")
    if not destination.exists() or destination.stat().st_size == 0: raise OSError(f"Video encoder produced no output: {destination}")
    return count
