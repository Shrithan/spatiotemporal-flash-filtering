"""Stable public API for analysis and filtering applications.

Application code should import this module (or the package-level re-exports)
instead of depending on internal streaming and analyzer implementation details.
All reported values are computational proxies; no result represents clinical
validation or a guarantee of medical safety or standards conformance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2

from .analyzers import AnalyzerConfig, analyze_frames
from .filtering import AdaptiveFilterConfig, adaptive_event_filter
from .localization import LocalizationConfig
from .metrics import temporal_metrics
from .streaming import analyze_video_stream, filter_video_stream, inspect_video_stream
from .video import read_video, write_video


@dataclass(frozen=True)
class AnalysisReport:
    """Structured temporal, event-channel, and media metadata."""

    source: str
    frame_count: int
    fps: float
    duration_seconds: float
    width: int
    height: int
    temporal: dict[str, float]
    evidence: dict[str, float | int | bool]
    medical_safety_claimed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class FilterReport:
    """Structured record of one completed filtering operation."""

    source: str
    output: str
    method: str
    frames_processed: int
    source_width: int
    source_height: int
    output_width: int
    output_height: int
    source_fps: float
    output_fps: float
    resolution_preserved: bool
    fps_preserved: bool
    output_analysis: AnalysisReport | None
    medical_safety_claimed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _video_metadata(path: str | Path) -> tuple[int, int, int, float]:
    source = Path(path)
    if source.suffix.lower() == ".npz":
        video = read_video(source)
        frames, height, width = video.frames.shape[:3]
        return int(frames), int(width), int(height), float(video.fps)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {source}")
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    capture.release()
    return frames, width, height, fps


def analyze_video(
    path: str | Path,
    *,
    analyzer_config: AnalyzerConfig = AnalyzerConfig(),
) -> AnalysisReport:
    """Analyze a video through the supported, structured public interface."""
    source = Path(path)
    frames, width, height, fps = _video_metadata(source)
    if source.suffix.lower() == ".npz":
        video = read_video(source)
        temporal = temporal_metrics(video.frames)
        evidence = analyze_frames(video.frames, video.fps, analyzer_config).summary
    else:
        temporal = analyze_video_stream(source)
        evidence = inspect_video_stream(source, analyzer_config)
    return AnalysisReport(
        source=str(source), frame_count=frames, fps=fps,
        duration_seconds=frames / fps if fps else 0.0,
        width=width, height=height, temporal=temporal, evidence=evidence,
    )


def filter_video(
    input_path: str | Path,
    output_path: str | Path,
    *,
    method: str = "adaptive",
    adaptive_config: AdaptiveFilterConfig = AdaptiveFilterConfig(),
    localization_config: LocalizationConfig = LocalizationConfig(),
    blend: float = 0.65,
    analyze_output: bool = True,
) -> FilterReport:
    """Filter one video and return verifiable output metadata.

    MP4 processing uses the constant-memory path. OpenCV does not copy audio;
    applications that require audio must remux it explicitly with FFmpeg.
    """
    source = Path(input_path)
    destination = Path(output_path)
    source_frames, source_width, source_height, source_fps = _video_metadata(source)
    if source.suffix.lower() == ".npz":
        video = read_video(source)
        if method != "adaptive":
            raise ValueError("The public lossless NPZ path currently supports method='adaptive'")
        result = adaptive_event_filter(video.frames, video.fps, filter_config=adaptive_config)
        write_video(destination, result.frames, video.fps)
        processed = len(result.frames)
    else:
        processed = filter_video_stream(
            source, destination, method, localization_config, blend, adaptive_config
        )
    _, output_width, output_height, output_fps = _video_metadata(destination)
    output_analysis = analyze_video(destination) if analyze_output else None
    return FilterReport(
        source=str(source), output=str(destination), method=method,
        frames_processed=processed, source_width=source_width,
        source_height=source_height, output_width=output_width,
        output_height=output_height, source_fps=source_fps, output_fps=output_fps,
        resolution_preserved=(source_width, source_height) == (output_width, output_height),
        fps_preserved=abs(source_fps - output_fps) <= 0.02,
        output_analysis=output_analysis,
    )
