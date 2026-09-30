"""Generate non-autoplay research demo assets from canonical project outputs.

The demo intentionally uses held frames and gentle fades instead of replaying
rapid alternation. It communicates the algorithm without embedding a fast
flashing stimulus in the README asset.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image

from flashfilter.filtering import adaptive_event_filter, global_filter, localized_filter
from flashfilter.luminance import linear_luminance
from flashfilter.localization import LocalizationConfig
from flashfilter.synthetic import generate_cases


WIDTH, HEIGHT, FPS = 960, 540, 24
BACKGROUND = np.full((HEIGHT, WIDTH, 3), (248, 250, 252), np.uint8)
INK = (31, 41, 55)
BLUE = (190, 105, 37)


def _text(frame: np.ndarray, text: str, origin: tuple[int, int], scale: float = 0.7,
          color: tuple[int, int, int] = INK, thickness: int = 1) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_DUPLEX, scale, color,
                thickness, cv2.LINE_AA)


def _centered(frame: np.ndarray, text: str, y: int, scale: float = 1.0,
              color: tuple[int, int, int] = INK, thickness: int = 2) -> None:
    width = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)[0][0]
    _text(frame, text, ((WIDTH - width) // 2, y), scale, color, thickness)


def _panel(canvas: np.ndarray, rgb: np.ndarray, x: int, y: int, width: int, height: int,
           title: str) -> None:
    image = cv2.cvtColor(np.clip(rgb * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    canvas[y:y + height, x:x + width] = image
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (75, 85, 99), 1)
    title_width = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, .55, 1)[0][0]
    _text(canvas, title, (x + (width - title_width) // 2, y - 12), .55)


def _title_slide() -> np.ndarray:
    frame = BACKGROUND.copy()
    _centered(frame, "Spatiotemporal Flash Filtering", 215, 1.25, thickness=2)
    _centered(frame, "Localized detection and suppression of rapid visual changes", 260, .66, BLUE, 1)
    _centered(frame, "Research prototype - computational visual-change proxies", 335, .55, thickness=1)
    return frame


def _detection_slide(case, localized) -> np.ndarray:
    frame = BACKGROUND.copy()
    index = 2
    change = np.abs(linear_luminance(case.frames[index]) - linear_luminance(case.frames[index - 1]))
    heat = cv2.applyColorMap(np.clip(change * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    mask = np.repeat(localized.masks[index][..., None], 3, axis=-1)
    _centered(frame, "Spatial evidence on a controlled localized event", 48, .78, thickness=2)
    _panel(frame, case.frames[index], 35, 125, 275, 275, "Source frame")
    _panel(frame, heat, 342, 125, 275, 275, "|Delta luminance| heatmap")
    _panel(frame, mask, 649, 125, 275, 275, "Feathered detection mask")
    _centered(frame, "The detector selects a region instead of averaging it away.", 465, .58)
    return frame


def _comparison_slide(case, global_result, localized, adaptive) -> np.ndarray:
    frame = BACKGROUND.copy()
    index = 4
    _centered(frame, "Same input, different spatial responses", 48, .8, thickness=2)
    items = (
        (case.frames[index], "Source"), (global_result.frames[index], "Global baseline"),
        (localized.frames[index], "Localized baseline"), (adaptive.frames[index], "Adaptive method"),
    )
    for column, (image, title) in enumerate(items):
        _panel(frame, image, 24 + column * 234, 145, 210, 210, title)
    _centered(frame, "Global filtering changes the full frame; local methods preserve untouched pixels.", 435, .52)
    return frame


def _image_slide(path: Path, title: str, footer: str) -> np.ndarray:
    frame = BACKGROUND.copy()
    _centered(frame, title, 42, .76, thickness=2)
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    scale = min(850 / image.shape[1], 410 / image.shape[0])
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    x = (WIDTH - resized.shape[1]) // 2
    y = 65 + (410 - resized.shape[0]) // 2
    frame[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    _centered(frame, footer, 515, .45)
    return frame


def _end_slide(metrics: dict) -> np.ndarray:
    frame = np.full((HEIGHT, WIDTH, 3), (31, 41, 55), np.uint8)
    _centered(frame, "Spatiotemporal Flash Filtering", 145, 1.08, (255, 255, 255), 2)
    _centered(frame, "Detection - Localization - Adaptive Filtering", 205, .63, (209, 250, 229), 1)
    _centered(frame, "Synthetic benchmarks - Ablations - Natural-media case study", 250, .54, (229, 231, 235), 1)
    _centered(frame, f"12 scenarios | {metrics['sweep_points']} sweep operating points | reproducible CSV results", 315, .53, (191, 219, 254), 1)
    _centered(frame, "github.com/Shrithan/spatiotemporal-flash-filtering", 380, .57, (255, 255, 255), 1)
    _centered(frame, "Research prototype - not clinically validated", 455, .5, (203, 213, 225), 1)
    return frame


def _append_hold(frames: list[np.ndarray], slide: np.ndarray, seconds: float) -> None:
    frames.extend([slide] * int(round(seconds * FPS)))


def _append_fade(frames: list[np.ndarray], first: np.ndarray, second: np.ndarray, seconds: float = .5) -> None:
    count = max(2, int(round(seconds * FPS)))
    for alpha in np.linspace(0, 1, count):
        frames.append(cv2.addWeighted(first, 1 - float(alpha), second, float(alpha), 0))


def generate(output_dir: Path) -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    output_dir.mkdir(parents=True, exist_ok=True)
    case = generate_cases(seed=7)["medium_flashing_region"]
    localized = localized_filter(case.frames, LocalizationConfig(), .65)
    global_result = global_filter(case.frames)
    adaptive = adaptive_event_filter(case.frames, case.fps)
    summary = json.loads((root / "experiments/results/canonical/sweep/sweep_definition.json").read_text())
    import pandas as pd
    sweep_rows = pd.read_csv(root / "experiments/results/canonical/sweep/parameter_sweep_summary.csv")
    metrics = {"sweep_points": int(len(sweep_rows)), "seed": summary["seed"]}

    slides = [
        _title_slide(),
        _detection_slide(case, localized),
        _comparison_slide(case, global_result, localized, adaptive),
        _image_slide(root / "experiments/results/canonical/sweep/suppression_distortion_pareto.png",
                     "Measured suppression-distortion tradeoff",
                     "Every point comes from the deterministic 12-scenario parameter sweep."),
        _image_slide(root / "paper/figures/adaptive_architecture.png",
                     "Event-specific adaptive architecture",
                     "Separate evidence channels map to separate local corrections."),
        _end_slide(metrics),
    ]
    durations = [4, 8, 9, 8, 6, 5]
    frames: list[np.ndarray] = []
    for index, (slide, duration) in enumerate(zip(slides, durations)):
        if index:
            _append_fade(frames, slides[index - 1], slide)
        _append_hold(frames, slide, duration - (.5 if index else 0))

    temporary = output_dir / "demo_mpeg4.mp4"
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the demo video")
    for frame in frames:
        writer.write(frame)
    writer.release()
    video = output_dir / "demo.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(temporary),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(video)], check=True)
        temporary.unlink()
    else:
        temporary.replace(video)

    # Ten-second, five-frame-per-second README asset. Each view is held for
    # five seconds; no rapid source alternation is replayed.
    gif_frames = []
    for slide in (slides[1], slides[2]):
        rgb = cv2.cvtColor(cv2.resize(slide, (640, 360)), cv2.COLOR_BGR2RGB)
        gif_frames.extend([Image.fromarray(rgb)] * 25)
    gif = output_dir / "demo.gif"
    gif_frames[0].save(gif, save_all=True, append_images=gif_frames[1:], duration=200,
                       loop=0, optimize=True)
    (output_dir / "demo_metadata.json").write_text(json.dumps({
        "synthetic_case": case.name,
        "seed": 7,
        "full_demo_seconds": len(frames) / FPS,
        "gif_seconds": 10,
        "rapid_flashing_replayed": False,
        "source_results": "experiments/results/canonical/sweep",
        **metrics,
    }, indent=2) + "\n")
    return video, gif


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    video, gif = generate(root / "docs/assets")
    print(video)
    print(gif)
