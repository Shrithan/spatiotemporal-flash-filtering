"""Video I/O using RGB float frames in [0, 1]."""
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

@dataclass(frozen=True)
class VideoData:
    frames: np.ndarray
    fps: float

def read_video(path: str | Path) -> VideoData:
    """Decode a video or lossless benchmark .npz file."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"Input does not exist: {source}. Generate or filter it before analysis."
        )
    if source.suffix.lower() == ".npz":
        with np.load(source) as data: return VideoData(data["frames"].astype(np.float32), float(data["fps"]))
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened(): raise FileNotFoundError(f"Could not open video: {source}")
    fps, frames = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0, []
    while True:
        ok, bgr = capture.read()
        if not ok: break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    capture.release()
    if not frames: raise ValueError(f"Video contains no decodable frames: {source}")
    return VideoData(np.stack(frames), fps)

def write_video(path: str | Path, frames: np.ndarray, fps: float = 30.0) -> None:
    """Write RGB float frames; NPZ is lossless and MP4 uses MP4V."""
    destination, array = Path(path), np.asarray(frames, dtype=np.float32)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim != 4 or array.shape[-1] != 3: raise ValueError("Expected frames with shape (T, H, W, 3)")
    if destination.suffix.lower() == ".npz":
        np.savez_compressed(destination, frames=np.clip(array, 0, 1), fps=float(fps)); return
    height, width = array.shape[1:3]
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened(): raise OSError(f"Could not create video writer: {destination}")
    for rgb in np.clip(array, 0, 1): writer.write(cv2.cvtColor(np.round(rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    writer.release()
    if not destination.exists() or destination.stat().st_size == 0:
        raise OSError(f"Video encoder produced no output: {destination}")
