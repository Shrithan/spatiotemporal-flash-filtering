
"""Temporal-change maps and summary scores."""
import numpy as np

def temporal_difference(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Compute Delta_t(x,y) = |Y_t(x,y) - Y_(t-1)(x,y)|."""
    previous_array, current_array = np.asarray(previous, dtype=np.float32), np.asarray(current, dtype=np.float32)
    if previous_array.shape != current_array.shape: raise ValueError("Brightness frames must have matching shapes")
    if previous_array.ndim != 2: raise ValueError("Expected two-dimensional brightness frames")
    return np.abs(current_array - previous_array)

def global_change_score(previous: np.ndarray, current: np.ndarray) -> float:
    """Compute the spatial mean global score D_t."""
    return float(np.mean(temporal_difference(previous, current)))

def temporal_change_maps(brightness_frames: np.ndarray) -> np.ndarray:
    """Return one change map per frame, with an all-zero first map."""
    frames = np.asarray(brightness_frames, dtype=np.float32)
    if frames.ndim != 3: raise ValueError("Expected brightness video with shape (T, H, W)")
    differences = np.zeros_like(frames)
    differences[1:] = np.abs(np.diff(frames, axis=0))
    return differences

def global_change_scores(brightness_frames: np.ndarray) -> np.ndarray:
    """Return global score D_t for every frame (zero at t=0)."""
    return temporal_change_maps(brightness_frames).mean(axis=(1, 2), dtype=np.float64).astype(np.float32)

def sustained_activity(change_maps: np.ndarray, window: int = 2) -> np.ndarray:
    """Average recent difference maps; complexity is O(THW)."""
    maps = np.asarray(change_maps, dtype=np.float32)
    if maps.ndim != 3 or window < 1: raise ValueError("Expected (T,H,W) maps and window >= 1")
    result = np.empty_like(maps)
    cumulative = np.concatenate([np.zeros_like(maps[:1]), np.cumsum(maps, axis=0)], axis=0)
    for index in range(len(maps)):
        start = max(0, index - window + 1)
        result[index] = (cumulative[index + 1] - cumulative[start]) / (index - start + 1)
    return result
