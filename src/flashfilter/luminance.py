"""Brightness representations for normalized RGB video frames."""
from __future__ import annotations
import numpy as np

def _validate_rgb(rgb: np.ndarray) -> np.ndarray:
    array = np.asarray(rgb, dtype=np.float32)
    if array.ndim < 3 or array.shape[-1] != 3:
        raise ValueError("Expected an RGB array with final dimension 3")
    if not np.all(np.isfinite(array)):
        raise ValueError("RGB data must contain only finite values")
    if array.size and (array.min() < 0.0 or array.max() > 1.0):
        raise ValueError("RGB data must be normalized to [0, 1]")
    return array

def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Invert the sRGB transfer function for normalized RGB data."""
    encoded = _validate_rgb(rgb)
    return np.where(encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4).astype(np.float32)

def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    """Apply the sRGB transfer function to linear RGB values in [0,1]."""
    linear = np.clip(np.asarray(rgb, dtype=np.float32), 0, 1)
    return np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055).astype(np.float32)

def linear_luminance(rgb: np.ndarray) -> np.ndarray:
    """Return relative linear luminance using Rec. 709 coefficients."""
    return np.tensordot(srgb_to_linear(rgb), np.array([0.2126, 0.7152, 0.0722]), axes=(-1, 0)).astype(np.float32)

def gamma_luma_proxy(rgb: np.ndarray) -> np.ndarray:
    """Return weighted gamma-encoded brightness (a proxy, not luminance)."""
    return np.tensordot(_validate_rgb(rgb), np.array([0.2126, 0.7152, 0.0722]), axes=(-1, 0)).astype(np.float32)

def brightness(rgb: np.ndarray, mode: str = "linear") -> np.ndarray:
    """Convert RGB frame(s) using ``linear`` or ``gamma_proxy`` mode."""
    if mode == "linear": return linear_luminance(rgb)
    if mode == "gamma_proxy": return gamma_luma_proxy(rgb)
    raise ValueError("mode must be 'linear' or 'gamma_proxy'")
