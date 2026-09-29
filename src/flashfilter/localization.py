"""Explainable block-based temporal-activity localization."""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from .temporal import sustained_activity

@dataclass(frozen=True)
class LocalizationConfig:
    threshold: float = 0.12
    block_size: int = 8
    temporal_window: int = 2
    cleanup_kernel: int = 3
    feather_sigma: float = 2.0

def block_activity_mask(activity: np.ndarray, threshold: float, block_size: int) -> np.ndarray:
    """Select complete blocks whose mean activity meets ``threshold``."""
    if activity.ndim != 2 or block_size < 1: raise ValueError("Expected 2-D activity and positive block_size")
    # Pad only the bottom/right edges, then use a reshape reduction.  This is
    # equivalent to the readable nested-loop formulation but avoids thousands
    # of Python iterations on HD/4K frames.
    height,width=activity.shape
    padded_height=((height+block_size-1)//block_size)*block_size
    padded_width=((width+block_size-1)//block_size)*block_size
    padded=np.full((padded_height,padded_width),np.nan,dtype=np.float32)
    padded[:height,:width]=activity
    grid=np.nanmean(
        padded.reshape(padded_height//block_size,block_size,padded_width//block_size,block_size),
        axis=(1,3),
    )
    selected=grid>=threshold
    return np.repeat(np.repeat(selected,block_size,axis=0),block_size,axis=1)[:height,:width].astype(np.float32)

def localize(change_maps: np.ndarray, config: LocalizationConfig = LocalizationConfig(), *, cleanup: bool = True, soft: bool = True) -> tuple[np.ndarray,np.ndarray]:
    """Return hard decision masks and optionally feathered masks in [0,1]."""
    activity=sustained_activity(change_maps,config.temporal_window); hard=np.zeros_like(activity,dtype=np.float32)
    kernel=np.ones((config.cleanup_kernel,config.cleanup_kernel),np.uint8)
    for t in range(1,len(activity)):
        hard[t]=block_activity_mask(activity[t],config.threshold,config.block_size)
        if cleanup and config.cleanup_kernel>1: hard[t]=cv2.morphologyEx(hard[t],cv2.MORPH_CLOSE,kernel)
    if not soft or config.feather_sigma<=0: return hard,hard.copy()
    soft_masks=np.stack([cv2.GaussianBlur(mask,(0,0),config.feather_sigma) for mask in hard])
    return hard,np.clip(soft_masks,0,1)
