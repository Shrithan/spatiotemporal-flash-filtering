"""Research candidate combining reversal detection and luminance clamping."""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from .filtering import FilterResult
from .localization import block_activity_mask
from .luminance import linear_luminance, linear_to_srgb, srgb_to_linear

@dataclass(frozen=True)
class EnhancedConfig:
    """Parameters for the enhanced, interpretable filter candidate."""
    transition_threshold: float = .04
    block_sizes: tuple[int, ...] = (4, 8)
    min_reversal_fraction: float = .15
    max_luminance_step: float = .08
    scene_cut_distance: float = .65
    feather_sigma: float = 1.5

def histogram_distance(previous_y: np.ndarray, current_y: np.ndarray, bins: int = 32) -> float:
    """Return total-variation distance between normalized luminance histograms."""
    previous=np.histogram(previous_y,bins=bins,range=(0,1))[0].astype(np.float32); current=np.histogram(current_y,bins=bins,range=(0,1))[0].astype(np.float32)
    previous/=max(float(previous.sum()),1); current/=max(float(current.sum()),1)
    return float(.5*np.abs(previous-current).sum())

def _clamp_luminance(rgb: np.ndarray, target_y: np.ndarray) -> np.ndarray:
    """Change linear luminance while approximately preserving chromaticity."""
    linear=srgb_to_linear(rgb); current_y=linear_luminance(rgb); ratio=target_y/np.maximum(current_y,1e-6); adjusted=linear*ratio[...,None]
    dark=current_y<1e-6
    if np.any(dark): adjusted[dark]=target_y[dark,None]
    return linear_to_srgb(np.clip(adjusted,0,1))

class EnhancedProcessor:
    """Stateful O(HW)-memory processor used by batch and streaming paths."""
    def __init__(self, config: EnhancedConfig = EnhancedConfig()):
        self.config=config; self.previous_input_y=None; self.previous_signed=None; self.previous_output_y=None

    def process(self, rgb: np.ndarray) -> tuple[np.ndarray,np.ndarray,bool]:
        current=np.asarray(rgb,dtype=np.float32); current_y=linear_luminance(current)
        if self.previous_input_y is None:
            self.previous_input_y=current_y; self.previous_output_y=current_y; return current.copy(),np.zeros_like(current_y),False
        signed=current_y-self.previous_input_y; magnitude=np.abs(signed); reversed_pixels=np.zeros_like(current_y,dtype=bool)
        if self.previous_signed is not None:
            reversed_pixels=(signed*self.previous_signed<0)&(magnitude>=self.config.transition_threshold)&(np.abs(self.previous_signed)>=self.config.transition_threshold)
        cut=histogram_distance(self.previous_input_y,current_y)>=self.config.scene_cut_distance and float(reversed_pixels.mean())<self.config.min_reversal_fraction
        mask=np.zeros_like(current_y,dtype=np.float32)
        if not cut and self.previous_signed is not None:
            reversal_activity=magnitude*reversed_pixels
            for size in self.config.block_sizes: mask=np.maximum(mask,block_activity_mask(reversal_activity,self.config.transition_threshold*self.config.min_reversal_fraction,size))
            if self.config.feather_sigma>0: mask=cv2.GaussianBlur(mask,(0,0),self.config.feather_sigma)
        delta=current_y-self.previous_output_y; clamped=self.previous_output_y+np.clip(delta,-self.config.max_luminance_step,self.config.max_luminance_step); adjusted=_clamp_luminance(current,clamped); output=(1-mask[...,None])*current+mask[...,None]*adjusted
        # Keep the transition polarity even when this frame resembles a cut.
        # A static post-cut frame has no reversal, while a flash returning in
        # the opposite direction can now be recognized on the next frame.
        self.previous_input_y=current_y; self.previous_signed=signed; self.previous_output_y=linear_luminance(output)
        return np.clip(output,0,1).astype(np.float32),np.clip(mask,0,1).astype(np.float32),cut

def enhanced_filter(frames: np.ndarray, config: EnhancedConfig = EnhancedConfig()) -> FilterResult:
    """Filter an RGB video with opposing-transition masks and luma clamping."""
    array=np.asarray(frames,dtype=np.float32); processor=EnhancedProcessor(config); outputs=[]; masks=[]
    for frame in array:
        output,mask,_=processor.process(frame); outputs.append(output); masks.append(mask)
    return FilterResult(np.stack(outputs),np.stack(masks))
