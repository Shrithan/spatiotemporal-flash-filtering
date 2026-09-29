"""Flash-aware frame reconstruction using one-frame lookahead and optical flow."""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from .filtering import FilterResult
from .localization import block_activity_mask
from .luminance import linear_luminance

@dataclass(frozen=True)
class FrameGenerationConfig:
    transition_threshold: float = .04
    endpoint_tolerance: float = .10
    block_sizes: tuple[int, ...] = (4, 8)
    min_active_fraction: float = .15
    feather_sigma: float = 1.5
    analysis_width: int = 320

def _resize_for_analysis(rgb: np.ndarray, width: int) -> tuple[np.ndarray,float,float]:
    height,source_width=rgb.shape[:2]
    if width<=0 or source_width<=width: return rgb,1.,1.
    target_height=max(2,round(height*width/source_width)); resized=cv2.resize(rgb,(width,target_height),interpolation=cv2.INTER_AREA)
    return resized,source_width/width,height/target_height

def motion_midpoint(previous: np.ndarray, following: np.ndarray, analysis_width: int = 320) -> np.ndarray:
    """Synthesize the temporal midpoint using bidirectional Farneback flow."""
    previous_small,scale_x,scale_y=_resize_for_analysis(previous,analysis_width); following_small,_,_=_resize_for_analysis(following,analysis_width)
    previous_gray=cv2.cvtColor(np.round(previous_small*255).astype(np.uint8),cv2.COLOR_RGB2GRAY); following_gray=cv2.cvtColor(np.round(following_small*255).astype(np.uint8),cv2.COLOR_RGB2GRAY)
    parameters=(.5,3,15,3,5,1.2,0)
    forward=cv2.calcOpticalFlowFarneback(previous_gray,following_gray,None,*parameters)
    backward=cv2.calcOpticalFlowFarneback(following_gray,previous_gray,None,*parameters)
    height,width=previous.shape[:2]
    if forward.shape[:2]!=(height,width):
        forward=cv2.resize(forward,(width,height),interpolation=cv2.INTER_LINEAR); backward=cv2.resize(backward,(width,height),interpolation=cv2.INTER_LINEAR)
        forward[...,0]*=scale_x; forward[...,1]*=scale_y; backward[...,0]*=scale_x; backward[...,1]*=scale_y
    grid_x,grid_y=np.meshgrid(np.arange(width,dtype=np.float32),np.arange(height,dtype=np.float32))
    warped_previous=cv2.remap(previous,grid_x-.5*forward[...,0],grid_y-.5*forward[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    warped_following=cv2.remap(following,grid_x-.5*backward[...,0],grid_y-.5*backward[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
    return ((warped_previous+warped_following)*.5).astype(np.float32)

def reconstruction_mask(previous: np.ndarray, current: np.ndarray, following: np.ndarray, config: FrameGenerationConfig = FrameGenerationConfig()) -> np.ndarray:
    """Find regions where B is an opposing excursion between similar A and C."""
    a,b,c=linear_luminance(previous),linear_luminance(current),linear_luminance(following)
    into=b-a; out=c-b
    opposing=(into*out<0)&(np.abs(into)>=config.transition_threshold)&(np.abs(out)>=config.transition_threshold)
    endpoints=np.abs(c-a)<=config.endpoint_tolerance
    activity=np.minimum(np.abs(into),np.abs(out))*opposing*endpoints
    mask=np.zeros_like(a,dtype=np.float32)
    for size in config.block_sizes:
        mask=np.maximum(mask,block_activity_mask(activity,config.transition_threshold*config.min_active_fraction,size))
    if config.feather_sigma>0: mask=cv2.GaussianBlur(mask,(0,0),config.feather_sigma)
    return np.clip(mask,0,1).astype(np.float32)

def reconstruct_triplet(previous: np.ndarray, current: np.ndarray, following: np.ndarray, config: FrameGenerationConfig = FrameGenerationConfig()) -> tuple[np.ndarray,np.ndarray]:
    """Reconstruct selected regions of B from motion-aligned neighbors A and C."""
    mask=reconstruction_mask(previous,current,following,config)
    if not np.any(mask>1e-6): return current.copy(),mask
    generated=motion_midpoint(previous,following,config.analysis_width)
    output=(1-mask[...,None])*current+mask[...,None]*generated
    return np.clip(output,0,1).astype(np.float32),mask

def framegen_filter(frames: np.ndarray, config: FrameGenerationConfig = FrameGenerationConfig()) -> FilterResult:
    """Apply triplet reconstruction while preserving frame count and FPS."""
    array=np.asarray(frames,dtype=np.float32); output=array.copy(); masks=np.zeros(array.shape[:3],dtype=np.float32)
    for index in range(1,len(array)-1): output[index],masks[index]=reconstruct_triplet(array[index-1],array[index],array[index+1],config)
    return FilterResult(output,masks)
