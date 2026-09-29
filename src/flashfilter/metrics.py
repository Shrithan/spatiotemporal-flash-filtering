"""Joint temporal-activity, distortion, localization, and area metrics."""
from __future__ import annotations
import math
import numpy as np
from skimage.metrics import structural_similarity
from .luminance import brightness
from .temporal import temporal_change_maps, global_change_scores

def distortion_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str,float]:
    """Return MAE, MSE, PSNR, and mean frame SSIM for RGB data in [0,1]."""
    reference=np.asarray(reference,dtype=np.float32); estimate=np.asarray(estimate,dtype=np.float32)
    if reference.shape != estimate.shape: raise ValueError("Videos must have matching shapes")
    error=estimate-reference; mae=float(np.mean(np.abs(error))); mse=float(np.mean(error**2)); psnr=math.inf if mse==0 else float(10*np.log10(1/mse))
    win=min(7,reference.shape[1],reference.shape[2]); win=win if win%2 else win-1
    ssim=float(np.mean([structural_similarity(reference[t],estimate[t],channel_axis=-1,data_range=1.0,win_size=win) for t in range(len(reference))]))
    return {"mae":mae,"mse":mse,"psnr":psnr,"ssim":ssim}

def temporal_metrics(frames: np.ndarray, pixel_threshold: float=.12, frame_threshold: float=.12) -> dict[str,float]:
    """Summarize the project's computational temporal-activity proxy."""
    y=brightness(frames); maps=temporal_change_maps(y); scores=global_change_scores(y); evaluated=scores[1:]
    return {"mean_activity":float(evaluated.mean()) if len(evaluated) else 0.,"peak_activity":float(evaluated.max()) if len(evaluated) else 0.,"frames_exceeding_ratio":float(np.mean(evaluated>=frame_threshold)) if len(evaluated) else 0.,"high_change_area_ratio":float(np.mean(maps[1:]>=pixel_threshold)) if len(evaluated) else 0.}

def localization_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str,float]:
    """Return pixel precision, recall, F1, and IoU with defined empty cases."""
    pred=np.asarray(predicted)>0.5; target=np.asarray(truth,dtype=bool)
    if pred.shape!=target.shape: raise ValueError("Masks must have matching shapes")
    tp=np.count_nonzero(pred&target); fp=np.count_nonzero(pred&~target); fn=np.count_nonzero(~pred&target)
    precision=tp/(tp+fp) if tp+fp else (1. if not np.any(target) else 0.)
    recall=tp/(tp+fn) if tp+fn else (1. if not np.any(pred) else 0.)
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.; union=tp+fp+fn
    return {"precision":float(precision),"recall":float(recall),"f1":float(f1),"iou":float(tp/union if union else 1.)}

def modified_area_ratio(reference: np.ndarray, estimate: np.ndarray, epsilon: float=1/255) -> float:
    """Fraction of pixel locations with any channel changed beyond epsilon."""
    return float(np.mean(np.max(np.abs(np.asarray(reference)-np.asarray(estimate)),axis=-1)>epsilon))

def residual_inside_regions(original: np.ndarray, filtered: np.ndarray, regions: np.ndarray) -> float:
    """Ratio of output to input activity inside supplied spatiotemporal regions."""
    before=temporal_change_maps(brightness(original)); after=temporal_change_maps(brightness(filtered)); selected=np.asarray(regions,dtype=bool)
    denominator=float(before[selected].sum()); return float(after[selected].sum()/denominator) if denominator else 0.
