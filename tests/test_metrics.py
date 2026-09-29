import math
import numpy as np
import pytest
from flashfilter.metrics import distortion_metrics, localization_metrics, modified_area_ratio

def test_distortion_known_values():
    a=np.zeros((2,8,8,3),dtype=np.float32); b=np.full_like(a,.5); metrics=distortion_metrics(a,b)
    assert metrics["mae"]==pytest.approx(.5) and metrics["mse"]==pytest.approx(.25) and metrics["psnr"]==pytest.approx(10*np.log10(4))

def test_identical_distortion_is_perfect():
    a=np.zeros((1,8,8,3),dtype=np.float32); metrics=distortion_metrics(a,a)
    assert metrics["mae"]==0 and math.isinf(metrics["psnr"]) and metrics["ssim"]==1

def test_localization_confusion_counts():
    truth=np.array([[1,1],[0,0]],bool); pred=np.array([[1,0],[1,0]],bool); metrics=localization_metrics(pred,truth)
    assert metrics["precision"]==.5 and metrics["recall"]==.5 and metrics["iou"]==pytest.approx(1/3)

def test_modified_area_counts_pixels_not_channels():
    a=np.zeros((1,2,2,3)); b=a.copy(); b[0,0,0,1]=1
    assert modified_area_ratio(a,b)==.25
