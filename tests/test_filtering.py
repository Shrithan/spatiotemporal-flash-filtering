import numpy as np
from flashfilter.filtering import global_filter, localized_filter
from flashfilter.localization import LocalizationConfig

def test_global_filter_blends_entire_triggered_frame():
    frames=np.stack([np.zeros((4,4,3)),np.ones((4,4,3))]).astype(np.float32)
    result=global_filter(frames,threshold=.1,blend=.5)
    assert np.allclose(result.frames[1],.5) and np.all(result.masks[1]==1)

def test_localized_filter_leaves_inactive_pixels_unchanged():
    frames=np.zeros((2,8,8,3),dtype=np.float32); frames[1,4:,4:]=1
    result=localized_filter(frames,LocalizationConfig(.1,4,1,1,0),blend=.5,cleanup=False,soft=False)
    assert np.allclose(result.frames[1,:4,:4],frames[1,:4,:4])
    assert np.allclose(result.frames[1,4:,4:],.5)
