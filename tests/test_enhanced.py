import numpy as np
from flashfilter.enhanced import EnhancedConfig, enhanced_filter, histogram_distance
from flashfilter.luminance import linear_luminance

def test_histogram_distance_identical_and_disjoint():
    black=np.zeros((8,8),np.float32); white=np.ones((8,8),np.float32)
    assert histogram_distance(black,black)==0 and histogram_distance(black,white)==1

def test_reversal_filter_ignores_first_step_then_limits_reversal():
    frames=np.stack([np.zeros((8,8,3)),np.ones((8,8,3)),np.zeros((8,8,3))]).astype(np.float32)
    result=enhanced_filter(frames,EnhancedConfig(.04,(4,),.1,.08,.65,0))
    assert np.all(result.masks[1]==0)
    assert np.all(result.masks[2]==1)
    assert float(linear_luminance(result.frames[2]).mean()) > 0
