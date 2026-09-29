import numpy as np
from flashfilter.framegen import FrameGenerationConfig, framegen_filter, reconstruction_mask

def test_middle_excursion_is_reconstructed_from_matching_endpoints():
    white=np.ones((8,8,3),np.float32); black=np.zeros_like(white)
    result=framegen_filter(np.stack([white,black,white]),FrameGenerationConfig(.04,.1,(4,),.1,0,8))
    assert np.all(result.masks[1]==1)
    assert float(result.frames[1].mean())>.99

def test_monotonic_transition_is_not_reconstructed():
    a=np.zeros((8,8,3),np.float32); b=np.full_like(a,.5); c=np.ones_like(a)
    assert np.all(reconstruction_mask(a,b,c,FrameGenerationConfig(.04,.1,(4,),.1,0,8))==0)

def test_frame_count_and_endpoints_are_preserved():
    frames=np.zeros((4,8,8,3),np.float32); frames[1]=1
    result=framegen_filter(frames,FrameGenerationConfig(.04,.1,(4,),.1,0,8))
    assert result.frames.shape==frames.shape
    assert np.array_equal(result.frames[0],frames[0]) and np.array_equal(result.frames[-1],frames[-1])
