import numpy as np
from flashfilter.localization import LocalizationConfig, block_activity_mask, localize

def test_one_active_block_is_localized_exactly():
    activity=np.zeros((8,8),dtype=np.float32); activity[:4,:4]=.5
    expected=np.zeros_like(activity); expected[:4,:4]=1
    assert np.array_equal(block_activity_mask(activity,.2,4),expected)

def test_localize_preserves_time_and_space_shape():
    changes=np.zeros((3,8,8),dtype=np.float32); changes[1:,4:,4:]=.8
    hard,soft=localize(changes,LocalizationConfig(.2,4,1,1,1),cleanup=False)
    assert hard.shape==soft.shape==changes.shape
    assert hard[1,6,6]==1 and hard[1,1,1]==0 and np.all((soft>=0)&(soft<=1))

def test_partial_edge_blocks_use_only_real_pixels():
    activity=np.zeros((5,7),dtype=np.float32); activity[4,4:]=.6
    mask=block_activity_mask(activity,.5,4)
    assert np.all(mask[4,4:]==1)
    assert np.all(mask[:4]==0)
