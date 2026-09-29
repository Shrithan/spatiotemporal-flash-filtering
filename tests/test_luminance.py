import numpy as np
import pytest
from flashfilter.luminance import gamma_luma_proxy, linear_luminance, srgb_to_linear

def test_srgb_endpoints_are_preserved():
    rgb = np.array([[[0., 0., 0.], [1., 1., 1.]]], dtype=np.float32)
    assert np.allclose(srgb_to_linear(rgb), rgb)
    assert np.allclose(linear_luminance(rgb), [[0., 1.]])

def test_primary_color_weight():
    assert gamma_luma_proxy(np.array([[[1., 0., 0.]]], dtype=np.float32))[0, 0] == pytest.approx(0.2126)

def test_invalid_range_rejected():
    with pytest.raises(ValueError, match="normalized"): linear_luminance(np.full((2, 2, 3), 1.1))
