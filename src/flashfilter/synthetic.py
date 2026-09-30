"""Deterministic synthetic videos and spatial ground truth."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SyntheticCase:
    name: str
    frames: np.ndarray
    masks: np.ndarray
    fps: float = 24.0
    category: str = "target"
    description: str = ""

@dataclass(frozen=True)
class AnalyzerCase:
    """Synthetic analyzer case with expected channel-level findings."""
    name: str
    frames: np.ndarray
    fps: float
    expect_luminance: bool
    expect_red: bool
    expect_pattern: bool
    description: str

def _base(length: int, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    frames = np.full((length, height, width, 3), 0.2, dtype=np.float32)
    masks = np.zeros((length, height, width), dtype=bool)
    return frames, masks

def generate_cases(length: int = 24, height: int = 64, width: int = 96, seed: int = 7) -> dict[str, SyntheticCase]:
    """Generate controlled temporal, chromatic, pattern, and confound cases."""
    if min(length, height, width) < 8: raise ValueError("Synthetic dimensions must be at least 8")
    rng = np.random.default_rng(seed)
    cases: dict[str, SyntheticCase] = {}
    def add(name, frames, masks, category, description):
        cases[name] = SyntheticCase(name, np.clip(frames, 0, 1), masks, category=category, description=description)

    f, m = _base(length, height, width); f[1::2] = .9; m[1:] = True
    add("whole_frame_alternation", f, m, "target", "Whole frame alternates between dark and bright.")
    f, m = _base(length, height, width); s=max(4,min(height,width)//8); y=(height-s)//2; x=(width-s)//2; f[1::2,y:y+s,x:x+s]=1; m[1:,y:y+s,x:x+s]=True
    add("small_flashing_square", f, m, "target", "Small intense alternating square; demonstrates global dilution.")
    f, m = _base(length,height,width); y0,y1=height//4,3*height//4; x0,x1=width//4,3*width//4; f[1::2,y0:y1,x0:x1]=.9; m[1:,y0:y1,x0:x1]=True
    add("medium_flashing_region",f,m,"target","Medium rectangular alternating region.")
    f,m=_base(length,height,width); boxes=[(4,height//3,4,width//3),(2*height//3,height-4,2*width//3,width-4)]
    for t in range(1,length):
        for y0,y1,x0,x1 in boxes: f[t,y0:y1,x0:x1]=.95 if (t+(x0>4))%2 else .1; m[t,y0:y1,x0:x1]=True
    add("multiple_regions",f,m,"target","Two independently phased alternating regions.")
    texture=rng.random((height,width,1),dtype=np.float32); f=np.repeat(texture[None],length,axis=0); f=np.repeat(f,3,axis=3); m=np.zeros((length,height,width),bool)
    add("static_texture",f,m,"negative","Static high-contrast texture.")
    f,m=_base(length,height,width); f[:]=np.linspace(.1,.9,length)[:,None,None,None]
    add("gradual_transition",f,m,"negative","Slow global illumination ramp.")
    f,m=_base(length,height,width); f[length//2:]=.85
    add("scene_cut",f,m,"confound","One abrupt whole-frame cut rather than repetition.")
    f,m=_base(length,height,width); wave=.45+.3*np.sin(np.linspace(0,2*np.pi,length)); f[:]=wave[:,None,None,None]
    add("global_brightness_change",f,m,"confound","Camera-like exposure oscillation.")
    f,m=_base(length,height,width); size=max(5,min(height,width)//8)
    for t in range(length):
        x=int((width-size)*t/max(1,length-1)); y=height//2-size//2; f[t,y:y+size,x:x+size]=.95
    add("moving_bright_object",f,m,"confound","Bright object translates without flashing.")
    f,m=_base(length,height,width); y0,y1=height//3,2*height//3; x0,x1=width//3,2*width//3
    for t in range(length): f[t,y0:y1,x0:x1]=.95 if (t//2)%2 else .05
    m[1:,y0:y1,x0:x1]=True
    add("rapid_local_alternation",f,m,"target","Localized region alternates every two frames.")

    # Chromatic target for the adaptive red evidence/correction channel.
    f,m=_base(length,height,width); y0,y1=height//4,3*height//4; x0,x1=width//4,3*width//4
    f[:,y0:y1,x0:x1]=.02
    f[1::2,y0:y1,x0:x1]=np.array([1.,0.,0.],np.float32); m[1:,y0:y1,x0:x1]=True
    add("localized_red_alternation",f,m,"target","Localized saturated-red/near-black alternation.")

    # Static regular-pattern evidence exists from the first frame, so its
    # spatial ground truth also begins at frame zero.
    f=np.zeros((length,height,width,3),np.float32)
    stripe_width=max(2,width//24)
    for x in range(0,width,2*stripe_width): f[:, :, x:x+stripe_width]=1
    m=np.ones((length,height,width),bool)
    add("persistent_regular_pattern",f,m,"target","Static high-contrast regular vertical pattern.")
    return cases


def generate_analyzer_cases(height: int = 64, width: int = 96) -> dict[str, AnalyzerCase]:
    """Generate boundary and confound cases for the multi-analyzer pipeline.

    Expected labels apply to this project's documented proxies, not to medical
    safety or formal conformance with an international standard.
    """
    fps = 12.0
    length = 12
    cases: dict[str, AnalyzerCase] = {}

    def add(name, frames, lum, red, pattern, description):
        cases[name] = AnalyzerCase(name, np.asarray(frames, np.float32), fps, lum, red, pattern, description)

    frames = np.zeros((length, height, width, 3), np.float32)
    frames[1::2] = 1
    add("global_luminance_alternation", frames, True, False, False, "Full-frame black/white alternation.")

    frames = np.full((length, height, width, 3), .05, np.float32)
    frames[1::2, height//4:3*height//4, width//4:3*width//4, 0] = 1
    add("localized_red_alternation", frames, True, True, False, "Localized red/near-black alternation.")

    striped = np.zeros((height, width, 3), np.float32)
    for x in range(0, width, 8): striped[:, x:x+4] = 1
    frames = np.repeat(striped[None], length, axis=0)
    add("persistent_regular_stripes", frames, False, False, True, "Static high-contrast regular vertical stripes.")

    levels = np.linspace(.05, .95, length, dtype=np.float32)
    frames = levels[:, None, None, None] * np.ones((1, height, width, 3), np.float32)
    add("monotonic_ramp", frames, False, False, False, "Large but one-direction brightness ramp.")

    frames = np.full((length, height, width, 3), .05, np.float32)
    frames[length//2:] = .95
    add("single_scene_cut", frames, False, False, False, "One high-magnitude transition without reversal.")

    # Consecutive strong steps go in the same direction before returning once;
    # a transition counter should not invent repeated flashes from the steps.
    values = np.array([.05, .25, .50, .80, .50, .25, .05, .05, .05, .05, .05, .05], np.float32)
    frames = values[:, None, None, None] * np.ones((1, height, width, 3), np.float32)
    add("multistep_excursion", frames, False, False, False, "Multiple same-direction steps and a single return.")

    # Left and right regions alternate at different times. Each region is less
    # than the configured 20% event area, so their areas must not be combined.
    frames = np.full((length, height, width, 3), .05, np.float32)
    box_h, box_w = height//3, width//3
    for index in range(length):
        if index % 4 == 1: frames[index, :box_h, :box_w] = .95
        if index % 4 == 3: frames[index, -box_h:, -box_w:] = .95
    add("out_of_sync_small_regions", frames, False, False, False, "Different small regions change out of sync.")
    return cases
