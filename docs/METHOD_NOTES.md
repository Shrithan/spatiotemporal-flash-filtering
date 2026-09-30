# Method Notes

This document is a technical explanation of the implemented system. The
project measures computational visual-change proxies; it does not estimate an
individual's seizure risk, certify standards compliance, or establish medical
safety.

## 1. Color and temporal representation

**Problem.** Frame differences should be computed on a scalar that reflects
display-light changes more faithfully than an average of gamma-encoded RGB.

**Input and output.** `linear_luminance` maps normalized decoded RGB arrays of
shape `(..., H, W, 3)` to relative linear luminance `Y` in `[0, 1]`.

**Operation.** The code assumes decoded values use the sRGB encoding curve. It
inverts that transfer function channel by channel, then computes

```text
Y = 0.2126 R_linear + 0.7152 G_linear + 0.0722 B_linear.
```

sRGB and BT.709 share chromaticity primaries and a D65 white point, which is
why their linear-light luminance coefficients match. BT.709 does not specify
the sRGB inverse transfer function. The optional `gamma_proxy` mode applies the
same weights directly to encoded values and is therefore described as a luma
or brightness proxy, not physical luminance.

Consecutive-frame activity is

```text
Delta_t(x, y) = abs(Y_t(x, y) - Y_(t-1)(x, y))
D_t = mean_(x, y) Delta_t(x, y).
```

Both operations are `O(HW)` per frame. They do not model display peak
luminance, HDR metadata, viewing geometry, temporal frequency, or physiology.

## 2. Baselines

### No filtering

Copies the source and provides the zero-distortion reference. It demonstrates
why similarity metrics cannot be interpreted alone: doing nothing gives
perfect MAE/SSIM but leaves all measured temporal activity.

### Global temporal filtering

**Problem.** Reduce a large whole-frame transition with one transparent rule.

If `D_t >= threshold`, every current pixel is blended toward the previous
filtered frame:

```text
I_hat_t = (1 - alpha) I_t + alpha I_hat_(t-1).
```

The threshold controls detection sensitivity; `alpha` controls suppression and
distortion. Runtime is `O(HW)` and state is one frame. Global averaging can
dilute a small intense region, while whole-frame blending modifies unrelated
content and recursive use of the previous output can produce ghosting.

### Localized block filtering

**Problem.** Retain the simplicity of temporal blending while avoiding changes
outside an active region.

The detector averages recent `Delta` maps, partitions the map into fixed
blocks, and selects a block when its mean exceeds a threshold. Morphological
closing can fill small gaps and a Gaussian blur feathers the hard boundary.
With the resulting soft mask `M_t`,

```text
I_hat_t = (1 - alpha M_t) I_t + alpha M_t I_hat_(t-1).
```

The main hyperparameters are temporal threshold, block size, temporal window,
blend, cleanup kernel, and feather sigma. The vectorized block reduction,
morphology, blur, and compositing are all `O(HW)` per frame. Smaller blocks can
localize small events more precisely but are more sensitive to texture and
noise. The detector confuses motion and cuts with changes of interest because
it has no motion or scene model.

Alternatives include connected components, adaptive quadtrees, optical-flow
alignment, and local temporal-frequency features. Fixed blocks were chosen
because their decision rule and cost are easy to inspect and test.

## 3. Event-Specific Adaptive Filtering

This is the proposed experimental method. It deliberately preserves three
evidence channels rather than reducing them to a single unexplained score.

### Luminance evidence and step limiting

Two consecutive signed luminance changes must exceed the configured magnitude
and have opposite signs at the same pixel. This detects an A-B-A-like reversal,
not an isolated first transition. At selected pixels, the current relative
luminance is limited to a maximum step `s` around the previous output
luminance. A neutral linear-RGB offset realizes that target, followed by soft
mask compositing.

This avoids the long recursive trail of ordinary blending, but the first
transition necessarily remains because the evidence requires a reversal.
Threshold and `max_luminance_step` are the dominant controls.

### Saturated-red evidence and local desaturation

The analyzer forms the documented proxy

```text
S = R - max(G, B), when R / (R + G + B) >= red_ratio; otherwise 0.
```

Opposing changes in `S` identify local red-transition evidence. Selected
pixels are blended toward equal RGB channels having the same relative linear
luminance. Desaturation is used only for this chromatic evidence; grayscale
does not solve black-white flashing. The ratio is a simple research proxy, not
a normative chromaticity definition.

### Pattern evidence and local contrast reduction

Each tile is reduced to horizontal and vertical mean profiles. A tile is
selected when it contains enough strong, alternating, approximately regularly
spaced edges. Local contrast is then contracted around a Gaussian local mean:

```text
I_pattern = local_mean + (1 - rho) (I - local_mean).
```

This `O(HW)` detector is interpretable but recognizes only horizontal and
vertical stripe-like structure. It can miss diagonal, curved, irregular, or
scale-mismatched patterns. Because pattern evidence is spatial rather than a
temporal flash signal, applying correction from frame zero is essential; a
later start would manufacture an artificial temporal transition.

### Composition and complexity

Each mask is Gaussian-feathered and its corresponding correction is composed
locally. The returned method mask is the pixelwise maximum of the enabled
masks. All passes are linear in the number of pixels, so processing a `T`-frame
video is `O(THW)`. Streaming execution stores constant temporal state plus a
small number of frame-sized buffers (`O(HW)` memory).

The adaptive method's important controls are evidence thresholds, minimum area,
maximum luminance step, desaturation strength, pattern contrast reduction, and
feather sigma. The canonical sweep varies a compact subset and reports every
setting instead of selecting only a favorable result.

## 4. Evaluation decisions

The controlled benchmark contains twelve deterministic scenarios at 24 frames
and 96 x 64 pixels using seed 7. It includes positive localized and full-frame
events plus scene cuts, motion, gradual illumination, global exposure changes,
static texture, a red boundary case, and a persistent regular pattern. Ground
truth denotes the synthetic event regions where meaningful; it is not a
medical label.

Metrics are read jointly:

- mean and peak temporal activity describe the defined signal proxy;
- MAE, MSE, PSNR, and SSIM describe source distortion;
- modified-area ratio describes spatial selectivity;
- precision, recall, F1, and IoU compare predicted and synthetic masks;
- elapsed seconds and FPS describe implementation runtime on the executing
  machine.

The parameter study minimizes both mean activity and MAE. A point is
non-dominated when no evaluated point is at least as good on both objectives
and strictly better on one. This is a comparison within the evaluated grid,
not proof of a globally optimal filter.

## 5. What the current evidence says

At default settings, the adaptive method improves aggregate mean activity and
localization IoU relative to the global and localized baselines, but it has
higher distortion and a much larger aggregate peak because of first-transition
latency. In the parameter sweep, adaptive configurations do not appear on the
MAE-versus-mean-activity frontier; tuned global and localized configurations
dominate them on those two aggregate objectives. The component ablation shows
that luminance correction supplies nearly all measured temporal suppression;
red correction adds a small benefit on the designed red case, while pattern
correction changes static appearance without reducing temporal activity.

These are valuable negative and mixed results. They suggest that future work
should prioritize first-transition handling, scene-cut rejection, motion
compensation, and better rules for deciding whether a non-temporal pattern
should be modified, rather than merely stacking stronger corrections.
