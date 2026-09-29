# Research review and enhanced-method design

This review separates three related but different goals: detecting
standards-defined flash patterns, reducing a computational temporal-activity
proxy, and perceptual video deflickering. The project currently targets the
second goal. It does not claim standards conformance or clinical validation.

## Relevant work

| Work | Relevant idea | How it informs this project |
|---|---|---|
| Nomura et al. (2000), *A New Adaptive Temporal Filter* | Temporal filtering strength adapts to a computed flicker index. | Motivates activity-dependent filtering, but its reported patient study does not validate our different implementation. |
| Carreira et al. (2015), *Automatic Detection of Flashing Video Content* | Real-time spatiotemporal analysis based on ITU-R guidance. | Supports explicit spatial area and temporal-transition analysis. |
| Alzubaidi et al. (2016), *Parallel Scheme for Real-Time Detection* | Streaming spatiotemporal detection and synthetic pattern generation. | Supports constant-memory processing and controlled test media. |
| Jordan (2025), *Evaluating Conformance of Video Safety Tools* | A flash is a pair of opposing transitions involving overlapping pixels; validation needs confounds, irregular timing, and multi-step changes. | Directly motivates polarity-reversal masks and expanded synthetic cases. |
| Yu and Srinath (2001), *An Efficient Method for Scene Cut Detection* | Successive-frame color-histogram distance is useful for cut detection. | Motivates resetting/suppressing temporal filtering on isolated cuts. |
| Liu et al. (2018), *Video Deflickering Using Multi-Frame Optimization* | Warped multiple-frame reconstruction balances fidelity and temporal/spatial coherence. | Motivates future optical-flow alignment; not yet included because complexity and warp artifacts need a separate evaluation. |
| Lei et al. (2023), *Blind Video Deflickering by Neural Filtering* | Learned filtering uses temporally consistent atlas information. | Demonstrates a strong research direction, but would add training data, large dependencies, and black-box behavior inconsistent with the current educational baseline. |

## Enhanced candidate

The implemented `enhanced` method combines only mechanisms with distinct jobs:

1. Compute signed linear-luminance transitions.
2. Require a strong transition followed by a strong transition of opposite
   polarity at the same pixels.
3. Pool reversal activity at 4- and 8-pixel block scales.
4. Use luminance-histogram distance to recognize isolated scene-cut candidates.
5. Feather the mask spatially.
6. Clamp selected output luminance to a maximum step of 0.08 while
   approximately preserving current-frame chromaticity.

Unlike RGB blending, luminance clamping retains the current frame's spatial
content and therefore avoids much of the visible double-image ghosting. The
algorithm remains causal and uses O(HW) state. Its causal limitation is
fundamental: the first transition cannot be known to be half of a flash until
an opposing transition arrives. A future one-frame-latency mode could revise
the buffered prior frame once this evidence appears.

## Measured candidate result

On the ten-scenario synthetic benchmark, compared with the original localized
method, the enhanced candidate reduced mean activity from 0.0356 to 0.0325,
reduced modified area from 15.62% to 7.19%, improved mean IoU from 0.6589 to
0.7773, and eliminated modifications in the scene-cut and moving-object
confounds. MAE improved from 0.0449 to 0.0436. Peak activity worsened from
0.0836 to 0.1948 because the causal detector leaves the first transition
untouched. These results justify retaining both methods rather than declaring
the candidate universally superior.

## Next experiments before a final default

- Add one-frame lookahead and retroactive symmetric filtering to address the
  first-transition peak.
- Evaluate optical-flow-aligned temporal references against the existing
  moving-object confound and real camera motion.
- Add irregular, multi-step, out-of-sync, saturated-color, and compression
  cases inspired by the conformance-test literature, without claiming that
  the resulting suite establishes compliance.
- Add no-reference perceptual temporal metrics and a blinded preference test;
  PSNR and SSIM alone cannot measure ghosting quality.
- Calibrate on a validation subset and report on a separate test subset.

## Frame-generation experiment

The implemented `framegen` method buffers A, B, and C, detects an opposing
excursion at B, estimates bidirectional Farneback flow between A and C at a
lower analysis resolution, scales the flow back to source resolution, and
reconstructs only the masked region of B. It preserves frame count, FPS, and
dimensions.

This works for isolated A-B-A excursions and rejects monotonic changes, scene
cuts, and the moving-object confound. It does **not** solve sustained A-B-A-B
alternation: overlapping triplets alternately reconstruct toward A and B,
merely changing phase. On the canonical benchmark its mean activity is 0.1131
and MAE is 0.0919, both worse than the enhanced and localized methods. On the
15-second full-frame alternation clip its mean activity remains 0.8927 versus
0.9059 for the source. The result is retained as a documented negative result.

The next defensible design is segment-level generation: detect a complete
alternating run, select a stable reference trajectory once for that run, then
motion-warp that reference through every frame. Independent triplet generation
should not be used as the final product for continuous flashing.

## References

Full bibliographic entries and DOIs are maintained in `paper/references.bib`.
