# Research Roadmap

This roadmap separates research milestones from any future user-facing product.
The repository reports computational visual-change proxies and does not claim
clinical validation, seizure prevention, medical safety, or formal standards
conformance.

## v0.1 — Reproducible research foundation

- [x] Installable `src` package and CLI
- [x] Deterministic synthetic benchmark and ground-truth masks
- [x] Global, localized, enhanced, frame-generation, and adaptive candidates
- [x] Luminance, saturated-red proxy, and regular-pattern evidence channels
- [x] Structured public `analyze_video` and `filter_video` APIs
- [x] Canonical CSV results, plots, tests, and technical report
- [x] GitHub Actions matrix for Python 3.11–3.13
- [ ] Re-run canonical results from a clean checkout before tagging v0.1.0

## v0.2 — Detection robustness

- [ ] Add a weaker provisional response to the first unusually large transition
- [ ] Reject isolated scene cuts and reset temporal state
- [ ] Report detection delay and first-transition residual activity
- [ ] Add raw versus motion-compensated temporal differences
- [ ] Test camera motion, moving flashes, pans, and cuts followed by flashes

## v0.3 — Temporal and chromatic analysis

- [ ] Estimate local transition frequency, duration, and periodicity
- [ ] Replace the simple red-ratio proxy with a documented perceptual color model
- [ ] Compare simple and perceptual chromatic analyzers in ablations
- [ ] Keep standards-inspired analysis distinct from conformance claims

## v0.4 — Optimization and real-media evaluation

- [ ] Add deterministic suppression–distortion parameter search
- [ ] Generate Pareto-frontier CSV and figures
- [ ] Expand the media manifest with ordinary-motion controls and diverse edits
- [ ] Add manual computational-event annotations where licensing permits
- [ ] Add per-channel diagnostic figures and event timelines

## v1.0 — Research release

- [ ] Preserve resolution, frame rate, timestamps, and audio in one supported API
- [ ] Benchmark 720p and 1080p streaming throughput
- [ ] Freeze the public API and configuration schema
- [ ] Regenerate every README and paper value from canonical result files
- [ ] Publish a tagged release before a separate application consumes the package

## Future application repository

A future `flashfilter-app` repository should depend on a pinned release of this
package. It should contain interface and deployment code, avoid autoplaying
source media, expose the research disclaimer prominently, and never duplicate
or silently modify the algorithm implementation.
