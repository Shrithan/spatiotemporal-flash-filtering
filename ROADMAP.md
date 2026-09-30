# Research Roadmap

This roadmap separates the completed v1.0 research artifact from possible
follow-up studies. The repository reports computational visual-change proxies;
it does not claim clinical validation, seizure prevention, medical safety, or
formal standards conformance.

## Completed in v1.0

- [x] Installable Python package, documented CLI, public API, and streaming I/O
- [x] Explicit sRGB linearization and relative-linear-luminance formulation
- [x] No-filter, global, and localized block baselines
- [x] Event-specific adaptive method with luminance, red, and pattern channels
- [x] Exploratory enhanced and frame-reconstruction failure investigations
- [x] Twelve-case deterministic synthetic benchmark with spatial ground truth
- [x] Per-case temporal, distortion, modification, localization, and runtime metrics
- [x] Compact suppression--distortion parameter study and Pareto analysis
- [x] Localized-method and adaptive-component ablations
- [x] Natural-media case studies with source metadata and hashes, not source videos
- [x] Generated publication figures and non-flashing README/full demo assets
- [x] Research-style report and verified bibliography
- [x] Unit/integration tests on Python 3.11--3.13
- [x] Manual synthetic-reproduction CI workflow and one-command local script

## Potential future research

These directions are independent research opportunities, not prerequisites for
using or explaining the current artifact.

- [ ] Detect scene cuts and reset temporal state before applying a correction
- [ ] Compare raw and motion-compensated temporal differences
- [ ] Estimate local temporal frequency, duration, and periodicity
- [ ] Investigate prospective handling of the first large transition
- [ ] Replace the simple red ratio with a documented perceptual chromaticity model
- [ ] Generalize pattern analysis beyond horizontal/vertical tile profiles
- [ ] Select parameters on held-out data rather than the evaluation set
- [ ] Evaluate broader, independently annotated natural-media datasets
- [ ] Add controlled 720p/1080p throughput and latency measurements
- [ ] Preserve audio and source timestamps in a supported end-to-end media pipeline
- [ ] Evaluate HDR transfer functions and metadata explicitly

## Separate application repository

A future product-facing repository should consume a pinned release of this
package rather than duplicate its algorithms. Interface and deployment work
should preserve the research disclaimer, avoid autoplaying rapid stimuli, and
keep computational proxy output distinct from medical or compliance claims.
