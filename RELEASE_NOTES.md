# v1.0.0 Release Notes

## Research question

Can spatially localized temporal filtering reduce rapid visual changes while
preserving more source information than whole-frame filtering?

## Included methods

- no-filter reference;
- global temporal-blending baseline;
- localized block-based temporal-blending baseline;
- event-specific adaptive method with separate luminance, saturated-red, and
  regular-pattern evidence and corrections;
- enhanced and frame-reconstruction variants retained as exploratory studies.

## Evaluation added for v1.0

- twelve deterministic synthetic scenarios with spatial ground truth;
- 49-point suppression--distortion parameter grid and 588 per-case results;
- non-dominated/Pareto analysis using aggregate MAE and mean temporal activity;
- seven-way adaptive component ablation;
- per-case temporal, distortion, localization, modification-area, and runtime
  outputs;
- natural-media case-study figures generated from paired decoded video;
- architecture, ablation, failure-case, and suppression--distortion figures;
- reproducible 40-second MP4 and non-flashing 10-second README GIF;
- revised technical report, verified bibliography, and method notes;
- one-command local reproduction and a manual full-reproduction CI workflow.

## Important measured result

At default settings, the event-specific adaptive method reduces aggregate mean
temporal activity from 0.10866 to 0.02966 and reaches mask IoU 0.8907, but its
MAE is 0.05624 and aggregate peak activity remains 0.16744. None of the
evaluated adaptive configurations lies on the aggregate MAE--mean-activity
frontier. The tuned global and localized baselines therefore provide stronger
tradeoffs on those two objectives in the current benchmark.

## Known limitations

- opposing-transition detection leaves the first large transition untreated;
- raw differences confuse motion and cuts with relevant intensity changes;
- recursive blending can ghost motion;
- the regular-pattern detector covers only horizontal/vertical tile profiles;
- the red ratio is a simple research proxy;
- the controlled benchmark is small, synthetic, SDR, and not clinically
  labeled;
- OpenCV filtering does not preserve audio or source timestamps automatically.

All thresholds and output metrics are experimental computational proxies. This
release is not clinically validated and does not establish medical safety or
formal standards compliance.
