# Examples

Generate a lossless synthetic input and filter it:

```bash
flashfilter generate-synthetic --output-dir experiments/generated
flashfilter analyze experiments/generated/small_flashing_square.npz
flashfilter filter experiments/generated/small_flashing_square.npz \
  --method localized \
  --threshold 0.12 \
  --block-size 8 \
  --temporal-window 2 \
  --output outputs/small_flashing_square_filtered.mp4
```

`analyze` reports normalized computational activity metrics. These values are
not medical-risk estimates and do not establish accessibility compliance.
