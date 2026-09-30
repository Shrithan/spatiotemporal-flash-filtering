#!/usr/bin/env bash
set -euo pipefail

# Reproduce the synthetic v1.0 research artifact. No external media is needed.
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS_DIR="${1:-experiments/results/canonical}"

"$PYTHON_BIN" -m pytest
"$PYTHON_BIN" -m flashfilter generate-synthetic --output-dir experiments/generated --seed 7
"$PYTHON_BIN" -m flashfilter benchmark --output-dir "$RESULTS_DIR" --seed 7
"$PYTHON_BIN" -m flashfilter benchmark-analyzers --output-dir "$RESULTS_DIR/analyzers"
"$PYTHON_BIN" -m flashfilter ablate --output-dir "$RESULTS_DIR/localized_ablation" --seed 7
"$PYTHON_BIN" -m flashfilter sweep --output-dir "$RESULTS_DIR/sweep" --seed 7
"$PYTHON_BIN" -m flashfilter ablate-adaptive --output-dir "$RESULTS_DIR/adaptive_ablation" --seed 7
if [[ "$RESULTS_DIR" == "experiments/results/canonical" ]]; then
  "$PYTHON_BIN" experiments/generate_v1_figures.py --results-dir "$RESULTS_DIR" --output-dir paper/figures
  "$PYTHON_BIN" paper/build_pdf.py
else
  "$PYTHON_BIN" experiments/generate_v1_figures.py --results-dir "$RESULTS_DIR" --output-dir "$RESULTS_DIR/figures_v1"
fi

echo "Reproduction complete: $RESULTS_DIR"
