# Similarity

This directory contains the similarity benchmark scripts, outputs, and checkpoint state.

## Structure

- `benchmark_similarity.py`
  Runs the benchmark and writes plots plus a checkpoint.

- `overlay_similarity_plots.py`
  Compares two benchmark runs and writes an overlay plot.

- `utils.py`
  Shared comparison and plotting utilities.

- `checkpoints/`
  Stores benchmark checkpoint files.

- `plots/`
  Stores plots written by `benchmark_similarity.py`.

## Checkpoint

`checkpoints/checkpoint.json` stores the current benchmark state:

- `data_index`
- `timing`
- `attribution_results`

It is overwritten on each completed datapoint and can be used to resume a run with:

```bash
python benchmark_similarity.py --start-from-checkpoint
```

## Overlay

`overlay_similarity_plots.py` expects two checkpoint files:

- one from a non-deterministic run
- one from a deterministic run

Example:

```bash
python overlay_similarity_plots.py \
  --non-deterministic-file checkpoints/<checkpoint>.json \
  --deterministic-file checkpoints/<det-checkpoint>.json \
  --output overlay.png
```
