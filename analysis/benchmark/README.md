# Similarity

This directory contains the similarity benchmark scripts, outputs, and checkpoint state.

## Run Benchmark
From the `analysis/benchmark` directory:
```bash
python ./benchmark.py --verbose
```

Alternatively, from the repository root:
```bash
python analysis/benchmark/benchmark.py --verbose
```


## Structure

- `benchmark.py`
  Runs the benchmark and writes plots, a results summary, and checkpoints.

- `overlay_similarity_plots.py`
  Compares two benchmark runs and writes an overlay plot.

- `utils.py`
  Shared comparison and plotting utilities.

- `checkpoints/`
  Stores benchmark checkpoint files.

- `plots/`
  Stores plots written by `benchmark.py`.

- `results/`
  Stores `RESULTS.md` written by `benchmark.py`.



## Checkpoint

`checkpoints/checkpoint.json` stores the current benchmark state:

- `data_index`
- `results`

It is overwritten on each completed datapoint and can be used to resume a run with:

```bash
python benchmark.py --start-from-checkpoint
```

You can see all available CLI options with:

```bash
python benchmark.py --help
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
