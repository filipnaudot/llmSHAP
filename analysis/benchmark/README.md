# Attribution and runtime benchmarks

## Attribution accuracy and variation

The attribution_benchmark.py runner compares methods on shared coalition payoffs
and measures variation across fresh runs, using all-MiniLM-L6-v2 embeddings.
See [ATTRIBUTION_BENCHMARK.md](ATTRIBUTION_BENCHMARK.md) for setup, outputs, and recovery.

From the repository root, preview its workload without API calls or model downloads:

~~~bash
env/bin/python analysis/benchmark/attribution_benchmark.py plan
~~~

runtime_benchmark.py retains the earlier runtime experiment described below.
Generated checkpoints, plots, and results are excluded from Git.

This directory contains the similarity benchmark scripts, outputs, and checkpoint state.

## Run Benchmark
From the `analysis/benchmark` directory:
```bash
python ./runtime_benchmark.py --verbose
```

Alternatively, from the repository root:
```bash
python analysis/benchmark/runtime_benchmark.py --verbose
```


## Structure

- `runtime_benchmark.py`
  Runs the benchmark and writes plots, a results summary, and checkpoints.

- `overlay_similarity_plots.py`
  Compares two benchmark runs and writes an overlay plot.

- `utils.py`
  Shared comparison and plotting utilities.

- `checkpoints/`
  Stores benchmark checkpoint files.

- `plots/`
  Stores plots written by `runtime_benchmark.py`.

- `results/`
  Stores `RESULTS.md` written by `runtime_benchmark.py`.



## Checkpoint

`checkpoints/checkpoint.json` stores the current benchmark state:

- `data_index`
- `results`

It is overwritten on each completed datapoint and can be used to resume a run with:

```bash
python runtime_benchmark.py --start-from-checkpoint
```

You can see all available CLI options with:

```bash
python runtime_benchmark.py --help
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
