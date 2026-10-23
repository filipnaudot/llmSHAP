# Attribution benchmark

The separate `attribution_benchmark.py` runner leaves `runtime_benchmark.py` and the existing speed plots alone.
It produces two tables using cosine similarity from sentence-transformers/all-MiniLM-L6-v2 embeddings and floor-rounded stratified sampling.

## Preview the workload

From the repository root, use this project's Python environment:

~~~bash
env/bin/python analysis/benchmark/attribution_benchmark.py plan
~~~

Planning never initializes an API client, requires no API key, and writes no files.
Defaults select two cases for each symptom count 4–10, both paper models,
three independent repetitions, and 20 offline sampling seeds.
The live experiment uses only the 4- and 6-symptom cases.
With the included reduced dataset, the plan is 40,024 generations before retries,
across both models. The default hard limit is 50,000 generation attempts,
including retries, unsuccessful attempts, and work retained from interrupted trials.

## Local smoke run

This uses synthetic, prompt-dependent responses and the real embedding model.
It needs no OpenAI API key. The embedding packages and model weights must be available:

~~~bash
env/bin/python analysis/benchmark/attribution_benchmark.py run \
  --models dummy --feature-counts 4 --cases-per-count 1 \
  --repetitions 2 --sampler-seeds 2 --threads 2
~~~

Synthetic results are explicitly labeled and must not be used as paper results.

## Small API pilot

Run this yourself only when ready to make paid API calls:

~~~bash
env/bin/python analysis/benchmark/attribution_benchmark.py run \
  --models gpt-4.1-mini --feature-counts 4 --cases-per-count 1 \
  --repetitions 1 --sampler-seeds 2 --max-requests 1000
~~~

One repetition tests execution but cannot estimate between-run variability.
Check response status, output length, usage, and latency before a larger collection.
The text model uses temperature 0.2; the reasoning model uses low reasoning effort.
Both use the shared `--max-output-tokens` option, defaulting to 100000.
Requests are automatically capped at each model's supported maximum: 32,768 for
[GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini) and 128,000 for
[GPT-5 nano](https://developers.openai.com/api/docs/models/gpt-5-nano).
The default therefore sends 32,768 for GPT-4.1 mini and 100,000 for GPT-5 nano.
The plan and manifest record the requested limit and the effective model limits.
Override the shared limit when starting a new run, for example `--max-output-tokens 4096`.
Empty or incomplete outputs stop collection instead of being silently scored.
Changing inference settings requires a new run.

To collect the full agreed scope later, the command is:

~~~bash
env/bin/python analysis/benchmark/attribution_benchmark.py run
~~~

API collection requires the existing OpenAI extra and OPENAI_API_KEY (or the repository's .env).
Planning needs no embedding model or API key.
Collection and analysis require the existing embeddings extra:

~~~bash
env/bin/python -m pip install -e '.[openai,embeddings]'
~~~

The first embedding model load can download its weights.
Collection checks embedding availability before making model API calls.
Tests use a mock encoder and do not download packages or model weights.

## Outputs and recovery

Existing run folders are not moved or changed.
Earlier runs retain their original payoff and cannot be resumed with this changed source.
Start a new run for the embedding payoff.

By default, runs live under `analysis/benchmark/checkpoints/attribution-benchmark/<UTC timestamp>/`,
which is already excluded from version control. An explicit `--output PATH` must be a new directory.

- `manifest.json`: selected cases and original row IDs, fixed prompts, dataset/source hashes,
  model settings, seeds, sampling protocol, and software versions.
- Model subdirectories: raw textual responses, response/model/request IDs, response status,
  reported token usage, request-attempt records, and latency.
- `accuracy.md`: approximation agreement on shared sampled games.
- `stochastic.md`: variation, efficiency residual, and supplementary timing from live redraws.
- `results.json`: numerical results, all per-table/per-seed accuracy vectors, live attribution
  vectors and endpoints, coverage, request counts, and reported usage.

Resume or regenerate tables with the saved run directory:

~~~bash
env/bin/python analysis/benchmark/attribution_benchmark.py resume PATH
env/bin/python analysis/benchmark/attribution_benchmark.py analyze PATH
~~~

Resume uses the saved configuration and refuses changed source or dataset content.
Complete shared-table responses are reused; only missing coalitions are queried.
A failed live trial is restarted with a fresh cache and a new journal because combining
fragments would invalidate its timing and redraw protocol. Earlier attempts remain recorded
and count toward the budget. Completed live trials are not repeated.
Concurrent collection/analysis of the same directory is prevented with a POSIX file lock.

A terminated request may have reached the provider before its response was saved;
exactly-once billing cannot be guaranteed after a crash. Connection failures may also lack usage.
Reported token totals therefore cover returned response usage, not a billing guarantee.
Only a partial final journal line is discarded after an interrupted write; other corruption fails.

An exhausted budget can be increased explicitly without changing inference settings:

~~~bash
env/bin/python analysis/benchmark/attribution_benchmark.py resume PATH --max-requests 60000
~~~

Analysis never initializes an API client. Partial collections are clearly labeled.
Keep the source version used for collection available: resume and analysis validate its hashes.

## Interpretation

For each case/model, one independently generated full-context reference is held fixed across
all methods, tables, and live repetitions. All coalition outputs, including the full coalition,
are separate draws scored against that reference; the full payoff is not forced to 1.

The accuracy experiment collects three independent tables with one response per coalition.
All methods see identical payoffs within a table. Exhaustive Shapley is exact for that sampled game,
not the unknown expected-payoff game. Stratified methods use 20 paired seeds per table;
seed and table results are averaged within each case before cases receive equal weight.
Table cells show mean ± sample SD of case means, not confidence intervals.
Cosine with a zero vector is undefined, excluded, and counted explicitly. MAE remains defined.

The live experiment uses the library's attribution engine with one generation per evaluation.
Each repetition creates a new cache; the stratified seed is held fixed to isolate redraw effects.
Method order is deterministically shuffled. Concurrency and seeds are recorded.
Variation is the root mean of per-feature sample variances across independent repetitions
of the same case. A single repetition has undefined variation.
Efficiency error is the absolute difference between summed attributions and the recorded
full-minus-empty payoff, using the same cached endpoints when caching is enabled.
Elapsed times include durable journal writes; retain the original speed figures for the main
runtime results rather than mixing these supplementary timings into those figures.

Caching can preserve efficiency for a realized sampled game while its attributions still vary
between repetitions. Neither experiment establishes symmetry or null-player premises for
real symptoms, medical validity, or accurate estimation of the expected-payoff Shapley value.
