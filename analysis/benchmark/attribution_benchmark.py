"""Attribution accuracy and variation benchmark.

Run python analysis/benchmark/attribution_benchmark.py --help for commands.
Only run and resume can contact the model API.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from math import fsum, sqrt
import os
from pathlib import Path
import platform
import random
from statistics import mean, stdev, variance
import sys
import tempfile
import threading
import time

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPOSITORY / "src"), str(REPOSITORY / "analysis")]

from data import SymptomDataset
from llmSHAP import BasicPromptCodec, DataHandler, Generation, ShapleyAttribution, EmbeddingCosineSimilarity
from llmSHAP.attribution_methods import CounterfactualSampler, FullEnumerationSampler, SlidingWindowSampler, StratifiedSampler
from llmSHAP.llm import LLMInterface

SCHEMA_VERSION = 1
MODEL_NAMES = ("gpt-4.1-mini", "gpt-5-nano", "dummy")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_OUTPUT_LIMITS = {"gpt-4.1-mini": 32768, "gpt-5-nano": 128000}
METHODS = ("Counterfactual", "Sliding window (w=3)",
           "Stratified (ratio=0.2)", "Stratified (ratio=0.6)", "Exact Shapley")
DATASET = REPOSITORY / "analysis/data/symptom_to_diagnosis/reduced_symptom_dataset.csv"
OUTPUTS = Path(__file__).resolve().parent / "checkpoints/attribution-benchmark"
SOURCE_FILES = ("src/llmSHAP/attribution_methods/coalition_sampler.py",
                "src/llmSHAP/attribution_methods/shapley_attribution.py",
                "src/llmSHAP/attribution_methods/attribution_function.py",
                "src/llmSHAP/data_handler.py", "src/llmSHAP/prompt_codec.py",
                "src/llmSHAP/value_functions.py",
                "analysis/data/symptom_to_diagnosis/symptom.py",
                "analysis/benchmark/attribution_benchmark.py")


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _source_hashes():
    return {name: _digest((REPOSITORY / name).read_bytes()) for name in SOURCE_FILES}


def _write_json(path, value):
    """Replace a JSON artifact atomically, without partial checkpoints."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _records(path):
    """Ignore only an interrupted, unterminated final journal record."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_bytes().splitlines(keepends=True)
            if line.endswith(b"\n")]


class _Journal:
    def __init__(self, path, root):
        self.path = path
        self.key = path.relative_to(root).as_posix()
        self.lock = threading.Lock()
        self.records = _records(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raw = path.read_bytes()
            complete_size = raw.rfind(b"\n") + 1
            if complete_size < len(raw):
                with path.open("r+b") as stream:
                    stream.truncate(complete_size)

    def append(self, record):
        with self.lock:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self.records.append(record)


class _Budget:
    def __init__(self, limit, used=0):
        self.limit = limit
        self.used = used
        self.lock = threading.Lock()

    def reserve(self):
        with self.lock:
            if self.used >= self.limit:
                raise RuntimeError("Generation-attempt budget exhausted; saved work can be analyzed offline.")
            self.used += 1


class _Backend:
    """Keep API response metadata that the library's text interface discards."""
    def __init__(self, settings, timeout, seed):
        self.settings = settings
        self.seed = seed
        self.client = None
        if settings["name"] != "dummy":
            from dotenv import load_dotenv
            from openai import OpenAI
            load_dotenv(REPOSITORY / ".env")
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY is not set; configure it before API collection.")
            self.client = OpenAI(max_retries=0, timeout=timeout)

    def generate(self, messages, draw_id):
        if self.client is None:
            rng = random.Random(f"{self.seed}:{draw_id}")
            symptoms = messages[-1]["content"].split("SYMPTOM:")[1:]
            findings = " ".join(part.split("\n")[0].strip() for part in symptoms)
            output = f"Synthetic {rng.choice(('alpha', 'beta', 'gamma'))} assessment. "
            output += findings or "No findings available."
            return {"output": output, "status": "completed", "usage": None,
                    "resolved_model": "dummy", "response_id": None, "request_id": None}
        parameters = {"model": self.settings["name"], "input": messages,
                      "max_output_tokens": self.settings["max_output_tokens"],
                      "store": False, "metadata": {"benchmark_draw": draw_id}}
        if self.settings["reasoning"] is not None:
            parameters["reasoning"] = {"effort": self.settings["reasoning"]}
        else:
            parameters["temperature"] = self.settings["temperature"]
        response = self.client.responses.create(**parameters)
        return {"output": response.output_text, "status": response.status,
                "usage": response.usage.model_dump() if response.usage is not None else None,
                "resolved_model": response.model, "response_id": response.id,
                "request_id": getattr(response, "_request_id", None),
                "incomplete_details": (response.incomplete_details.model_dump()
                                       if response.incomplete_details is not None else None)}

    def close(self):
        if self.client is not None:
            self.client.close()


def _retryable(error):
    status = getattr(error, "status_code", 0) or 0
    if getattr(error, "code", None) == "insufficient_quota":
        return False
    return (type(error).__name__ in {"APIConnectionError", "APITimeoutError"}
            or status in {408, 409, 429} or status >= 500)


class _RecordedModel(LLMInterface):
    def __init__(self, backend, journal, budget, retries):
        self.backend = backend
        self.journal = journal
        self.budget = budget
        self.retries = retries
        self.stopped = threading.Event()
        self.lock = threading.Lock()
        self.draws = Counter(record["mask"] for record in journal.records
                             if record["event"] == "start")

    def generate(self, prompt, tools=None, images=None):
        mask = prompt["mask"]
        with self.lock:
            self.draws[mask] += 1
            draw_id = _digest(f"{self.journal.key}:{mask}:{self.draws[mask]}".encode())[:32]
        for attempt in range(self.retries + 1):
            if self.stopped.is_set():
                raise RuntimeError("Trial stopped after an earlier generation failure.")
            self.budget.reserve()
            self.journal.append({"event": "start", "mask": mask, "draw_id": draw_id,
                                 "attempt": attempt, "started_at": _timestamp()})
            started = time.perf_counter()
            try:
                response = self.backend.generate(prompt["messages"], draw_id)
            except Exception as error:
                self.journal.append({"event": "error", "mask": mask, "draw_id": draw_id,
                                     "attempt": attempt, "error_type": type(error).__name__,
                                     "status_code": getattr(error, "status_code", None),
                                     "elapsed_seconds": time.perf_counter() - started})
                if attempt < self.retries and _retryable(error):
                    time.sleep(random.uniform(0.5, 1.0) * min(2 ** attempt, 10))
                    continue
                self.stopped.set()
                raise RuntimeError(f"Generation failed ({type(error).__name__}); "
                                   f"inspect {self.journal.path}.") from error
            accepted = response["status"] == "completed" and bool(response["output"].strip())
            self.journal.append({"event": "response", "mask": mask, "draw_id": draw_id,
                                 "attempt": attempt, "accepted": accepted, **response,
                                 "elapsed_seconds": time.perf_counter() - started})
            if not accepted:
                self.stopped.set()
                raise RuntimeError(f"Empty or incomplete model response ({response['status']}); "
                                   "inspect the journal and output-token limit before continuing.")
            return response["output"]


def _handler(case):
    prompt = {"initial_query": "A patient is showing the following symptom(s):"}
    for index, symptom in enumerate(case["symptoms"], start=1):
        prompt[f"symptom_{index}"] = f"\nSYMPTOM: {symptom}"
    prompt["end_query"] = ("\nBased on the symptom(s), what disease or condition do you think the patient most likely have?"
                           "\nANSWER BRIEFLY.")
    return DataHandler(prompt, permanent_keys={"initial_query", "end_query"})


class _CoalitionCodec(BasicPromptCodec):
    def build_prompt(self, data_handler, indexes):
        selected = set(indexes)
        players = data_handler.get_keys(exclude_permanent_keys=True)
        mask = sum(1 << position for position, key in enumerate(players) if key in selected)
        return {"mask": mask, "messages": super().build_prompt(data_handler, selected)}


def _sampler(method, players, seed):
    if method == METHODS[0]:
        return CounterfactualSampler()
    if method == METHODS[1]:
        return SlidingWindowSampler(players, w_size=3)
    if method == METHODS[2]:
        return StratifiedSampler(0.2, seed=seed)
    if method == METHODS[3]:
        return StratifiedSampler(0.6, seed=seed)
    if method == METHODS[4]:
        return FullEnumerationSampler(len(players))
    raise ValueError(f"Unknown method: {method}")


def _mask(coalition):
    return sum(1 << key for key in coalition)


def _attributions(payoffs, method, seed):
    """Reuse library samplers; table lookup needs no generation threads or caches."""
    players = list(range((len(payoffs) - 1).bit_length()))
    sampler = _sampler(method, players, seed)
    return [fsum(weight * (payoffs[_mask(coalition | {feature})] - payoffs[_mask(coalition)])
                 for coalition, weight in sampler(feature, players)) for feature in players]


def _cosine(left, right):
    norm_left = sqrt(fsum(value * value for value in left))
    norm_right = sqrt(fsum(value * value for value in right))
    if norm_left == 0 or norm_right == 0:
        return None
    return max(-1.0, min(1.0, fsum(a * b for a, b in zip(left, right)) / (norm_left * norm_right)))


def _request_count(size, method, cached, seed):
    players = list(range(size))
    evaluations = 2  # Separately generated full and empty endpoints.
    masks = {0, (1 << size) - 1}
    sampler = _sampler(method, players, seed)
    for feature in players:
        for coalition, _ in sampler(feature, players):
            evaluations += 2
            if cached:
                masks.update((_mask(coalition), _mask(coalition | {feature})))
    return len(masks) if cached else evaluations


def _manifest(args):
    dataset = args.dataset.resolve()
    data = SymptomDataset.load(str(dataset))
    cases = []
    rng = random.Random(args.seed)
    for count in sorted(set(args.feature_counts)):
        candidates = [index for index, entry in enumerate(data) if len(entry.concepts()) == count]
        if len(candidates) < args.cases_per_count:
            raise ValueError(f"Only {len(candidates)} cases have {count} features; "
                             f"requested {args.cases_per_count}.")
        for index in sorted(rng.sample(candidates, args.cases_per_count)):
            cases.append({"id": f"case_{index:04d}", "dataset_index": index,
                          "disease": data[index].disease_name, "symptoms": data[index].concepts(),
                          "system_prompt": data[index].system_prompt()})
    models = [{"name": name, "temperature": 0.2 if name != "gpt-5-nano" else None,
               "reasoning": "low" if name == "gpt-5-nano" else None,
               "max_output_tokens": min(args.max_output_tokens, MODEL_OUTPUT_LIMITS.get(name, args.max_output_tokens))}
              for name in dict.fromkeys(args.models)]
    versions = {}
    for name in ("llmSHAP", "openai", "python-dotenv", "tqdm", "sentence-transformers", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"schema_version": SCHEMA_VERSION, "created_at": _timestamp(),
            "dataset": {"path": str(dataset), "sha256": _digest(dataset.read_bytes())},
            "source_sha256": _source_hashes(),
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "packages": versions},
            "config": {"models": models, "cases": cases, "seed": args.seed,
                       "repetitions": args.repetitions, "sampler_seeds": args.sampler_seeds,
                       "requested_max_output_tokens": args.max_output_tokens,
                       "live_feature_counts": sorted(set(args.live_feature_counts)),
                       "threads": args.threads, "max_requests": args.max_requests,
                       "timeout": args.timeout, "retries": args.retries,
                       "payoff": "embedding cosine similarity to one fixed reference per case/model",
                       "embedding_model": EMBEDDING_MODEL,
                       "coalition_evaluations": 1, "stratified_rounding": "max(1, floor(ratio * size))"}}


def _plan(manifest):
    config = manifest["config"]
    cases = config["cases"]
    shared = config["repetitions"] * sum(2 ** len(case["symptoms"]) for case in cases)
    live = config["repetitions"] * sum(
        _request_count(len(case["symptoms"]), method, cached, config["seed"])
        for case in cases if len(case["symptoms"]) in config["live_feature_counts"]
        for method in METHODS for cached in (False, True))
    per_model = {"references": len(cases), "shared_tables": shared, "live_trials": live}
    return {"cases": len(cases), "cases_by_feature_count": dict(Counter(len(c["symptoms"]) for c in cases)),
            "models": [model["name"] for model in config["models"]],
            "repetitions": config["repetitions"], "generations_per_evaluation": 1,
            "requested_max_output_tokens": config["requested_max_output_tokens"],
            "effective_max_output_tokens": {model["name"]: model["max_output_tokens"] for model in config["models"]},
            "calls_per_model_excluding_retries": per_model,
            "total_calls_excluding_retries": len(config["models"]) * sum(per_model.values()),
            "attempt_budget_including_retries": config["max_requests"]}


def _load_manifest(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported benchmark artifact version.")
    if manifest["source_sha256"] != _source_hashes():
        raise ValueError("Benchmark or core source changed; use the original source to resume or analyze this run.")
    return manifest


@contextmanager
def _run_lock(root):
    import fcntl
    with (root / "run.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another process is collecting or analyzing this run.") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _accepted(records):
    return {record["mask"]: record["output"] for record in records
            if record["event"] == "response" and record["accepted"]}


def _collect(root, manifest):
    similarity = EmbeddingCosineSimilarity(model_name=EMBEDDING_MODEL)
    config = manifest["config"]
    attempts = sum(record["event"] == "start" for path in root.rglob("*.jsonl")
                   for record in _records(path))
    budget = _Budget(config["max_requests"], attempts)
    for settings in config["models"]:
        backend = _Backend(settings, config["timeout"], config["seed"])
        model_root = root / settings["name"]
        try:
            for case in config["cases"]:
                handler = _handler(case)
                codec = _CoalitionCodec(system=case["system_prompt"])
                players = handler.get_keys(exclude_permanent_keys=True)
                journal = _Journal(model_root / "references" / f"{case['id']}.jsonl", root)
                if not _accepted(journal.records):
                    model = _RecordedModel(backend, journal, budget, config["retries"])
                    model.generate(codec.build_prompt(handler, players))
                for repetition in range(config["repetitions"]):
                    path = model_root / "tables" / case["id"] / f"{repetition:02d}.jsonl"
                    journal = _Journal(path, root)
                    saved = _accepted(journal.records)
                    pending = [mask for mask in range(2 ** len(players)) if mask not in saved]
                    random.Random(f"{config['seed']}:{case['id']}:{repetition}").shuffle(pending)
                    if pending:
                        print(f"{settings['name']}: {case['id']} table {repetition + 1}, "
                              f"{len(pending)} coalitions remaining", flush=True)
                        model = _RecordedModel(backend, journal, budget, config["retries"])

                        def generate(mask):
                            indexes = [key for position, key in enumerate(players) if mask & (1 << position)]
                            model.generate(codec.build_prompt(handler, indexes))

                        with ThreadPoolExecutor(max_workers=config["threads"]) as executor:
                            for _ in executor.map(generate, pending):
                                pass
            trials = [(case, repetition, method_index, cached)
                      for case in config["cases"] if len(case["symptoms"]) in config["live_feature_counts"]
                      for repetition in range(config["repetitions"])
                      for method_index in range(len(METHODS)) for cached in (False, True)]
            random.Random(config["seed"]).shuffle(trials)
            for case, repetition, method_index, cached in trials:
                trial = (model_root / "live" / case["id"] / f"{repetition:02d}"
                         / f"{method_index}_{'cache' if cached else 'redraw'}")
                if (trial / "result.json").exists():
                    continue
                # Failed live trials are restarted, never stitched into a measured run.
                attempt = 1 + len(list(trial.glob("attempt_*")))
                journal = _Journal(trial / f"attempt_{attempt:03d}" / "responses.jsonl", root)
                model = _RecordedModel(backend, journal, budget, config["retries"])
                reference = next(iter(_accepted(_records(
                    model_root / "references" / f"{case['id']}.jsonl")).values()))
                reference_generation = Generation(output=reference)

                def payoff(base, generation):
                    return similarity(reference_generation, generation)

                handler = _handler(case)
                players = handler.get_keys(exclude_permanent_keys=True)
                method = METHODS[method_index]
                attribution = ShapleyAttribution(model=model, data_handler=handler,
                                                prompt_codec=_CoalitionCodec(system=case["system_prompt"]),
                                                sampler=_sampler(method, players, config["seed"]),
                                                use_cache=cached, verbose=False,
                                                num_threads=config["threads"], value_function=payoff)
                print(f"{settings['name']}: {case['id']} live {repetition + 1}, "
                      f"{method}, cache={cached}", flush=True)
                started = time.perf_counter()
                result = attribution.attribution()
                elapsed = time.perf_counter() - started
                scores = [result.attribution[f"symptom_{index}"]["score"]
                          for index in range(1, len(players) + 1)]
                _write_json(trial / "result.json",
                            {"model": settings["name"], "case": case["id"], "method": method,
                             "cache": cached, "repetition": repetition, "scores": scores,
                             "empty_payoff": result.empty_baseline,
                             "full_payoff": result.grand_coalition_value,
                             "efficiency_error": abs(fsum(scores) - (
                                 result.grand_coalition_value - result.empty_baseline)),
                             "elapsed_seconds": elapsed, "journal": journal.path.relative_to(root).as_posix(),
                             "generation_attempts": sum(r["event"] == "start" for r in journal.records)})
        finally:
            backend.close()
    print(f"Collection complete: {budget.used} generation attempts recorded.", flush=True)


def _accuracy_rows(root, config):
    rows = []
    similarity = EmbeddingCosineSimilarity(model_name=EMBEDDING_MODEL)
    for model in config["models"]:
        model_root = root / model["name"]
        for case in config["cases"]:
            references = _accepted(_records(model_root / "references" / f"{case['id']}.jsonl"))
            if not references:
                continue
            reference = Generation(output=next(iter(references.values())))
            for repetition in range(config["repetitions"]):
                outputs = _accepted(_records(model_root / "tables" / case["id"] / f"{repetition:02d}.jsonl"))
                size = 2 ** len(case["symptoms"])
                if set(outputs) != set(range(size)):
                    continue
                payoffs = [similarity(reference, Generation(output=outputs[mask])) for mask in range(size)]
                gold = _attributions(payoffs, METHODS[-1], config["seed"])
                for method in METHODS:
                    seeds = range(config["sampler_seeds"]) if method.startswith("Stratified") else range(1)
                    for offset in seeds:
                        scores = gold if method == METHODS[-1] else _attributions(payoffs, method, config["seed"] + offset)
                        rows.append({"model": model["name"], "case": case["id"],
                                     "repetition": repetition, "method": method,
                                     "sampler_seed": config["seed"] + offset,
                                     "scores": scores, "cosine": _cosine(scores, gold),
                                     "mae": mean(abs(a - b) for a, b in zip(scores, gold))})
    return rows


def _mean_or_none(values):
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def _cell(values):
    values = [value for value in values if value is not None]
    if not values:
        return "N/A"
    average = mean(values)
    return f"{average:.4f} ± {stdev(values):.4f}" if len(values) > 1 else f"{average:.4f}"


def _analyze(root, manifest):
    config = manifest["config"]
    accuracy = _accuracy_rows(root, config)
    live = [json.loads(path.read_text()) for path in sorted(root.glob("*/live/**/result.json"))]
    accuracy_lines = ["# Approximation accuracy", "",
                      "Same fixed reference and realized payoff table for all methods.",
                      "Each cell is mean ± sample SD across case means; seeds and tables are averaged within cases.",
                      "Zero-vector cosine is undefined and excluded, with its count reported.",
                      "Exact refers to each sampled game, not the unknown expected-payoff game.", "",
                      "| Model | Method | Cases | Tables | Cosine ↑ | MAE ↓ | Undefined cosine |",
                      "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    stochastic_lines = ["# Stochastic behavior", "",
                        "One generation per evaluation; fresh cache per trial and fixed sampler seed across repetitions.",
                        "Variation is the RMS of per-feature sample SD across independent repetitions of the same case.",
                        "Efficiency error uses each trial's own recorded full and empty endpoints.",
                        "Cells are mean ± sample SD across cases. Timing includes durable response logging.",
                        "Single-repetition cases have undefined variation and are excluded from that column.", "",
                        "| Model | Method | Cache | Cases | Trials | Variation ↓ | Efficiency error ↓ | Time (s) |",
                        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for model in config["models"]:
        for method in METHODS:
            group = [row for row in accuracy if row["model"] == model["name"] and row["method"] == method]
            by_case = defaultdict(list)
            for row in group:
                by_case[row["case"]].append(row)
            cosine = [_mean_or_none([row["cosine"] for row in values]) for values in by_case.values()]
            errors = [mean(row["mae"] for row in values) for values in by_case.values()]
            tables = len({(row["case"], row["repetition"]) for row in group})
            undefined = sum(row["cosine"] is None for row in group)
            accuracy_lines.append(f"| {model['name']} | {method} | {len(by_case)} | {tables} | "
                                  f"{_cell(cosine)} | {_cell(errors)} | {undefined} |")
            for cached in (False, True):
                group = [row for row in live if row["model"] == model["name"]
                         and row["method"] == method and row["cache"] == cached]
                by_case = defaultdict(list)
                for row in group:
                    by_case[row["case"]].append(row)
                variation = [sqrt(mean(variance(column) for column in zip(*(row["scores"] for row in values))))
                             for values in by_case.values() if len(values) > 1]
                efficiency = [mean(row["efficiency_error"] for row in values) for values in by_case.values()]
                elapsed = [mean(row["elapsed_seconds"] for row in values) for values in by_case.values()]
                stochastic_lines.append(f"| {model['name']} | {method} | {'On' if cached else 'Off'} | "
                                        f"{len(by_case)} | {len(group)} | {_cell(variation)} | "
                                        f"{_cell(efficiency)} | {_cell(elapsed)} |")
    starts = 0
    failures = 0
    tokens = Counter()
    resolved_models = set()
    for path in root.rglob("*.jsonl"):
        for record in _records(path):
            starts += record["event"] == "start"
            failures += record["event"] == "error" or (
                record["event"] == "response" and not record["accepted"])
            if record["event"] == "response":
                resolved_models.add(record["resolved_model"])
                usage = record["usage"] or {}
                for name in ("input_tokens", "output_tokens", "total_tokens"):
                    tokens[name] += usage.get(name, 0)
                tokens["reasoning_tokens"] += (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
    complete = (len(accuracy) == len(config["models"]) * len(config["cases"]) * config["repetitions"]
                * (3 + 2 * config["sampler_seeds"])
                and len(live) == len(config["models"]) * config["repetitions"] * 2 * len(METHODS)
                * sum(len(case["symptoms"]) in config["live_feature_counts"] for case in config["cases"]))
    banner = ("Complete collection." if complete else "PARTIAL COLLECTION: coverage differs; do not treat as final results.")
    if all(model["name"] == "dummy" for model in config["models"]):
        banner += " SYNTHETIC OFFLINE DATA; not paper results."
    for name, lines in (("accuracy.md", accuracy_lines), ("stochastic.md", stochastic_lines)):
        (root / name).write_text(banner + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    _write_json(root / "results.json",
                {"complete": complete, "plan": _plan(manifest), "generation_attempts": starts,
                 "failed_or_rejected_attempts": failures, "reported_tokens": dict(tokens),
                 "resolved_models": sorted(resolved_models), "accuracy": accuracy, "live": live})
    print(f"{banner}\nTables: {root / 'accuracy.md'} and {root / 'stochastic.md'}", flush=True)


def _positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        command = commands.add_parser(name, help="Estimate calls without API access." if name == "plan"
                                      else "Collect responses and write both tables.")
        command.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES[:2]))
        command.add_argument("--dataset", type=Path, default=DATASET)
        command.add_argument("--feature-counts", nargs="+", type=int, choices=range(4, 11), default=list(range(4, 11)))
        command.add_argument("--cases-per-count", type=_positive, default=2)
        command.add_argument("--repetitions", type=_positive, default=3)
        command.add_argument("--sampler-seeds", type=_positive, default=20)
        command.add_argument("--live-feature-counts", nargs="+", type=int, choices=range(4, 11), default=[4, 6])
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--threads", type=_positive, default=10)
        command.add_argument("--max-requests", type=_positive, default=50000,
                             help="Hard generation-attempt limit, including retries and earlier attempts on resume.")
        command.add_argument("--timeout", type=_positive, default=180)
        command.add_argument("--retries", type=int, choices=range(4), default=2)
        command.add_argument("--max-output-tokens", type=_positive, default=100000,
                             help="Output-token limit for all models, capped at each model maximum. Default: 100000.")
        if name == "run":
            command.add_argument("--output", type=Path,
                                 default=OUTPUTS / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    for name in ("resume", "analyze"):
        command = commands.add_parser(name, help="Continue saved configuration." if name == "resume"
                                      else "Regenerate tables offline; never contacts the API.")
        command.add_argument("directory", type=Path)
        if name == "resume":
            command.add_argument("--max-requests", type=_positive,
                                 help="Explicitly increase the saved generation-attempt budget.")
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command in {"plan", "run"}:
            manifest = _manifest(args)
            plan = _plan(manifest)
            print(json.dumps(plan, indent=2), flush=True)
            if args.command == "plan":
                return 0
            if plan["total_calls_excluding_retries"] > args.max_requests:
                raise ValueError("Planned collection exceeds --max-requests; reduce the scope or set an explicit larger budget.")
            root = args.output.resolve()
            root.mkdir(parents=True, exist_ok=False)
            _write_json(root / "manifest.json", manifest)
        else:
            root = args.directory.resolve()
            manifest = _load_manifest(root)
        with _run_lock(root):
            if args.command == "resume" and args.max_requests is not None:
                old_limit = manifest["config"]["max_requests"]
                if args.max_requests <= old_limit:
                    raise ValueError("The replacement budget must exceed the saved limit.")
                manifest.setdefault("budget_changes", []).append(
                    {"at": _timestamp(), "old": old_limit, "new": args.max_requests})
                manifest["config"]["max_requests"] = args.max_requests
                _write_json(root / "manifest.json", manifest)
            if args.command != "analyze":
                dataset = manifest["dataset"]
                if _digest(Path(dataset["path"]).read_bytes()) != dataset["sha256"]:
                    raise ValueError("Dataset changed; refusing to resume mixed data.")
                _collect(root, manifest)
            _analyze(root, manifest)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"Benchmark stopped: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Responses are saved; resume the same run directory.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
