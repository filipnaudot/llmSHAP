"""Offline checks for scientific comparisons, accounting, and restart behavior."""

import importlib.util
import json
from math import fsum
from pathlib import Path
from types import SimpleNamespace

import pytest

BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "analysis/benchmark/attribution_benchmark.py"
SPEC = importlib.util.spec_from_file_location("attribution_benchmark", BENCHMARK_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


@pytest.fixture(autouse=True)
def mock_embedding_model(monkeypatch):
    class MockEncoder:
        def encode(self, texts, convert_to_numpy=True):
            vectors = []
            for text in texts:
                vector = [0.0] * 16
                for word in text.split():
                    vector[sum(word.encode()) % len(vector)] += 1
                vectors.append(vector)
            return vectors

    monkeypatch.setattr(benchmark.EmbeddingCosineSimilarity, "_model", MockEncoder())


def _small_args(output):
    return ["run", "--models", "dummy", "--feature-counts", "4",
            "--cases-per-count", "1", "--repetitions", "2", "--sampler-seeds", "2",
            "--threads", "1", "--retries", "0", "--output", str(output)]


def _response(output="A synthetic answer", status="completed"):
    return {"output": output, "status": status, "usage": None,
            "resolved_model": "dummy", "response_id": None, "request_id": None}


def test_plan_is_offline_and_matches_agreed_scope(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("Planning must not construct an API backend.")

    monkeypatch.setattr(benchmark._Backend, "__init__", forbidden)
    monkeypatch.setattr(benchmark.EmbeddingCosineSimilarity, "__init__", forbidden)
    assert benchmark.main(["plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["cases"] == 14
    assert plan["total_calls_excluding_retries"] == 40024
    assert plan["calls_per_model_excluding_retries"] == {
        "references": 14, "shared_tables": 12192, "live_trials": 7806}


@pytest.mark.parametrize("method", benchmark.METHODS)
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("size", [4, 6])
def test_table_attributions_and_call_counts_match_core(method, cached, size):
    payoffs = [0.2 + 0.1 * mask.bit_count() + (0.5 if mask & 3 == 3 else 0)
               for mask in range(1 << size)]
    case = {"symptoms": ["headache", "rash", "cough", "fever", "nausea", "fatigue"][:size]}
    handler = benchmark._handler(case)

    class TableModel(benchmark.LLMInterface):
        def __init__(self):
            self.calls = 0

        def generate(self, prompt, tools=None, images=None):
            self.calls += 1
            return str(payoffs[prompt["mask"]])

    model = TableModel()
    players = handler.get_keys(exclude_permanent_keys=True)
    result = benchmark.ShapleyAttribution(
        model=model, data_handler=handler, prompt_codec=benchmark._CoalitionCodec(),
        sampler=benchmark._sampler(method, players, 42), use_cache=cached,
        verbose=False, num_threads=3, value_function=lambda base, generation: float(generation.output)).attribution()
    actual = [result.attribution[f"symptom_{index}"]["score"] for index in range(1, size + 1)]
    assert actual == pytest.approx(benchmark._attributions(payoffs, method, 42))
    assert model.calls == benchmark._request_count(size, method, cached, 42)
    if method == benchmark.METHODS[-1]:
        assert actual == pytest.approx([0.35, 0.35] + [0.1] * (size - 2))
        assert fsum(actual) == pytest.approx(payoffs[-1] - payoffs[0])


def test_codec_preserves_pinned_context_and_feature_order():
    handler = benchmark._handler({"symptoms": ["headache", "rash", "cough", "fever"]})
    players = handler.get_keys(exclude_permanent_keys=True)
    prompt = benchmark._CoalitionCodec(system="fixed instruction").build_prompt(
        handler, {players[0], players[3]})
    assert prompt["mask"] == 9
    assert prompt["messages"][0]["content"] == "fixed instruction"
    text = prompt["messages"][1]["content"]
    assert "A patient" in text and "ANSWER BRIEFLY." in text
    assert text.index("headache") < text.index("fever")
    assert "rash" not in text and "cough" not in text


def test_zero_cosine_is_undefined_and_signed_scores_are_retained():
    assert benchmark._cosine([0, 0], [1, 2]) is None
    assert benchmark._cosine([1, -1], [-1, 1]) == pytest.approx(-1)


@pytest.mark.parametrize("response", [_response("", "completed"), _response("Partial answer", "incomplete")])
def test_invalid_responses_are_recorded_and_stop_trial(tmp_path, response):
    journal = benchmark._Journal(tmp_path / "responses.jsonl", tmp_path)
    budget = benchmark._Budget(5)
    backend = SimpleNamespace(generate=lambda messages, draw_id: response)
    model = benchmark._RecordedModel(backend, journal, budget, retries=2)
    with pytest.raises(RuntimeError, match="Empty or incomplete"):
        model.generate({"mask": 0, "messages": []})
    assert budget.used == 1
    assert journal.records[-1]["accepted"] is False
    assert benchmark._accepted(journal.records) == {}
    with pytest.raises(RuntimeError, match="earlier generation failure"):
        model.generate({"mask": 1, "messages": []})
    assert budget.used == 1


def test_retry_attempts_count_toward_hard_budget(tmp_path, monkeypatch):
    class RateLimit(RuntimeError):
        status_code = 429
        code = "rate_limit_exceeded"

    calls = []

    def generate(messages, draw_id):
        calls.append(draw_id)
        if len(calls) == 1:
            raise RateLimit()
        return _response()

    monkeypatch.setattr(benchmark.time, "sleep", lambda seconds: None)
    journal = benchmark._Journal(tmp_path / "responses.jsonl", tmp_path)
    budget = benchmark._Budget(2)
    model = benchmark._RecordedModel(SimpleNamespace(generate=generate), journal, budget, retries=2)
    assert model.generate({"mask": 0, "messages": []}) == "A synthetic answer"
    assert budget.used == 2
    assert calls[0] == calls[1]
    with pytest.raises(RuntimeError, match="budget exhausted"):
        model.generate({"mask": 1, "messages": []})
    assert len(calls) == 2


def test_quota_exhaustion_is_not_retried(tmp_path):
    class QuotaError(RuntimeError):
        status_code = 429
        code = "insufficient_quota"

    def generate(messages, draw_id):
        raise QuotaError()

    journal = benchmark._Journal(tmp_path / "responses.jsonl", tmp_path)
    budget = benchmark._Budget(10)
    model = benchmark._RecordedModel(SimpleNamespace(generate=generate), journal, budget, retries=2)
    with pytest.raises(RuntimeError, match="Generation failed"):
        model.generate({"mask": 0, "messages": []})
    assert budget.used == 1


def test_journal_recovers_only_an_unterminated_tail(tmp_path):
    path = tmp_path / "responses.jsonl"
    path.write_bytes(b'{"event":"start","mask":0}\n{"event":')
    journal = benchmark._Journal(path, tmp_path)
    assert journal.records == [{"event": "start", "mask": 0}]
    journal.append({"event": "error", "mask": 0})
    assert len(benchmark._records(path)) == 2
    path.write_bytes(b'{"invalid":}\n')
    with pytest.raises(json.JSONDecodeError):
        benchmark._records(path)


def test_offline_collection_analysis_and_completed_resume(tmp_path, monkeypatch):
    output = tmp_path / "run"
    assert benchmark.main(_small_args(output)) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    results = json.loads((output / "results.json").read_text())
    assert results["complete"]
    assert results["generation_attempts"] == benchmark._plan(manifest)["total_calls_excluding_retries"]
    assert len(results["accuracy"]) == 2 * (3 + 2 * 2)
    assert len(results["live"]) == 20
    gold = [row for row in results["accuracy"] if row["method"] == benchmark.METHODS[-1]]
    assert all(row["mae"] == 0 for row in gold)
    cached_exact = [row for row in results["live"]
                    if row["method"] == benchmark.METHODS[-1] and row["cache"]]
    assert all(row["efficiency_error"] < 1e-12 for row in cached_exact)
    assert "SYNTHETIC OFFLINE DATA" in (output / "accuracy.md").read_text()
    assert benchmark.main(["resume", str(output)]) == 0
    assert json.loads((output / "results.json").read_text())["generation_attempts"] == results["generation_attempts"]

    def forbidden(*args, **kwargs):
        pytest.fail("Offline analysis must not initialize any backend.")

    monkeypatch.setattr(benchmark._Backend, "__init__", forbidden)
    previous = (output / "results.json").read_bytes()
    assert benchmark.main(["analyze", str(output)]) == 0
    assert (output / "results.json").read_bytes() == previous


@pytest.mark.parametrize("phase", ["tables", "live"])
def test_interrupted_collection_reuses_tables_but_restarts_live_trials(tmp_path, monkeypatch, phase):
    output = tmp_path / "run"
    original = benchmark._RecordedModel.generate
    failed = False

    def interrupted(self, prompt, tools=None, images=None):
        nonlocal failed
        if f"/{phase}/" in self.journal.key and not failed and len(benchmark._accepted(self.journal.records)) >= 2:
            failed = True
            self.stopped.set()
            raise RuntimeError("Injected interruption")
        return original(self, prompt, tools, images)

    monkeypatch.setattr(benchmark._RecordedModel, "generate", interrupted)
    assert benchmark.main(_small_args(output)) == 1
    assert failed
    assert benchmark.main(["analyze", str(output)]) == 0
    assert "PARTIAL COLLECTION" in (output / "accuracy.md").read_text()
    monkeypatch.setattr(benchmark._RecordedModel, "generate", original)
    assert benchmark.main(["resume", str(output)]) == 0
    results = json.loads((output / "results.json").read_text())
    assert results["complete"]
    for path in output.glob("*/tables/**/*.jsonl"):
        accepted = [r for r in benchmark._records(path) if r["event"] == "response" and r["accepted"]]
        assert len(accepted) == len({r["mask"] for r in accepted}) == 16
    if phase == "live":
        assert any("attempt_002" in row["journal"] for row in results["live"])


def test_resume_rejects_source_and_dataset_changes(tmp_path, monkeypatch):
    args = benchmark._parser().parse_args(_small_args(tmp_path / "unused"))
    manifest = benchmark._manifest(args)
    output = tmp_path / "saved"
    output.mkdir()

    def forbidden(*args, **kwargs):
        pytest.fail("Mismatched runs must fail before backend initialization.")

    monkeypatch.setattr(benchmark._Backend, "__init__", forbidden)
    manifest["source_sha256"] = {}
    benchmark._write_json(output / "manifest.json", manifest)
    assert benchmark.main(["resume", str(output)]) == 1
    manifest["source_sha256"] = benchmark._source_hashes()
    manifest["dataset"]["sha256"] = "changed"
    benchmark._write_json(output / "manifest.json", manifest)
    assert benchmark.main(["resume", str(output)]) == 1


def test_request_budget_rejects_oversized_run_before_creating_output(tmp_path):
    output = tmp_path / "not-created"
    args = _small_args(output) + ["--max-requests", "1"]
    assert benchmark.main(args) == 1
    assert not output.exists()


def test_single_repetition_reports_undefined_variation(tmp_path):
    output = tmp_path / "pilot"
    args = _small_args(output)
    args[args.index("--repetitions") + 1] = "1"
    assert benchmark.main(args) == 0
    assert "N/A" in (output / "stochastic.md").read_text()


@pytest.mark.parametrize("name", ["gpt-4.1-mini", "gpt-5-nano"])
def test_api_payload_and_usage_capture_without_network(name):
    calls = []
    response = SimpleNamespace(output_text="A complete answer", status="completed",
                               usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 12, "output_tokens": 7}),
                               model=f"{name}-snapshot", id="response_test",
                               _request_id="request_test", incomplete_details=None)

    def create(**kwargs):
        calls.append(kwargs)
        return response

    backend = object.__new__(benchmark._Backend)
    backend.settings = {"name": name, "reasoning": "low" if name == "gpt-5-nano" else None,
                        "temperature": 0.2, "max_output_tokens": 32768 if name == "gpt-4.1-mini" else 100000}
    backend.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    result = backend.generate([{"role": "user", "content": "test"}], "draw_test")
    assert result["usage"] == {"input_tokens": 12, "output_tokens": 7}
    assert result["resolved_model"] == f"{name}-snapshot"
    assert calls[0]["store"] is False
    assert calls[0]["max_output_tokens"] == backend.settings["max_output_tokens"]
    if name == "gpt-5-nano":
        assert calls[0]["reasoning"] == {"effort": "low"}
        assert "temperature" not in calls[0]
    else:
        assert calls[0]["temperature"] == 0.2
        assert "reasoning" not in calls[0]


@pytest.mark.parametrize("command", ["plan", "run"])
def test_shared_output_limit_parser(command):
    default = benchmark._parser().parse_args([command])
    assert default.max_output_tokens == 100000
    custom = benchmark._parser().parse_args([command, "--max-output-tokens", "512"])
    assert custom.max_output_tokens == 512
    manifest = benchmark._manifest(custom)
    assert all(model["max_output_tokens"] == 512 for model in manifest["config"]["models"])


@pytest.mark.parametrize("requested, expected", [(100000, [32768, 100000]),
                                                (512, [512, 512]),
                                                (200000, [32768, 128000])])
def test_output_limit_is_capped_and_recorded_per_model(requested, expected):
    args = benchmark._parser().parse_args(["plan", "--max-output-tokens", str(requested)])
    manifest = benchmark._manifest(args)
    config = manifest["config"]
    assert config["requested_max_output_tokens"] == requested
    assert [model["max_output_tokens"] for model in config["models"]] == expected
    plan = benchmark._plan(manifest)
    assert plan["requested_max_output_tokens"] == requested
    assert plan["effective_max_output_tokens"] == dict(zip(("gpt-4.1-mini", "gpt-5-nano"), expected))


def test_embedding_model_is_recorded():
    manifest = benchmark._manifest(benchmark._parser().parse_args(["plan"]))
    assert manifest["config"]["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert manifest["config"]["payoff"].startswith("embedding cosine")


def test_missing_embeddings_fail_before_model_calls(tmp_path, monkeypatch):
    def missing(*args, **kwargs):
        raise ImportError("Embedding dependencies missing")

    def forbidden(*args, **kwargs):
        pytest.fail("Missing embeddings must fail before initializing an inference backend.")

    monkeypatch.setattr(benchmark, "EmbeddingCosineSimilarity", missing)
    monkeypatch.setattr(benchmark._Backend, "__init__", forbidden)
    assert benchmark.main(_small_args(tmp_path / "run")) == 1
