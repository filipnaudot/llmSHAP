import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean

if __package__ in {None, ""}: sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmSHAP.types import Any
from llmSHAP import Attribution, DataHandler, BasicPromptCodec, Generation, ShapleyAttribution, EmbeddingCosineSimilarity
from llmSHAP.llm import OpenAIInterface, DummyLLM, LLMInterface
from llmSHAP.attribution_methods import CounterfactualSampler, SlidingWindowSampler, FullEnumerationSampler
from data import SymptomDataset
from utils import AttributionComparator, plot_similarities, plot_similarity_convergence, plot_timing


COUNTERFACTUAL_METHOD_NAME = "Counterfactual"
SLIDING_WINDOW_METHOD_SIZE = 3; SLIDING_WINDOW_METHOD_NAME = f"Sliding window (w={SLIDING_WINDOW_METHOD_SIZE})"
SHAPLEY_METHOD_NAME = "Shapley value"
GOLD_STANDARD_CONFIG = (SHAPLEY_METHOD_NAME, False)

CHECKPOINTS_DIRECTORY = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINT_PATH = CHECKPOINTS_DIRECTORY / "checkpoint.json"
RESULTS_DIRECTORY = Path(__file__).resolve().parent / "results"
RESULTS_PATH = RESULTS_DIRECTORY / "RESULTS.md"



def _build_data_handler(data: SymptomDataset) -> DataHandler:
    """Build the prompt handler.

    Args:
        data: Input datapoint.

    Returns:
        Configured data handler.
    """
    prompt_dict = {"initial_query": "A patient is showing the following symptom(s):"}
    for index, concept in enumerate(data.concepts(), start=1):
        prompt_dict[f"symptom_{index}"] = f"\nSYMPTOM: {concept}"
    prompt_dict["end_query"] = ("\nBased on the symptom(s), what disease or condition do you think the patient most likely have?"
                                "\nANSWER BRIEFLY.")
    return DataHandler(prompt_dict, permanent_keys={"initial_query", "end_query"})


def _format_method_name(method_name: str, use_cache: bool) -> str:
    """Return the display name for one method config.

    Args:
        method_name: Base method name.
        use_cache: Whether generation cache is enabled.

    Returns:
        Display name with cache suffix when needed.
    """
    return f"{method_name} (cache)" if use_cache else method_name


def _calculate_efficiency(attribution_result: Attribution) -> float:
    """Return efficiency as a percentage.

    Args:
        attribution_result: Attribution output.

    Returns:
        Efficiency percentage.
    """
    attribution_total = attribution_result.empty_baseline + sum([value["score"] for value in attribution_result.attribution.values()])
    return (attribution_result.grand_coalition_value / attribution_total) * 100
    

def _write_checkpoint(data_index: int, results: dict[str, list[dict[str, Any]]]) -> None:
    """Persist the current benchmark state.

    Args:
        data_index: Last completed datapoint index.
        results: Collected benchmark results.
    """
    CHECKPOINTS_DIRECTORY.mkdir(exist_ok=True)
    entry = {"data_index": data_index, "results": results}
    with CHECKPOINT_PATH.open("w", encoding="utf-8") as f:
        json.dump(entry, f, default=str)


def _load_checkpoint() -> dict[str, Any] | None:
    """Load a checkpoint and normalize legacy format.

    Returns:
        Checkpoint dict or ``None``.
    """
    if not CHECKPOINT_PATH.exists(): return None
    with CHECKPOINT_PATH.open(encoding="utf-8") as f:
        checkpoint = json.load(f)
    if "results" in checkpoint: return checkpoint
    timing_results = checkpoint.get("timing", {})
    attribution_results = checkpoint.get("attribution_results", {})
    results = {}
    for method_name, _, use_cache in _create_samplers([]):
        display_name = _format_method_name(method_name, use_cache)
        method_timings = timing_results.get(display_name, timing_results.get(method_name, []))
        method_attributions = attribution_results.get(display_name, attribution_results.get(method_name, []))
        results[display_name] = [
            {
                "attribution": attribution_entry["attribution"],
                "feature_count": attribution_entry["feature_count"],
                "time": timing_entry["time"],
                "efficiency": attribution_entry["efficiency"],
            }
            for attribution_entry, timing_entry in zip(method_attributions, method_timings)
        ]
    checkpoint["results"] = results
    return checkpoint


def _write_results_markdown(results: dict[str, list[dict[str, Any]]], model_name: str) -> None:
    """Write the benchmark summary table.

    Args:
        results: Collected benchmark results.
        model_name: Evaluated model name.
    """
    RESULTS_DIRECTORY.mkdir(exist_ok=True)
    gold_method_name = _format_method_name(*GOLD_STANDARD_CONFIG)
    similarities = AttributionComparator(gold_method_name=gold_method_name).compare(results)
    feature_counts = sorted({result["feature_count"] for method_results in results.values() for result in method_results})
    feature_count_frequency = {
        feature_count: sum(1 for result in results[gold_method_name] if result["feature_count"] == feature_count)
        for feature_count in feature_counts
    } if results.get(gold_method_name) else {}
    table_rows = []
    for method_name, _, use_cache in _create_samplers([]):
        display_name = _format_method_name(method_name, use_cache)
        method_results = results.get(display_name, [])
        if not method_results:
            similarity = "N/A"
            average_time = "N/A"
            average_efficiency = "N/A"
            feature_count_spread = "N/A"
        else:
            similarity_value = 1.0 if display_name == gold_method_name else similarities.get(display_name, {}).get("mean_similarity")
            similarity = "N/A" if similarity_value is None else f"{similarity_value:.4f}"
            average_time = f"{mean(result['time'] for result in method_results):.4f}"
            average_efficiency = f"{mean(result['efficiency'] for result in method_results):.2f}%"
            feature_count_summary = similarities.get(display_name, {}).get("feature_count_summary")
            if display_name == gold_method_name:
                feature_count_spread = "0.0000"
            elif feature_count_summary is None:
                feature_count_spread = "N/A"
            else:
                feature_count_spread = f"{feature_count_summary['spread']:.4f}"
        table_rows.append(
            f"| {display_name} | {model_name} | {similarity} | {feature_count_spread} | "
            f"{average_time} | {average_efficiency} |"
        )
    coverage_summary = "N/A"
    if feature_counts:
        frequencies = sorted(set(feature_count_frequency.values()))
        if len(frequencies) == 1:
            coverage_summary = f"{feature_counts[0]}-{feature_counts[-1]} features ({frequencies[0]} cases per count)"
        else:
            coverage_summary = (
                f"{feature_counts[0]}-{feature_counts[-1]} features "
                f"(cases per count vary from {frequencies[0]} to {frequencies[-1]})"
            )
    markdown = "\n".join(["# Benchmark Results", "", f"Feature-count coverage: {coverage_summary}", "",
                          "| Method | Model | Similarity | Spread | Average Time (s) | Efficiency |",
                          "| --- | --- | ---: | ---: | ---: | ---: |",
                          *table_rows, "",])
    with RESULTS_PATH.open("w", encoding="utf-8") as f: f.write(markdown)


def _create_samplers(players: list[int]) -> list[tuple[str, Any, bool]]:
    """Create the sampler configuration.

    Args:
        players: Feature keys for the datapoint.

    Returns:
        Sampler configuration tuples.
    """
    return [(COUNTERFACTUAL_METHOD_NAME, CounterfactualSampler(), False),
            (COUNTERFACTUAL_METHOD_NAME, CounterfactualSampler(), True),
            (SLIDING_WINDOW_METHOD_NAME, SlidingWindowSampler(players, w_size=SLIDING_WINDOW_METHOD_SIZE), False),
            (SLIDING_WINDOW_METHOD_NAME, SlidingWindowSampler(players, w_size=SLIDING_WINDOW_METHOD_SIZE), True),
            (SHAPLEY_METHOD_NAME, FullEnumerationSampler(len(players)), True),
            (SHAPLEY_METHOD_NAME, FullEnumerationSampler(len(players)), False),]
    

def _get_llm(args: argparse.Namespace) -> tuple[LLMInterface, str]:
    model_name: str = "gpt-4.1-mini"
    llm: LLMInterface = OpenAIInterface(model_name=model_name, temperature=0.2, max_tokens=64)
    if args.dummy_llm:
        model_name = "Dummy model"
        llm = DummyLLM(model_name=model_name, random=False)
    elif args.reasoning:
        model_name = "gpt-5-nano"
        llm = OpenAIInterface(model_name=model_name, reasoning="low", max_tokens=1024)
    return llm, model_name




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-from-checkpoint", action="store_true", help="Resume from the saved checkpoint.")
    parser.add_argument("--threads", type=int, default=10, help="Number of threads for coalition evaluation. Default is 10.")
    parser.add_argument("--debug", action="store_true", help="Print full outputs and attribution details.")
    parser.add_argument("--verbose", action="store_true", help="Print progress and timing information.")
    llm_group = parser.add_mutually_exclusive_group()
    llm_group.add_argument("--dummy_llm", action="store_true", help="Use dummy LLM interface.")
    llm_group.add_argument("--reasoning", action="store_true", help="Use reasoning model.")
    args = parser.parse_args()

    llm, model_name = _get_llm(args)
    if args.verbose:
        print(f"Model: {model_name}")
    
    checkpoint = _load_checkpoint() if args.start_from_checkpoint else None
    if checkpoint is None:
        results = {_format_method_name(method_name, use_cache): [] for method_name, _, use_cache in _create_samplers([])}
        start_index = 0
    else:
        results = checkpoint["results"]
        start_index = checkpoint["data_index"] + 1

    data = SymptomDataset.load()
    for data_index, entry in enumerate(data[start_index:], start=start_index):
        handler = _build_data_handler(entry)
        players = handler.get_keys(exclude_permanent_keys=True)
        if args.verbose: print(f"\n\nFeatures: {len(players)}")
        samplers = _create_samplers(players)
        for name, sampler, cache in samplers:
            display_name = _format_method_name(name, cache)
            if args.verbose: print(f"Method: {display_name}", end="\n     ")
            shap = ShapleyAttribution(model=llm,
                                      data_handler=handler,
                                      prompt_codec=BasicPromptCodec(system=entry.system_prompt()),
                                      sampler=sampler,
                                      use_cache=cache,
                                      verbose=False,
                                      num_threads=args.threads,)
                                    #   value_function=EmbeddingCosineSimilarity(model_name = "text-embedding-3-small", api_url_endpoint = "https://api.openai.com/v1"))
                                    #   value_function=EmbeddingCosineSimilarity())
            
            start_time = time.perf_counter() # Start clock
            result = shap.attribution()
            elapsed = time.perf_counter() - start_time # Stop clock

            if args.verbose: print(f"Time: {elapsed}")
            if args.debug: print("\n\n### OUTPUT ###"); print(result.output); print("\n\n### ATTRIBUTION ###"); print(result.attribution)

            efficiency = _calculate_efficiency(result)
            results[display_name].append({"attribution": result.attribution,
                                          "feature_count": len(players),
                                          "time": elapsed,
                                          "efficiency": efficiency,})

        similarities = AttributionComparator(gold_method_name=_format_method_name(*GOLD_STANDARD_CONFIG)).compare(results)
        plot_similarities(similarities)
        plot_similarity_convergence(similarities)
        plot_timing(results, normalize=True)
        _write_checkpoint(data_index, results)
        _write_results_markdown(results, model_name)
