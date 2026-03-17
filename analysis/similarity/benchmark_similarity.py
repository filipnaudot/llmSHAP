import argparse
import json
import sys
import time
from pathlib import Path
if __package__ in {None, ""}: sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmSHAP import DataHandler, BasicPromptCodec, ShapleyAttribution, EmbeddingCosineSimilarity
from llmSHAP.llm import OpenAIInterface
from llmSHAP.attribution_methods import CounterfactualSampler, SlidingWindowSampler, FullEnumerationSampler
from data import SymptomDataset
from utils import AttributionComparator, plot_similarities, plot_similarity_convergence, plot_timing

COUNTERFACTUAL_METHOD_NAME = "Counterfactual"
SLIDING_WINDOW_METHOD_SIZE = 3; SLIDING_WINDOW_METHOD_NAME = f"Sliding window (w={SLIDING_WINDOW_METHOD_SIZE})"
SHAPLEY_CACHE_METHOD_NAME = "Shapley value - Cache"
SHAPLEY_METHOD_NAME = "Shapley value"
METHOD_NAMES = (
    COUNTERFACTUAL_METHOD_NAME,
    SLIDING_WINDOW_METHOD_NAME,
    SHAPLEY_CACHE_METHOD_NAME,
    SHAPLEY_METHOD_NAME,
)

CHECKPOINTS_DIRECTORY = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINT_PATH = CHECKPOINTS_DIRECTORY / "checkpoint.json"


def _build_data_handler(data: SymptomDataset):
    prompt_dict = {"initial_query": "A patient is showing the following symptom(s):"}
    for index, concept in enumerate(data.concepts(), start=1):
        prompt_dict[f"symptom_{index}"] = f"\nSYMPTOM: {concept}"
    prompt_dict["end_query"] = ("\nBased on these symptom(s), what disease or condition do you think they most likely have?"
                                "\nANSWER BRIEFLY.")
    return DataHandler(prompt_dict, permanent_keys={"initial_query", "end_query"})


def _calculate_efficiency(attribution_result):
    attribution_total = attribution_result.empty_baseline + sum([value["score"] for value in attribution_result.attribution.values()])
    return (attribution_result.grand_coalition_value / attribution_total) * 100
    

def _write_checkpoint(data_index, timing_results, attribution_results):
    CHECKPOINTS_DIRECTORY.mkdir(exist_ok=True)
    entry = {
        "data_index": data_index,
        "timing": timing_results,
        "attribution_results": attribution_results,
    }
    with CHECKPOINT_PATH.open("w", encoding="utf-8") as f:
        json.dump(entry, f, default=str)


def _load_checkpoint():
    if not CHECKPOINT_PATH.exists(): return None
    with CHECKPOINT_PATH.open(encoding="utf-8") as f: return json.load(f)


def create_samplers(players):
    return [(COUNTERFACTUAL_METHOD_NAME, CounterfactualSampler(), False),
            (SLIDING_WINDOW_METHOD_NAME, SlidingWindowSampler(players, w_size=SLIDING_WINDOW_METHOD_SIZE), False),
            (SHAPLEY_CACHE_METHOD_NAME, FullEnumerationSampler(len(players)), True),
            (SHAPLEY_METHOD_NAME, FullEnumerationSampler(len(players)), False),]
    

if __name__ == "__main__":
    DEBUG = False
    VERBOSE = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-from-checkpoint", action="store_true")
    parser.add_argument("--threads", type=int, default=10)
    args = parser.parse_args()

    prompt_codec = BasicPromptCodec(system="Answer the question briefly.")
    # llm = OpenAIInterface(model_name="gpt-4.1-mini", temperature=0.2, max_tokens=64)
    llm = OpenAIInterface(model_name="gpt-4.1-mini", temperature=0.0, seed=42, max_tokens=64)
    data = SymptomDataset.load()

    checkpoint = _load_checkpoint() if args.start_from_checkpoint else None
    if checkpoint is None:
        timing_results = {method_name: [] for method_name in METHOD_NAMES}
        attribution_results = {method_name: [] for method_name in METHOD_NAMES}
        start_index = 0
    else:
        timing_results = checkpoint["timing"]
        attribution_results = checkpoint["attribution_results"]
        start_index = checkpoint["data_index"] + 1

    for data_index, entry in enumerate(data[start_index:], start=start_index):
        handler = _build_data_handler(entry)
        players = handler.get_keys(exclude_permanent_keys=True)
        if VERBOSE: print(f"\n\nFeatures: {len(players)}")
        samplers = create_samplers(players)
        for name, sampler, cache in samplers:
            if VERBOSE: print(f"Method: {name}", end="\n     ")
            shap = ShapleyAttribution(model=llm,
                                      data_handler=handler,
                                      prompt_codec=prompt_codec,
                                      sampler=sampler,
                                      use_cache=cache,
                                      verbose=False,
                                      num_threads=args.threads,
                                    #   value_function=EmbeddingCosineSimilarity(model_name = "text-embedding-3-small", api_url_endpoint = "https://api.openai.com/v1")
                                      value_function=EmbeddingCosineSimilarity()
                                    )
            
            start_time = time.perf_counter() # Start clock
            result = shap.attribution()
            elapsed = time.perf_counter() - start_time # Stop clock

            if VERBOSE: print(f"Time: {elapsed}")
            if DEBUG: print("\n\n### OUTPUT ###"); print(result.output); print("\n\n### ATTRIBUTION ###"); print(result.attribution)

            efficiency = _calculate_efficiency(result)
            attribution_results[name].append({"attribution": result.attribution,
                                              "feature_count": len(players),
                                              "efficiency": efficiency,})
            timing_results[name].append({"num_features": len(players), "time": elapsed})


        similarities = AttributionComparator(gold_method_name=SHAPLEY_METHOD_NAME).compare(attribution_results)
        plot_similarities(similarities)
        plot_similarity_convergence(similarities)
        plot_timing(timing_results)
        _write_checkpoint(data_index, timing_results, attribution_results)
