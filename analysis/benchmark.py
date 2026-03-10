import json
import time
from statistics import mean

import matplotlib.pyplot as plt

from llmSHAP import DataHandler, BasicPromptCodec, ShapleyAttribution, EmbeddingCosineSimilarity
from llmSHAP.llm import OpenAIInterface
from llmSHAP.attribution_methods import CounterfactualSampler, SlidingWindowSampler, FullEnumerationSampler
from comparison import AttributionComparator
from data import SymptomDataset

NUM_THREADS = 3
COUNTERFACTUAL_METHOD_NAME = "Counterfactual"
SLIDING_WINDOW_METHOD_NAME = "Sliding window (w=3)"; SLIDING_WINDOW_METHOD_SIZE = 3
SHAPLEY_CACHE_METHOD_NAME = "Shapley value - Cache"
SHAPLEY_METHOD_NAME = "Shapley value"
METHOD_NAMES = (
    COUNTERFACTUAL_METHOD_NAME,
    SLIDING_WINDOW_METHOD_NAME,
    SHAPLEY_CACHE_METHOD_NAME,
    SHAPLEY_METHOD_NAME,
)


def _build_data_handler(data: SymptomDataset):
    prompt_dict = {"initial_query": "A patient is showing the following symptom(s):"}
    for index, concept in enumerate(data.concepts(), start=1):
        prompt_dict[f"symptom_{index}"] = f"\nSYMPTOM: {concept}"
    prompt_dict["end_query"] = ("\nBased on these symptom(s), what disease or condition do you think they most likely have?"
                                "\nANSWER BRIEFLY.")
    return DataHandler(prompt_dict, permanent_keys={"initial_query", "end_query"})


def _calculate_efficiency(attribution_result):
    attribution_total = attribution_result.empty_baseline + sum([value["score"] for value in attribution_result.attribution.values()])
    return (attribution_result.grand_coalition_value // attribution_total)*100
    

def _plot_similarities(similarities):
    # Bar plot
    method_names = list(similarities.keys())
    mean_similarities = [method_stats["mean_similarity"] for method_stats in similarities.values()]
    plt.bar(method_names, mean_similarities)
    plt.ylabel("Mean Similarity")
    plt.title("Attribution Similarity to standard Shapley value")
    plt.xticks(rotation=30, ha="right")
    for index, mean_value in enumerate(mean_similarities):
        plt.text(index, mean_value, f"{mean_value:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig("./similarities_chart.png")
    plt.close()

    # Line plot
    any_series_plotted = False
    for method_name, method_stats in similarities.items():
        average_similarity_by_feature_count = method_stats.get("by_feature_count", {})
        if not average_similarity_by_feature_count: continue
        sorted_feature_counts = sorted(average_similarity_by_feature_count.keys())
        averaged_similarities = [average_similarity_by_feature_count[feature_count] for feature_count in sorted_feature_counts]
        plt.plot(sorted_feature_counts, averaged_similarities, marker="o", label=method_name)
        any_series_plotted = True
    if any_series_plotted:
        plt.xlabel("Number of Features")
        plt.ylabel("Average Similarity")
        plt.title("Average Similarity vs. Feature Count")
        plt.legend()
        plt.xticks(sorted_feature_counts)
        plt.tight_layout()
        plt.savefig("./similarities_by_feature_count.png")
        plt.close()


def _plot_timing(timing_results):
    for name, results in timing_results.items():
        grouped = {}
        for result in results:
            grouped.setdefault(result["num_features"], []).append(result["time"])
        num_features_list = sorted(grouped.keys())
        time_result_list = [mean(grouped[x]) for x in num_features_list]
        plt.plot(num_features_list, time_result_list, marker='o', label=name)
    plt.xlabel("Number of Features")
    plt.ylabel("Average Time (s)")
    plt.title("Attribution Runtime by Number of Features")
    plt.legend()
    plt.xticks(sorted(num_features_list))
    plt.tight_layout()
    plt.savefig("./timing_chart.png")

    # Second plot: log-scaled y-axis
    plt.yscale("log")
    plt.title("Attribution Runtime by Number of Features (Log Scale)")
    plt.savefig("./timing_chart_log.png")
    plt.close()


def _plot_similarity_convergence(similarities):
    for method_name, method_stats in similarities.items():
        per_datapoint_similarities = method_stats["per_datapoint"]
        if not per_datapoint_similarities: continue

        running_mean_values = []
        cumulative_similarity_sum = 0.0
        for number_of_points, current_similarity_value in enumerate(per_datapoint_similarities, start=1):
            cumulative_similarity_sum += current_similarity_value
            running_mean_values.append(cumulative_similarity_sum / number_of_points)

        # The "final" target is the running mean after all datapoints
        final_mean_similarity = running_mean_values[-1]
        absolute_differences_to_final = [abs(running_mean - final_mean_similarity) for running_mean in running_mean_values[:-1]]
        plt.plot(range(1, len(absolute_differences_to_final) + 1), absolute_differences_to_final, marker="o", label=method_name,)

    plt.xlabel("Number of Data Points")
    plt.ylabel("Running mean and Final mean Diff")
    plt.title("Similarity Convergence to Gold Standard")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./similarity_convergence.png")
    plt.close()


def _log(i, similarities, timing_results, attribution_results=None):
    entry = {
        "data_index": i,
        "similarities": similarities,
        "timing": timing_results,
    }
    with open("log.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    if attribution_results:
        attribution_results_entry = {"attribution_results": attribution_results,}
        with open("full_attribution_log.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps(attribution_results_entry, default=str) + "\n")



if __name__ == "__main__":
    DEBUG = False
    VERBOSE = True

    prompt_codec = BasicPromptCodec(system="Answer the question briefly.")
    llm = OpenAIInterface("gpt-4.1-mini", temperature=0.2, max_tokens=64)
    data = SymptomDataset.load()
    
    timing_results = {method_name: [] for method_name in METHOD_NAMES}
    attribution_results = {method_name: [] for method_name in METHOD_NAMES}
    for i, entry in enumerate(data):
        handler = _build_data_handler(entry)
        players = handler.get_keys(exclude_permanent_keys=True)
        if VERBOSE: print(f"\n\nFeatures: {len(players)}")
        samplers = [
            (COUNTERFACTUAL_METHOD_NAME, CounterfactualSampler(), False),
            (SLIDING_WINDOW_METHOD_NAME, SlidingWindowSampler(players, w_size=SLIDING_WINDOW_METHOD_SIZE), False),
            (SHAPLEY_CACHE_METHOD_NAME, FullEnumerationSampler(len(players)), True),
            (SHAPLEY_METHOD_NAME, FullEnumerationSampler(len(players)), False),
        ]

        for name, sampler, cache in samplers:
            if VERBOSE: print(f"Method: {name}", end="\n     ")
            shap = ShapleyAttribution(model=llm,
                                      data_handler=handler,
                                      prompt_codec=prompt_codec,
                                      sampler=sampler,
                                      use_cache=cache,
                                      verbose=False,
                                      num_threads=NUM_THREADS,
                                      value_function=EmbeddingCosineSimilarity())
            
            start_time = time.perf_counter() # Start clock
            result = shap.attribution()
            elapsed = time.perf_counter() - start_time # Stop clock

            if VERBOSE: print(f"Time: {elapsed}")
            if DEBUG:
                print("\n\n### OUTPUT ###")
                print(result.output)
                print("\n\n### ATTRIBUTION ###")
                print(result.attribution)

            efficiency = _calculate_efficiency(result)
            attribution_results[name].append({"attribution": result.attribution,
                                              "feature_count": len(players),
                                              "efficiency": efficiency,})
            timing_results[name].append({"num_features": len(players), "time": elapsed})


        similarities = AttributionComparator(gold_method_name=SHAPLEY_METHOD_NAME).compare(attribution_results)
        _plot_similarities(similarities)
        _plot_similarity_convergence(similarities)
        _plot_timing(timing_results)
        _log(i, similarities, timing_results, attribution_results=attribution_results)
