# type: ignore
import csv
import math
from statistics import mean
import matplotlib.pyplot as plt
import time
import json

from llmSHAP import DataHandler, BasicPromptCodec, ShapleyAttribution, EmbeddingCosineSimilarity
from llmSHAP.llm import OpenAIInterface
from llmSHAP.attribution_methods import CounterfactualSampler, SlidingWindowSampler, FullEnumerationSampler


def _load_data(file_name):
    data = []
    with open(file_name, newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            disease = row["Disease"].strip()
            symptoms = [
                f"\nSYMPTOM: {value.strip().replace('_', ' ')}" for key, value in row.items()
                if key.startswith("Symptom_") and value and value.strip()
            ]
            data.append({"Disease": disease, "Symptoms": symptoms})
    return data


def _build_data_handler(data):
    prompt_dict = {"initial_query": "A patient is showing the following symptom(s):"}
    for i, symptom in enumerate(data["Symptoms"], start=1):
        prompt_dict[f"symptom_{i}"] = symptom
    prompt_dict["end_query"] = (
        "\nBased on these symptom(s), what disease or condition do you think they most likely have?"
        "\nANSWER BRIEFLY."
    )
    return DataHandler(prompt_dict, permanent_keys={"initial_query", "end_query"})



def _compare_attributions_to_gold(attribution_data, gold_function_name="Shapley value"):
    gold_entries = attribution_data[gold_function_name]
    number_of_datapoints = len(gold_entries)

    def extract_score_vector(attribution_mapping, ordered_feature_keys):
        score_vector = []
        for feature_key in ordered_feature_keys:
            feature_record = attribution_mapping.get(feature_key, {})
            if isinstance(feature_record, dict):
                score_vector.append(float(feature_record.get("score", 0.0)))
            else:
                try:
                    score_vector.append(float(feature_record))
                except Exception:
                    score_vector.append(0.0)
        return score_vector

    def cosine_similarity(vector_a, vector_b):
        dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
        magnitude_a = math.sqrt(sum(a * a for a in vector_a))
        magnitude_b = math.sqrt(sum(b * b for b in vector_b))
        return 0.0 if magnitude_a == 0.0 or magnitude_b == 0.0 else dot_product / (magnitude_a * magnitude_b)

    ordered_feature_keys_per_datapoint = [
        list(gold_entries[i]["attribution"].keys()) for i in range(number_of_datapoints)
    ]
    gold_score_vectors = [
        extract_score_vector(gold_entries[i]["attribution"], ordered_feature_keys_per_datapoint[i])
        for i in range(number_of_datapoints)
    ]
    feature_counts_per_datapoint = [gold_entries[i]["feature_count"] for i in range(number_of_datapoints)]

    similarity_results_by_function = {}
    for method_name, method_entries in attribution_data.items():
        if method_name == gold_function_name:
            continue

        per_datapoint_similarities = []
        similarities_grouped_by_feature_count = {}  # {feature_count: [similarities]}

        for datapoint_index in range(number_of_datapoints):
            method_score_vector = extract_score_vector(
                method_entries[datapoint_index]["attribution"],
                ordered_feature_keys_per_datapoint[datapoint_index],
            )
            similarity_value = cosine_similarity(
                gold_score_vectors[datapoint_index],
                method_score_vector
            )
            per_datapoint_similarities.append(similarity_value)

            feature_count = feature_counts_per_datapoint[datapoint_index]
            similarities_grouped_by_feature_count.setdefault(feature_count, []).append(similarity_value)

        average_similarity_by_feature_count = {
            feature_count: mean(similarities)
            for feature_count, similarities in similarities_grouped_by_feature_count.items()
        }

        similarity_results_by_function[method_name] = {
            "per_datapoint": per_datapoint_similarities,
            "mean_similarity": mean(per_datapoint_similarities) if per_datapoint_similarities else None,
            "by_feature_count": average_similarity_by_feature_count,
        }

    return similarity_results_by_function


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
        absolute_differences_to_final = [
            abs(running_mean - final_mean_similarity) for running_mean in running_mean_values[:-1]
        ]

        plt.plot(
            range(1, len(absolute_differences_to_final) + 1),
            absolute_differences_to_final,
            marker="o",
            label=method_name,
        )

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
        attribution_results_entry = {
            "attribution_results": attribution_results,
        }
        with open("full_attribution_log.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps(attribution_results_entry, default=str) + "\n")



if __name__ == "__main__":
    VERBOSE = False

    prompt_codec = BasicPromptCodec(system="Answer the question briefly.")
    llm = OpenAIInterface("gpt-4.1-mini", temperature=0.2, max_tokens=64)
    data = _load_data("reduced_symptom_dataset.csv")
    

    timing_results = {
        "Counterfactual": [],
        "Sliding window (w=3)": [],
        "Shapley value - Cache": [],
        "Shapley value": [],
    }
    attribution_results = {
        "Counterfactual": [],
        "Sliding window (w=3)": [],
        "Shapley value - Cache": [],
        "Shapley value": [],
    }
    for i, entry in enumerate(data):
        handler = _build_data_handler(entry)
        
        # Init all samplers
        players = handler.get_keys(exclude_permanent_keys=True)
        samplers = [
            # name                          sampler                                  use_cache
            ("Counterfactual",              CounterfactualSampler(),                 False),
            ("Sliding window (w=3)",        SlidingWindowSampler(players, w_size=3), False),
            ("Shapley value - Cache",       FullEnumerationSampler(len(players)),    True),
            ("Shapley value",               FullEnumerationSampler(len(players)),    False) # Gold standard
        ]

        # For each sampler, run attribution
        for name, sampler, cache in samplers:
            shap = ShapleyAttribution(model=llm,
                                    data_handler=handler,
                                    prompt_codec=prompt_codec,
                                    sampler=sampler,
                                    use_cache=cache,
                                    num_threads=3,
                                    similarity_function=EmbeddingCosineSimilarity())
            
            start_time = time.perf_counter() # Start clock
            result = shap.attribution()
            elapsed = time.perf_counter() - start_time # Stop clock

            if VERBOSE:
                print("\n\n### OUTPUT ###")
                print(result.output)
                print("\n\n### ATTRIBUTION ###")
                print(result.attribution)

            efficiency = _calculate_efficiency(result)
            attribution_results[name].append({
                "attribution": result.attribution,
                "feature_count": len(players),
                "efficiency": efficiency,
            })
            timing_results[name].append({"num_features": len(players), "time": elapsed})


        similarities = _compare_attributions_to_gold(attribution_results)
        _plot_similarities(similarities)
        _plot_similarity_convergence(similarities)
        _plot_timing(timing_results)
        _log(i, similarities, timing_results, attribution_results=attribution_results)