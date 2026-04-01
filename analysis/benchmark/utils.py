import math
from statistics import mean
from pathlib import Path

import matplotlib.pyplot as plt

PLOTS_DIRECTORY = Path(__file__).resolve().parent / "plots"



###################################################
# ATTRIBUTION COMPARISON
###################################################
class AttributionComparator:
    def __init__(
        self,
        gold_method_name: str = "Shapley value",
        attribution_key: str = "attribution",
        feature_count_key: str = "feature_count",
        score_key: str = "score",
    ) -> None:
        self.gold_method_name = gold_method_name
        self.attribution_key = attribution_key
        self.feature_count_key = feature_count_key
        self.score_key = score_key

    def _extract_score_vector(self, attribution_mapping, ordered_feature_keys):
        score_vector = []
        for feature_key in ordered_feature_keys:
            feature_record = attribution_mapping.get(feature_key, {})
            if isinstance(feature_record, dict):
                score_vector.append(float(feature_record.get(self.score_key, 0.0)))
                continue
            try:
                score_vector.append(float(feature_record))
            except Exception:
                score_vector.append(0.0)
        return score_vector

    @staticmethod
    def _cosine_similarity(vector_a, vector_b):
        dot_product = sum(value_a * value_b for value_a, value_b in zip(vector_a, vector_b))
        magnitude_a = math.sqrt(sum(value * value for value in vector_a))
        magnitude_b = math.sqrt(sum(value * value for value in vector_b))
        return 0.0 if magnitude_a == 0.0 or magnitude_b == 0.0 else dot_product / (magnitude_a * magnitude_b)

    def _summarize_by_feature_count(self, similarities_by_feature_count):
        if not similarities_by_feature_count:
            return None
        sorted_feature_counts = sorted(similarities_by_feature_count)
        mean_similarities = [
            mean(similarities_by_feature_count[feature_count])
            for feature_count in sorted_feature_counts
        ]
        return {
            "min_similarity": min(mean_similarities),
            "max_similarity": max(mean_similarities),
            "spread": max(mean_similarities) - min(mean_similarities),
        }

    def compare(self, attribution_data):
        gold_entries = attribution_data[self.gold_method_name]
        number_of_datapoints = len(gold_entries)
        ordered_feature_keys_per_datapoint = [
            list(gold_entries[datapoint_index][self.attribution_key].keys())
            for datapoint_index in range(number_of_datapoints)
        ]
        gold_score_vectors = [
            self._extract_score_vector(
                gold_entries[datapoint_index][self.attribution_key],
                ordered_feature_keys_per_datapoint[datapoint_index],
            )
            for datapoint_index in range(number_of_datapoints)
        ]
        feature_counts_per_datapoint = [
            gold_entries[datapoint_index][self.feature_count_key]
            for datapoint_index in range(number_of_datapoints)
        ]

        similarity_results_by_method = {}
        for method_name, method_entries in attribution_data.items():
            if method_name == self.gold_method_name:
                continue

            per_datapoint_similarities = []
            similarities_by_feature_count = {}
            for datapoint_index in range(number_of_datapoints):
                method_score_vector = self._extract_score_vector(
                    method_entries[datapoint_index][self.attribution_key],
                    ordered_feature_keys_per_datapoint[datapoint_index],
                )
                similarity_value = self._cosine_similarity(
                    gold_score_vectors[datapoint_index],
                    method_score_vector,
                )
                per_datapoint_similarities.append(similarity_value)
                feature_count = feature_counts_per_datapoint[datapoint_index]
                similarities_by_feature_count.setdefault(feature_count, []).append(similarity_value)

            similarity_results_by_method[method_name] = {
                "per_datapoint": per_datapoint_similarities,
                "mean_similarity": mean(per_datapoint_similarities) if per_datapoint_similarities else None,
                "by_feature_count": {
                    feature_count: mean(similarities)
                    for feature_count, similarities in similarities_by_feature_count.items()
                },
                "feature_count_summary": self._summarize_by_feature_count(similarities_by_feature_count),
            }
        return similarity_results_by_method




###################################################
# PLOTTING
###################################################
def plot_similarities(similarities):
    PLOTS_DIRECTORY.mkdir(exist_ok=True)
    method_names = list(similarities.keys())
    mean_similarities = [method_stats["mean_similarity"] for method_stats in similarities.values()]
    plt.bar(method_names, mean_similarities)
    plt.ylabel("Mean Similarity")
    plt.title("Attribution Similarity to standard Shapley value")
    plt.xticks(rotation=30, ha="right")
    for index, mean_value in enumerate(mean_similarities):
        plt.text(index, mean_value, f"{mean_value:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIRECTORY / "similarities_chart.png")
    plt.close()

    any_series_plotted = False
    for method_name, method_stats in similarities.items():
        average_similarity_by_feature_count = method_stats.get("by_feature_count", {})
        if not average_similarity_by_feature_count:
            continue
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
        plt.savefig(PLOTS_DIRECTORY / "similarities_by_feature_count.png")
        plt.close()


def plot_timing(timing_results, normalize: bool = False):
    PLOTS_DIRECTORY.mkdir(exist_ok=True)
    grouped_results = {}
    for name, results in timing_results.items():
        grouped = {}
        for result in results:
            grouped.setdefault(result["feature_count"], []).append(result["time"])
        grouped_results[name] = {feature_count: mean(times) for feature_count, times in grouped.items()}
    num_features_list = sorted({feature_count for grouped in grouped_results.values() for feature_count in grouped})
    baseline = max(value for grouped in grouped_results.values() for value in grouped.values()) if normalize else None
    for name, grouped in grouped_results.items():
        time_result_list = [grouped[x] / baseline for x in num_features_list if x in grouped] if normalize else [grouped[x] for x in num_features_list if x in grouped]
        feature_counts = [x for x in num_features_list if x in grouped]
        plt.plot(feature_counts, time_result_list, marker="o", label=name)
    plt.xlabel("Number of Features")
    plt.ylabel("Average Time (Normalized )" if normalize else "Average Time (s)")
    plt.title("Normalized Attribution Runtime by Number of Features" if normalize else "Attribution Runtime by Number of Features")
    plt.legend()
    plt.xticks(sorted(num_features_list))
    plt.tight_layout()
    plt.savefig(PLOTS_DIRECTORY / ("timing_chart_normalized.png" if normalize else "timing_chart.png"))
    plt.yscale("log")
    plt.title("Normalized Attribution Runtime by Number of Features (Log Scale)" if normalize else "Attribution Runtime by Number of Features (Log Scale)")
    plt.savefig(PLOTS_DIRECTORY / ("timing_chart_normalized_log.png" if normalize else "timing_chart_log.png"))
    plt.close()


def plot_similarity_convergence(similarities):
    PLOTS_DIRECTORY.mkdir(exist_ok=True)
    for method_name, method_stats in similarities.items():
        per_datapoint_similarities = method_stats["per_datapoint"]
        if not per_datapoint_similarities:
            continue

        running_mean_values = []
        cumulative_similarity_sum = 0.0
        for number_of_points, current_similarity_value in enumerate(per_datapoint_similarities, start=1):
            cumulative_similarity_sum += current_similarity_value
            running_mean_values.append(cumulative_similarity_sum / number_of_points)

        final_mean_similarity = running_mean_values[-1]
        absolute_differences_to_final = [abs(running_mean - final_mean_similarity) for running_mean in running_mean_values[:-1]]
        plt.plot(range(1, len(absolute_differences_to_final) + 1), absolute_differences_to_final, marker="o", label=method_name)

    plt.xlabel("Number of Data Points")
    plt.ylabel("Running mean and Final mean Diff")
    plt.title("Similarity Convergence to Gold Standard")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIRECTORY / "similarity_convergence.png")
    plt.close()
