import math
from statistics import mean


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

    def _cosine_similarity(self, vector_a, vector_b):
        dot_product = sum(value_a * value_b for value_a, value_b in zip(vector_a, vector_b))
        magnitude_a = math.sqrt(sum(value * value for value in vector_a))
        magnitude_b = math.sqrt(sum(value * value for value in vector_b))
        return 0.0 if magnitude_a == 0.0 or magnitude_b == 0.0 else dot_product / (magnitude_a * magnitude_b)

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

        similarity_results_by_function = {}
        for method_name, method_entries in attribution_data.items():
            if method_name == self.gold_method_name:
                continue

            per_datapoint_similarities = []
            similarities_grouped_by_feature_count = {}
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
                similarities_grouped_by_feature_count.setdefault(feature_count, []).append(similarity_value)

            similarity_results_by_function[method_name] = {
                "per_datapoint": per_datapoint_similarities,
                "mean_similarity": mean(per_datapoint_similarities) if per_datapoint_similarities else None,
                "by_feature_count": {
                    feature_count: mean(similarities)
                    for feature_count, similarities in similarities_grouped_by_feature_count.items()
                },
            }
        return similarity_results_by_function
