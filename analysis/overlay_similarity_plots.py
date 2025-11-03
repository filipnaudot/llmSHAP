import json
import math
from statistics import mean
import argparse
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

GOLD_METHOD = "Shapley value"


def load_full_attr(path):
    with open(path, "r", encoding="utf-8") as f:
        line = f.read().strip()
    obj = json.loads(line)
    return obj["attribution_results"]


def extract_score_vector(attr_mapping, ordered_keys):
    vec = []
    for k in ordered_keys:
        v = attr_mapping.get(k, {})
        if isinstance(v, dict):
            vec.append(float(v.get("score", 0.0)))
        else:
            try:
                vec.append(float(v))
            except Exception:
                vec.append(0.0)
    return vec


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    ma = math.sqrt(sum(x * x for x in a))
    mb = math.sqrt(sum(x * x for x in b))
    if ma == 0.0 or mb == 0.0:
        return 0.0
    return dot / (ma * mb)


def compare_to_gold(attribution_data, gold_name=GOLD_METHOD):
    gold_entries = attribution_data[gold_name]
    n = len(gold_entries)

    ordered_feature_keys_per_dp = [
        list(gold_entries[i]["attribution"].keys()) for i in range(n)
    ]
    gold_score_vectors = [
        extract_score_vector(gold_entries[i]["attribution"], ordered_feature_keys_per_dp[i])
        for i in range(n)
    ]
    feature_counts_per_dp = [gold_entries[i]["feature_count"] for i in range(n)]

    results = {}
    for method_name, method_entries in attribution_data.items():
        if method_name == gold_name:
            continue

        per_dp_sims = []
        grouped = {}  # feature_count -> list[sim]

        for i in range(n):
            method_vec = extract_score_vector(
                method_entries[i]["attribution"],
                ordered_feature_keys_per_dp[i],
            )
            sim = cosine_similarity(gold_score_vectors[i], method_vec)
            per_dp_sims.append(sim)

            fc = feature_counts_per_dp[i]
            grouped.setdefault(fc, []).append(sim)

        by_fc = {fc: mean(vals) for fc, vals in grouped.items()}
        results[method_name] = {
            "per_datapoint": per_dp_sims,
            "mean_similarity": mean(per_dp_sims) if per_dp_sims else None,
            "by_feature_count": by_fc,
        }

    return results

def plot_overlay(sim_nondet, sim_det, out_path):
    method_names = list(sim_nondet.keys())
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots()

    for idx, method in enumerate(method_names):
        base_color = color_cycle[idx % len(color_cycle)]

        # non-deterministic (solid)
        non_by_fc = sim_nondet[method].get("by_feature_count", {})
        if non_by_fc:
            xs = sorted(non_by_fc.keys())
            ys = [non_by_fc[x] for x in xs]
            ax.plot(xs, ys, marker="o", label=method, color=base_color)

        # deterministic (dashed) — no label, same color
        det_by_fc = sim_det.get(method, {}).get("by_feature_count", {})
        if det_by_fc:
            xs_d = sorted(det_by_fc.keys())
            ys_d = [det_by_fc[x] for x in xs_d]
            ax.plot(xs_d, ys_d, marker="o", linestyle="--", color=base_color)

    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Average Similarity to gold")
    ax.set_title("Similarity vs. Feature Count")

    method_legend = ax.legend(title="Methods", loc="lower left")

    style_handles = [
        Line2D([0], [0], color="black", linestyle="-", label="non-deterministic"),
        Line2D([0], [0], color="black", linestyle="--", label="deterministic"),
    ]
    ax.add_artist(method_legend)
    ax.legend(handles=style_handles, title="Run type", loc="center left", bbox_to_anchor=(0.00, 0.35))

    all_counts = set()
    for m in sim_nondet.values():
        all_counts.update(m["by_feature_count"].keys())
    if all_counts:
        ax.set_xticks(sorted(all_counts))

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Overlay similarity-per-feature-count plots from two attribution runs."
    )
    parser.add_argument("non_deterministic_file", help="full_attribution_log.jsonl (non-det)")
    parser.add_argument("deterministic_file", help="full_attribution_log.jsonl (det)")
    parser.add_argument("-o", "--output", default="similarities_by_feature_count_overlay.png",help="Output PNG path")
    args = parser.parse_args()

    attr_non_det = load_full_attr(args.non_deterministic_file)
    attr_det = load_full_attr(args.deterministic_file)

    sim_non_det = compare_to_gold(attr_non_det)
    sim_det = compare_to_gold(attr_det)

    plot_overlay(sim_non_det, sim_det, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
