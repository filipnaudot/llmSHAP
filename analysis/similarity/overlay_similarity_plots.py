import json
import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from utils import AttributionComparator


def load_full_attr(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj["attribution_results"]

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
    parser.add_argument("--non-deterministic-file", help="checkpoint.json (non-det)")
    parser.add_argument("--deterministic-file", help="checkpoint.json (det)")
    parser.add_argument("-o", "--output", default="similarities_by_feature_count_overlay.png",help="Output PNG path")
    args = parser.parse_args()

    attr_non_det = load_full_attr(args.non_deterministic_file)
    attr_det = load_full_attr(args.deterministic_file)

    comparator = AttributionComparator()
    sim_non_det = comparator.compare(attr_non_det)
    sim_det = comparator.compare(attr_det)

    plot_overlay(sim_non_det, sim_det, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
