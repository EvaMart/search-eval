"""
How different are the synthetic queries of one tool from each other?

For each query set, embed every query of the 500-tool README experiment sample
and take the cosine similarity of every pair of queries *of the same tool*
("within"). As a baseline, the same number of pairs of queries from two
*different* tools ("between"). High within-tool similarity means the queries
are paraphrases of one another; lower means they cover more ways of asking for
the tool. Near-duplicates are pairs at or above --dup-threshold.

Also reports, per tool, the mean within-tool similarity, and compares the sets
on the tools both have queries for (paired).

Defaults: stored v1 (Qwen) vs revised v2 (GLM, README), as in
query_versions_similarity. Writes a per-tool CSV, a JSON summary and a plot.

Usage:
    python -m search_eval.analysis.query_diversity
    python -m search_eval.analysis.query_diversity --set "v1 GLM=data/v1_glm/results.jsonl" ...
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from search_eval.analysis.query_versions_similarity import DEFAULT_SETS
from search_eval.common import (
    BLUE, DATA, DEFAULT_MODEL, EVAL, GREEN, INK, INK2, MUTED, ORANGE, PURPLE, SURFACE,
    load_model, parse_sets, provenance, read_completions, style_axis,
)


def stats(v: np.ndarray, dup: float) -> dict:
    return {"n_pairs": int(len(v)), "median": float(np.median(v)),
            "mean": float(np.mean(v)),
            "iqr": [float(x) for x in np.percentile(v, [25, 75])],
            f"frac_ge_{dup}": float(np.mean(v >= dup))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--set", dest="sets", action="append", metavar="LABEL=RESULTS")
    p.add_argument("--sample", type=Path, default=DATA / "readme_v2_rev" / "sample.json")
    p.add_argument("--dup-threshold", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=20261013)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out-dir", type=Path, default=EVAL)
    args = p.parse_args()
    sets = parse_sets(args.sets, DEFAULT_SETS)
    tool_ids = json.loads(args.sample.read_text())["tool_ids"]
    queries = {label: read_completions(Path(path), set(tool_ids)) for label, path in sets}

    model = load_model(args.model)
    all_q = sorted({q for qs in queries.values() for ql in qs.values() for q in ql})
    q_emb = dict(zip(all_q, model.encode(all_q, normalize_embeddings=True,
                                         batch_size=64, show_progress_bar=True)))

    rng = np.random.default_rng(args.seed)
    dup = args.dup_threshold
    within, between, per_tool, summary = {}, {}, {}, {}
    for label, _ in sets:
        qs = {t: ql for t, ql in queries[label].items() if len(ql) >= 2}
        w, tool_mean, tool_max = [], {}, []
        for t, ql in qs.items():
            e = np.stack([q_emb[q] for q in ql])
            sims = [float(e[i] @ e[j]) for i, j in combinations(range(len(ql)), 2)]
            w += sims
            tool_mean[t] = (float(np.mean(sims)), len(ql))
            tool_max.append(max(sims))
        # Between-tool baseline: as many random pairs from two different tools.
        flat = [(t, q) for t, ql in qs.items() for q in ql]
        b = []
        while len(b) < len(w):
            i, j = rng.integers(len(flat), size=2)
            if flat[i][0] != flat[j][0]:
                b.append(float(q_emb[flat[i][1]] @ q_emb[flat[j][1]]))
        within[label], between[label], per_tool[label] = np.array(w), np.array(b), tool_mean
        means = np.array([m for m, _ in tool_mean.values()])
        summary[label] = {
            "n_tools": len(qs), "n_queries": sum(len(ql) for ql in qs.values()),
            "queries_per_tool_median": float(np.median([len(ql) for ql in qs.values()])),
            "within": stats(within[label], dup), "between": stats(between[label], dup),
            "per_tool_mean": {"median": float(np.median(means)),
                              "iqr": [float(x) for x in np.percentile(means, [25, 75])]},
            "frac_tools_with_a_near_duplicate_pair": float(np.mean(np.array(tool_max) >= dup)),
        }
    labels = [label for label, _ in sets]
    if len(labels) == 2:
        a, b_ = labels
        common = sorted(set(per_tool[a]) & set(per_tool[b_]))
        diff = np.array([per_tool[b_][t][0] - per_tool[a][t][0] for t in common])
        summary["paired"] = {"sets": [a, b_], "n_tools": len(common),
                             "median_diff_per_tool_mean (second - first)": float(np.median(diff)),
                             "frac_tools_second_more_diverse": float(np.mean(diff < 0))}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"query_diversity_n{len(tool_ids)}_seed{args.seed}"
    with open(f"{stem}.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["set", "tool_id", "n_queries", "mean_within_cosine"])
        for label in labels:
            for t, (m, n) in per_tool[label].items():
                wr.writerow([label, t, n, f"{m:.6f}"])
    out = {"model": args.model, "seed": args.seed, "sets": dict(sets), **provenance(),
           **summary}
    Path(f"{stem}.json").write_text(json.dumps(out, indent=2) + "\n")
    plot(within, between, summary, args, Path(f"{stem}.png"))
    print(json.dumps(summary, indent=2))
    print(f"wrote {stem}.csv / .json / .png")


def plot(within: dict, between: dict, s: dict, args, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = [BLUE, ORANGE, GREEN, PURPLE]
    labels = list(within)
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axis(ax)
    allv = np.concatenate(list(within.values()) + list(between.values()))
    bins = np.linspace(np.floor(allv.min() * 20) / 20, 1.0, 41)
    for label, color in zip(labels, colors):
        ax.hist(within[label], bins=bins, density=True, color=color, alpha=0.25,
                edgecolor=SURFACE, linewidth=1)
        ax.hist(within[label], bins=bins, density=True, histtype="step", color=color,
                linewidth=2, label=f"{label}: same tool (median {np.median(within[label]):.2f})")
        ax.hist(between[label], bins=bins, density=True, histtype="step", color=color,
                linewidth=1.5, linestyle=(0, (2, 2)),
                label=f"{label}: different tools (median {np.median(between[label]):.2f})")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    for label, color in zip(labels, colors):
        for v in (within[label], between[label]):
            ax.axvline(np.median(v), color=color, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.axvline(args.dup_threshold, color=MUTED, linewidth=1, linestyle=":")
    ax.annotate(f"near-duplicate ≥ {args.dup_threshold}", xy=(args.dup_threshold, ax.get_ylim()[1]),
                xytext=(4, -12), textcoords="offset points", color=INK2, fontsize=8)
    ax.set_xlabel("Cosine similarity between two queries "
                  f"({args.model.split('/')[-1]})", color=INK2, fontsize=9.5)
    ax.set_ylabel("Density", color=INK2, fontsize=9.5)
    leg = ax.legend(frameon=False, fontsize=8.5, loc="lower left",
                    bbox_to_anchor=(0, 1.0), ncol=1, borderaxespad=0.2)
    for text in leg.get_texts():
        text.set_color(INK)
    fig.suptitle("How similar are the queries of one tool to each other?",
                 x=0.06, ha="left", color=INK, fontsize=11.5, fontweight="semibold")
    sub = []
    for label in labels:
        d = s[label]
        sub.append(f"{label}: {d['n_queries']} queries, {d['n_tools']} tools · "
                   f"pairs ≥ {args.dup_threshold}: {d['within'][f'frac_ge_{args.dup_threshold}']:.1%} · "
                   f"tools with one: {d['frac_tools_with_a_near_duplicate_pair']:.0%}")
    ax.set_title("\n".join(sub), loc="left", color=INK2, fontsize=8.5,
                 pad=14 + 13 * 2 * len(labels), linespacing=1.5)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
