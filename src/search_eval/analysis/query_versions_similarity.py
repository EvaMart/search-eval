"""
Do v1 and v2 (README) synthetic queries stay equally close to their records?

The query_source_similarity analysis, run side by side for two query sets on
the 500-tool README experiment sample (experiments/readme_queries). Each query
is compared (cosine) with

- "same":      its own tool's description,
- "neighbour": the descriptions of the tool's N nearest other records,
- "random":    the descriptions of one fixed set of M random records, shared by
  every query of both versions (records with the tool's own text are skipped).

The record text is the description block of the *v1* context for both versions
(`source_text`), so the README a v2 query may have drawn on is not part of what
it is scored against: the question is whether the README pulls queries away
from what the record says.

Defaults: stored v1 (Qwen, data/results.jsonl) vs revised v2 (GLM,
data/readme_v2_rev). Every completion goes through `parse_queries`. The CSV has
one row per query and same/neighbour record, and one random_mean and one
random_max row per query (not one per random record: that is ~500 MB).

Usage:
    python -m search_eval.analysis.query_versions_similarity
    python -m search_eval.analysis.query_versions_similarity \
        --set "v1 GLM=data/v1_glm/results.jsonl" \
        --set "v2 revised=data/readme_v2_rev/results.jsonl"
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from search_eval.common import (
    BLUE, DATA, DEFAULT_MODEL, EVAL, GREEN, INK, INK2, ORANGE, SURFACE,
    load, load_model, nearest_neighbours, parse_sets, provenance, read_completions,
    record_embeddings, style_axis,
)

KINDS = ("same", "neighbour", "random")
DEFAULT_SETS = [
    f"v1 stored (Qwen)={DATA / 'results.jsonl'}",
    f"v2 revised (GLM, README)={DATA / 'readme_v2_rev' / 'results.jsonl'}",
]


def summarise(scored: list[tuple], n_tools: int) -> tuple[dict, dict]:
    """scored: (tool_id, query_idx, kind, cosine)."""
    by_kind = {k: np.array([s[3] for s in scored if s[2] == k]) for k in KINDS}
    by_query: dict[tuple, dict] = {}
    for tool_id, qi, kind, cos in scored:
        d = by_query.setdefault((tool_id, qi), {"same": None, "random": [], "neighbour": []})
        if kind == "same":
            d["same"] = cos
        else:
            d[kind].append(cos)
    same = by_kind["same"]
    s = {"n_tools_sampled": n_tools,
         "n_tools_with_queries": len({k[0] for k in by_query}),
         "n_queries": len(same)}
    for kind, values in by_kind.items():
        s[kind] = {"n_pairs": len(values), "median": float(np.median(values)),
                   "iqr": [float(x) for x in np.percentile(values, [25, 75])]}
    for kind in ("random", "neighbour"):
        control = by_kind[kind]
        s[kind]["auc_same_vs_this"] = float(
            (same[:, None] > control[None, :]).mean()
            + 0.5 * (same[:, None] == control[None, :]).mean())
        s[kind]["frac_queries_own_record_beats_all"] = float(np.mean(
            [d["same"] > max(d[kind]) for d in by_query.values()]))
    return by_kind, s


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--set", dest="sets", action="append", metavar="LABEL=RESULTS",
                   help="a query set to compare (repeatable; default: stored v1 vs v2 revised)")
    p.add_argument("--sample", type=Path, default=DATA / "readme_v2_rev" / "sample.json")
    p.add_argument("--requests", type=Path, default=DATA / "query_requests.jsonl",
                   help="all records' v1 contexts: the record text and the corpus")
    p.add_argument("--n-neighbours", type=int, default=5)
    p.add_argument("--n-random", type=int, default=500)
    p.add_argument("--seed", type=int, default=20261013)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out-dir", type=Path, default=EVAL)
    args = p.parse_args()
    sets = parse_sets(args.sets, DEFAULT_SETS)

    tool_ids = json.loads(args.sample.read_text())["tool_ids"]
    _, sources, names = load(DATA / "results.jsonl", args.requests)
    queries = {label: read_completions(Path(path), set(tool_ids)) for label, path in sets}

    model = load_model(args.model)
    ids, matrix = record_embeddings(model, args.model, sources, args.out_dir)
    index = {i: n for n, i in enumerate(ids)}

    rng = np.random.default_rng(args.seed)
    random_ids = [ids[n] for n in rng.choice(len(ids), args.n_random, replace=False)]
    neighbours = {t: nearest_neighbours(t, ids, matrix, sources, args.n_neighbours)
                  for t in tool_ids}

    all_q = sorted({q for qs in queries.values() for ql in qs.values() for q in ql})
    q_emb = dict(zip(all_q, model.encode(all_q, normalize_embeddings=True,
                                         batch_size=64, show_progress_bar=True)))

    stem = args.out_dir / f"query_versions_similarity_n{len(tool_ids)}_seed{args.seed}"
    results, summaries = {}, {}
    with open(f"{stem}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "tool_id", "tool_name", "query_idx", "query", "kind",
                    "record_id", "record_name", "source_kind", "cosine"])
        for label, _ in sets:
            scored = []
            for t in tool_ids:
                own = sources[t][0].strip().lower()
                controls = [r for r in random_ids
                            if r != t and sources[r][0].strip().lower() != own]
                pairs = ([("same", t)] + [("neighbour", r) for r in neighbours[t]]
                         + [("random", r) for r in controls])
                recs = np.stack([matrix[index[r]] for _, r in pairs])
                for qi, q in enumerate(queries[label].get(t, [])):
                    cosines = recs @ q_emb[q]
                    for (kind, rec), cos in zip(pairs, cosines):
                        scored.append((t, qi, kind, float(cos)))
                        if kind != "random":
                            w.writerow([label, t, names[t], qi, q, kind, rec, names[rec],
                                        sources[rec][1], f"{cos:.6f}"])
                    # One row per query for the random controls, not hundreds.
                    rand = cosines[1 + len(neighbours[t]):]
                    for kind, cos in (("random_mean", rand.mean()), ("random_max", rand.max())):
                        w.writerow([label, t, names[t], qi, q, kind, "", "", "", f"{cos:.6f}"])
            results[label], summaries[label] = summarise(scored, len(tool_ids))

    summary = {"model": args.model, "seed": args.seed, "n_random": args.n_random,
               "n_neighbours": args.n_neighbours, "sets": dict(sets), **provenance(),
               **summaries}
    Path(f"{stem}.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    plot(results, summary, args, Path(f"{stem}.png"))
    print(json.dumps(summaries, indent=2))
    print(f"wrote {stem}.csv / .json / .png")


def plot(results: dict, summary: dict, args, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = (  # back to front
        ("random", ORANGE, f"Random: {args.n_random} fixed random records"),
        ("neighbour", GREEN, f"Neighbour: the tool's {args.n_neighbours} nearest other records"),
        ("same", BLUE, "Same: the query's own record"),
    )
    labels = list(results)
    fig, axes = plt.subplots(len(labels), 1, figsize=(8, 3.6 * len(labels) + 1.2),
                             dpi=200, sharex=True, squeeze=False)
    axes = axes[:, 0]
    fig.patch.set_facecolor(SURFACE)
    allv = np.concatenate([v for r in results.values() for v in r.values()])
    bins = np.linspace(np.floor(allv.min() * 20) / 20, np.ceil(allv.max() * 20) / 20, 41)
    ymax = 0
    for ax, label in zip(axes, labels):
        style_axis(ax)
        for kind, color, name in series:
            v = results[label][kind]
            ax.hist(v, bins=bins, density=True, color=color, alpha=0.30,
                    edgecolor=SURFACE, linewidth=1)
            ax.hist(v, bins=bins, density=True, histtype="step", color=color,
                    linewidth=2, label=f"{name}")
        ymax = max(ymax, ax.get_ylim()[1])
    for ax, label in zip(axes, labels):
        s = summary[label]
        ax.set_ylim(0, ymax * 1.15)
        for row, (kind, color, _) in enumerate(series):
            med = np.median(results[label][kind])
            ax.axvline(med, color=color, linewidth=1.5, linestyle=(0, (4, 3)))
            ax.annotate(f"median {med:.2f}", xy=(med, ax.get_ylim()[1]),
                        xytext=(4, -12 - 13 * row), textcoords="offset points",
                        color=INK2, fontsize=8.5)
        rnd, nb = s["random"], s["neighbour"]
        ax.set_title(
            f"{label} · {s['n_queries']} queries from {s['n_tools_with_queries']} tools\n"
            f"vs random: AUC {rnd['auc_same_vs_this']:.3f}, own record beats all "
            f"{rnd['frac_queries_own_record_beats_all']:.0%} · "
            f"vs neighbour: AUC {nb['auc_same_vs_this']:.3f}, own record beats all "
            f"{nb['frac_queries_own_record_beats_all']:.0%}",
            loc="left", color=INK, fontsize=9, linespacing=1.5)
        ax.set_ylabel("Density", color=INK2, fontsize=9.5)
    axes[-1].set_xlabel("Cosine similarity, query vs record description "
                        f"({summary['model'].split('/')[-1]})", color=INK2, fontsize=9.5)
    handles, hl = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles[::-1], hl[::-1], frameon=False, fontsize=8.5,
                     loc="upper left", bbox_to_anchor=(0.05, 0.955), ncol=1)
    for text in leg.get_texts():
        text.set_color(INK)
    fig.suptitle("Query vs its own record, its nearest neighbours, and random records",
                 x=0.06, ha="left", color=INK, fontsize=11.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_figheight()))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
