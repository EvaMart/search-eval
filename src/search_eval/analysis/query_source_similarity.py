"""
Do synthetic queries stay close to the record they were written from?

For a random sample of tools, embed each stored synthetic query and compare it
(cosine similarity) with

- "same":      its own tool's source text,
- "random":    the source text of N random *other* tools (easy negative controls),
- "neighbour": the source text of the tool's N nearest other records, by
  record-to-record embedding similarity (hard negative controls: same topic,
  different tool -- what search actually has to tell apart).

If the queries are grounded in their records, "same" should sit clearly above
"random"; how far it sits above "neighbour" says how *specific* they are.
Records whose text is identical to the tool's are not neighbours (they are
copies, not other tools). Writes a per-pair CSV, a JSON summary and a histogram
of the three distributions.

The source text is the description block of the context the LLM saw (the tool's
description, or its Galaxy help text when it has none -- see `_descriptions` in
the pipeline's synthetic_queries.py). Records with no description fall back to
their EDAM terms. The tool name is left out so a shared name cannot inflate the
score (stored queries never contain it anyway: `parse_queries` drops those).

Queries are the *stored* ones: each completion goes through `parse_queries`,
exactly as `rsetl synthetic-queries import` does.

The defaults reproduce the 50-tool sample reviewed by hand on 2026-10-09
(seed 20261009: 45 tools with queries + 5 the LLM answered with none, which
contribute no queries here).

Neighbours need every record embedded; that matrix is cached in --out-dir
(record_embeddings_<model>_<hash>.npy) and reused while the inputs are unchanged.

Usage:
    python -m search_eval.analysis.query_source_similarity
    python -m search_eval.analysis.query_source_similarity --n-tools 500 --seed 7
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

from application.services.semantic_search.synthetic_queries import parse_queries
from search_eval.common import (
    BLUE, DATA, DEFAULT_MODEL, EVAL, GREEN, INK, INK2, ORANGE, SURFACE,
    load, load_model, nearest_neighbours, provenance, record_embeddings, style_axis,
)


def sample_tools(completions: dict, n_tools: int, n_empty: int, seed: int) -> list[str]:
    """
    Tools with queries plus tools the LLM answered empty, in file order.

    Selection is on the raw completion (as for the manual review), so the
    default seed gives back the same 50 tools.
    """
    with_queries, empty = [], []
    for tool_id, r in completions.items():
        try:
            raw = json.loads(r["completion"]).get("queries")
        except (json.JSONDecodeError, AttributeError):
            continue
        (with_queries if raw else empty).append(tool_id)
    random.seed(seed)
    return random.sample(with_queries, n_tools - n_empty) + random.sample(
        empty, min(n_empty, len(empty))
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results", type=Path, default=DATA / "results.jsonl")
    p.add_argument("--requests", type=Path, default=DATA / "query_requests.jsonl")
    p.add_argument("--n-tools", type=int, default=50)
    p.add_argument("--n-empty", type=int, default=5,
                   help="how many of the sampled tools are LLM-empty answers")
    p.add_argument("--n-controls", type=int, default=5,
                   help="random other records compared with each query")
    p.add_argument("--n-neighbours", type=int, default=5,
                   help="nearest-neighbour records compared with each query")
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out-dir", type=Path, default=EVAL)
    args = p.parse_args()

    completions, sources, names = load(args.results, args.requests)
    tool_ids = sample_tools(completions, args.n_tools, args.n_empty, args.seed)

    model = load_model(args.model)
    ids, matrix = record_embeddings(model, args.model, sources, args.out_dir)
    r_emb = dict(zip(ids, matrix))

    # Every query and the records it is compared with.
    rng = np.random.default_rng(args.seed)
    rows = []  # (tool_id, tool_name, query_idx, query, kind, record_id)
    for tool_id in tool_ids:
        r = completions[tool_id]
        queries = parse_queries(r["completion"], r["names"]) or []
        neighbours = nearest_neighbours(tool_id, ids, matrix, sources, args.n_neighbours)
        for qi, query in enumerate(queries):
            rows.append((tool_id, r["tool_name"], qi, query, "same", tool_id))
            controls = set()
            while len(controls) < args.n_controls:
                other = ids[rng.integers(len(ids))]
                if other != tool_id:
                    controls.add(other)
            for other in sorted(controls):
                rows.append((tool_id, r["tool_name"], qi, query, "random", other))
            for other in neighbours:
                rows.append((tool_id, r["tool_name"], qi, query, "neighbour", other))

    queries = sorted({row[3] for row in rows})
    q_emb = dict(zip(queries, model.encode(queries, normalize_embeddings=True,
                                           batch_size=64, show_progress_bar=True)))

    stem = args.out_dir / f"query_source_similarity_n{args.n_tools}_seed{args.seed}"
    scored = []
    with open(f"{stem}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tool_id", "tool_name", "query_idx", "query", "kind",
                    "record_id", "record_name", "source_kind", "cosine"])
        for tool_id, name, qi, query, kind, rec in rows:
            cos = float(np.dot(q_emb[query], r_emb[rec]))
            scored.append((tool_id, qi, kind, cos))
            w.writerow([tool_id, name, qi, query, kind, rec, names[rec],
                        sources[rec][1], f"{cos:.6f}"])

    by_kind = {k: np.array([s[3] for s in scored if s[2] == k])
               for k in ("same", "random", "neighbour")}
    by_query: dict[tuple, dict] = {}
    for tool_id, qi, kind, cos in scored:
        d = by_query.setdefault((tool_id, qi), {"same": None, "random": [], "neighbour": []})
        if kind == "same":
            d["same"] = cos
        else:
            d[kind].append(cos)
    same = by_kind["same"]
    summary = {
        "model": args.model, "seed": args.seed, "n_tools_sampled": len(tool_ids),
        "n_tools_with_queries": len({k[0] for k in by_query}),
        "n_queries": len(same), **provenance(),
    }
    for kind, values in by_kind.items():
        summary[kind] = {
            "n_pairs": len(values),
            "median": float(np.median(values)),
            "iqr": [float(x) for x in np.percentile(values, [25, 75])],
        }
    for kind in ("random", "neighbour"):
        control = by_kind[kind]
        # P(same > control) over all pairs: the ROC AUC of cosine as a "same" detector.
        summary[kind]["auc_same_vs_this"] = float(
            (same[:, None] > control[None, :]).mean()
            + 0.5 * (same[:, None] == control[None, :]).mean())
        summary[kind]["frac_queries_own_record_beats_all"] = float(np.mean(
            [d["same"] > max(d[kind]) for d in by_query.values()]))
    Path(f"{stem}.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(by_kind, summary, Path(f"{stem}.png"))
    print(json.dumps(summary, indent=2))
    print(f"wrote {stem}.csv / .json / .png")


def plot(by_kind: dict, s: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = (  # back to front
        ("random", ORANGE, "Random: 5 random other records"),
        ("neighbour", GREEN, "Neighbour: the tool's 5 nearest other records"),
        ("same", BLUE, "Same: the query's own record"),
    )

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style_axis(ax)
    allv = np.concatenate(list(by_kind.values()))
    bins = np.linspace(np.floor(allv.min() * 20) / 20, np.ceil(allv.max() * 20) / 20, 41)
    for kind, color, label in series:
        values = by_kind[kind]
        ax.hist(values, bins=bins, density=True, color=color, alpha=0.30,
                edgecolor=SURFACE, linewidth=1)
        ax.hist(values, bins=bins, density=True, histtype="step", color=color,
                linewidth=2, label=f"{label} (n={len(values)})")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    # Medians can sit close together; stagger their labels so they never overlap.
    for row, (kind, color, _) in enumerate(series):
        med = np.median(by_kind[kind])
        ax.axvline(med, color=color, linewidth=1.5, linestyle=(0, (4, 3)))
        ax.annotate(f"median {med:.2f}", xy=(med, ax.get_ylim()[1]),
                    xytext=(4, -12 - 13 * row), textcoords="offset points",
                    color=INK2, fontsize=8.5)

    ax.set_xlabel("Cosine similarity, query vs record text "
                  f"({s['model'].split('/')[-1]})", color=INK2, fontsize=9.5)
    ax.set_ylabel("Density", color=INK2, fontsize=9.5)
    handles, labels = ax.get_legend_handles_labels()
    order = [2, 1, 0]  # same, neighbour, random
    leg = ax.legend([handles[i] for i in order], [labels[i] for i in order],
                    frameon=False, fontsize=8.5, loc="lower left",
                    bbox_to_anchor=(0, 1.0), ncol=1, borderaxespad=0.2)
    for text in leg.get_texts():
        text.set_color(INK)
    fig.suptitle("Query vs its own record, its nearest neighbours, and random records",
                 x=0.06, ha="left", color=INK, fontsize=11.5, fontweight="semibold")
    rnd, nb = s["random"], s["neighbour"]
    ax.set_title(
        f"{s['n_queries']} queries from {s['n_tools_with_queries']} sampled tools · seed {s['seed']}\n"
        f"Same vs random: AUC {rnd['auc_same_vs_this']:.3f}, own record beats all 5 for "
        f"{rnd['frac_queries_own_record_beats_all']:.0%} of queries\n"
        f"Same vs neighbour: AUC {nb['auc_same_vs_this']:.3f}, own record beats all 5 for "
        f"{nb['frac_queries_own_record_beats_all']:.0%} of queries",
        loc="left", color=INK2, fontsize=8.5, pad=50, linespacing=1.5)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
