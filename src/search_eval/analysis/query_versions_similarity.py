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

--all-tools scores every tool of the corpus (--requests) instead of the sample.

Usage:
    python -m search_eval.analysis.query_versions_similarity
    python -m search_eval.analysis.query_versions_similarity --all-tools \
        --set "v1 stored (Qwen)=data/results.jsonl" \
        --set "v2 revised (Qwen)=data/v2_rev_qwen/results.jsonl"
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
    load, load_model, nearest_neighbours_many, parse_sets, provenance, read_completions,
    record_embeddings, style_axis,
)

DEFAULT_SETS = [
    f"v1 stored (Qwen)={DATA / 'results.jsonl'}",
    f"v2 revised (GLM, README)={DATA / 'readme_v2_rev' / 'results.jsonl'}",
]


def auc(same: np.ndarray, control: np.ndarray) -> float:
    """P(same > control) + P(tie) / 2 over all pairs, from ranks (no pair matrix)."""
    control = np.sort(control)
    less = np.searchsorted(control, same, "left")
    ties = np.searchsorted(control, same, "right") - less
    return float((less + 0.5 * ties).sum() / (len(same) * len(control)))


def summarise(scores: dict, n_tools: int) -> tuple[dict, dict]:
    """
    scores: per query, "same" (n,), "neighbour" (n, k) and "random" (n, m), NaN
    where a random record was skipped for that query's tool; "tool" (n,) its tool.
    """
    same = scores["same"]
    rand = scores["random"]
    by_kind = {"same": same, "neighbour": scores["neighbour"].ravel(),
               "random": rand[~np.isnan(rand)]}
    s = {"n_tools_sampled": n_tools,
         "n_tools_with_queries": len(set(scores["tool"])),
         "n_queries": len(same)}
    for kind, values in by_kind.items():
        s[kind] = {"n_pairs": len(values), "median": float(np.median(values)),
                   "iqr": [float(x) for x in np.percentile(values, [25, 75])]}
    best = {"random": np.nanmax(rand, axis=1), "neighbour": scores["neighbour"].max(axis=1)}
    for kind in ("random", "neighbour"):
        s[kind]["auc_same_vs_this"] = auc(same, by_kind[kind])
        s[kind]["frac_queries_own_record_beats_all"] = float(np.mean(same > best[kind]))
    return by_kind, s


def score(tool_ids, queries, q_emb, matrix, index, neighbours, random_rows, skip):
    """Cosines of every query of these tools with its own, neighbour and random records."""
    rows = [(t, qi, q) for t in tool_ids for qi, q in enumerate(queries.get(t, []))]
    k = len(next(iter(neighbours.values()), []))
    out = {"tool": np.array([t for t, _, _ in rows], dtype=object),
           "same": np.empty(len(rows), np.float32),
           "neighbour": np.empty((len(rows), k), np.float32),
           "random": np.empty((len(rows), len(random_rows)), np.float32)}
    rand = matrix[random_rows]
    for start in range(0, len(rows), 20000):
        chunk = rows[start:start + 20000]
        sl = slice(start, start + len(chunk))
        q = np.stack([q_emb[q] for _, _, q in chunk])
        own = matrix[[index[t] for t, _, _ in chunk]]
        nb = matrix[[[index[r] for r in neighbours[t]] for t, _, _ in chunk]]
        out["same"][sl] = np.einsum("nd,nd->n", q, own)
        out["neighbour"][sl] = np.einsum("nd,nkd->nk", q, nb)
        out["random"][sl] = q @ rand.T
    for n, (t, _, _) in enumerate(rows):
        if t in skip:
            out["random"][n, skip[t]] = np.nan
    return rows, out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--set", dest="sets", action="append", metavar="LABEL=RESULTS",
                   help="a query set to compare (repeatable; default: stored v1 vs v2 revised)")
    p.add_argument("--sample", type=Path, default=DATA / "readme_v2_rev" / "sample.json")
    p.add_argument("--all-tools", action="store_true",
                   help="every tool in --requests instead of the --sample")
    p.add_argument("--requests", type=Path, default=DATA / "query_requests.jsonl",
                   help="all records' v1 contexts: the record text and the corpus")
    p.add_argument("--n-neighbours", type=int, default=5)
    p.add_argument("--n-random", type=int, default=500)
    p.add_argument("--seed", type=int, default=20261013)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out-dir", type=Path, default=EVAL)
    args = p.parse_args()
    sets = parse_sets(args.sets, DEFAULT_SETS)

    _, sources, names = load(DATA / "results.jsonl", args.requests)
    tool_ids = (sorted(sources) if args.all_tools
                else json.loads(args.sample.read_text())["tool_ids"])
    queries = {label: read_completions(Path(path), set(tool_ids)) for label, path in sets}

    model = load_model(args.model)
    ids, matrix = record_embeddings(model, args.model, sources, args.out_dir)
    index = {i: n for n, i in enumerate(ids)}

    rng = np.random.default_rng(args.seed)
    random_ids = [ids[n] for n in rng.choice(len(ids), args.n_random, replace=False)]
    random_rows = [index[r] for r in random_ids]
    with_queries = [t for t in tool_ids if any(queries[label].get(t) for label, _ in sets)]
    neighbours = nearest_neighbours_many(with_queries, ids, matrix, sources,
                                         args.n_neighbours)
    # Random records skipped for a tool: itself, and copies of its text.
    by_text: dict[str, list[int]] = {}
    for n, r in enumerate(random_ids):
        by_text.setdefault(sources[r][0].strip().lower(), []).append(n)
    skip = {}
    for t in with_queries:
        hit = set(by_text.get(sources[t][0].strip().lower(), []))
        hit |= {n for n, r in enumerate(random_ids) if r == t}
        if hit:
            skip[t] = sorted(hit)

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
            rows, scores = score(tool_ids, queries[label], q_emb, matrix, index,
                                 neighbours, random_rows, skip)
            for n, (t, qi, q) in enumerate(rows):
                recs = [("same", t, scores["same"][n])] + [
                    ("neighbour", r, c) for r, c in zip(neighbours[t], scores["neighbour"][n])]
                for kind, rec, cos in recs:
                    w.writerow([label, t, names[t], qi, q, kind, rec, names[rec],
                                sources[rec][1], f"{cos:.6f}"])
                # One row per query for the random controls, not hundreds.
                rand = scores["random"][n]
                for kind, cos in (("random_mean", np.nanmean(rand)),
                                  ("random_max", np.nanmax(rand))):
                    w.writerow([label, t, names[t], qi, q, kind, "", "", "", f"{cos:.6f}"])
            results[label], summaries[label] = summarise(scores, len(tool_ids))
            del scores

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
    lo = min(v.min() for r in results.values() for v in r.values())
    hi = max(v.max() for r in results.values() for v in r.values())
    bins = np.linspace(np.floor(lo * 20) / 20, np.ceil(hi * 20) / 20, 41)
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
