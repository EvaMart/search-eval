"""
How well does each retriever, the RRF hybrid, and the Observatory's current search
find the right tool for the evaluation queries (search_eval.eval_queries)?

Each retriever is run once per query and its hit lists are cached
(--runs, resumed on a rerun); the hybrids are fused from those same lists
(hybrid.fuse), so they cost no extra queries. Systems:

    dense_description, bm25_description, bm25_names, dense_queries
                        the four retrievers alone (top 100 each)
    hybrid_3            RRF of dense_description, bm25_description, dense_queries
    hybrid_4            RRF of all four
    hybrid_4_boost      hybrid_4 with exact/near name matches weighted (name_boosts)
    hybrid_4_adaptive   hybrid_4_boost, and bm25_names dropped (weight 0) for a
                        query none of its hits is a name match of
    observatory_api     the current search (top 10: one API page)

Scores, per query type and overall, on the top 10 (all the API page gives):
Recall@1, @5, @10 and MRR@10. A hit is any of the query's `relevant` tools (the
tool and its same-name records); `--strict` counts only the tool itself. Task
queries that are near-copies of an indexed synthetic query (own_query_sim >=
0.95) are also scored apart ("task, no near-copies"), since they favour
dense_queries.

Writes to data/eval/: a CSV (system x group x metric), a JSON summary (with the
index counts and date: results move when the index is updated) and a heatmap.

Usage:
    python -m search_eval.analysis.retriever_comparison
    python -m search_eval.analysis.retriever_comparison --strict
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from search_eval.common import EVAL, INK, INK2, MUTED, SURFACE
from search_eval.eval_queries import NEAR_COPY, OUT_DIR as QUERIES_DIR
from search_eval.retrieval import hybrid, observatory_api
from search_eval.retrieval.index import COLLECTION, NAMES_COLLECTION, Hit, client, prime

K = 10
DEPTH = hybrid.DEPTH
API = "observatory_api"
SINGLE = list(hybrid.RETRIEVERS)
HYBRID_3 = ["dense_description", "bm25_description", "dense_queries"]
SYSTEMS = SINGLE + ["hybrid_3", "hybrid_4", "hybrid_4_boost", "hybrid_4_adaptive", API]
GROUPS = ["all", "name", "name_exact", "name_variant", "name_typo", "task",
          "task_keyword", "task_description", "task_question", "task, no near-copies"]
PLOT_GROUPS = ["name_exact", "name_variant", "name_typo", "task_keyword",
               "task_description", "task_question"]


def load_runs(path: Path) -> dict[tuple[str, str], list[str]]:
    """(retriever, query) -> tool ids, best first, from earlier runs."""
    runs = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            runs[(r["retriever"], r["query"])] = r["tool_ids"]
    return runs


def run_all(queries: list[str], path: Path, workers: int) -> dict[tuple[str, str], list[str]]:
    """Every retriever on every query, appending new hit lists to `path`."""
    runs = load_runs(path)
    jobs = [(r, q) for r in SINGLE + [API] for q in queries if (r, q) not in runs]
    logging.info("%d runs cached, %d to do", len(runs), len(jobs))
    if not jobs:
        return runs
    prime([q for r, q in jobs if r.startswith("dense")])

    def one(job):
        retriever, query = job
        if retriever == API:
            return job, [h.tool_id for h in observatory_api.search(query, K)]
        return job, [h.tool_id for h in hybrid.RETRIEVERS[retriever](query, DEPTH)]

    # The API is a public service: one request at a time. Qdrant gets `workers`.
    api_jobs = [j for j in jobs if j[0] == API]
    index_jobs = [j for j in jobs if j[0] != API]
    with path.open("a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=workers) as pool, \
            ThreadPoolExecutor(max_workers=1) as api_pool:
        futures = [pool.submit(one, j) for j in index_jobs] + \
                  [api_pool.submit(one, j) for j in api_jobs]
        for n, future in enumerate(futures, 1):
            (retriever, query), ids = future.result()
            runs[(retriever, query)] = ids
            fh.write(json.dumps({"retriever": retriever, "query": query, "tool_ids": ids},
                                ensure_ascii=False) + "\n")
            if n % 500 == 0 or n == len(futures):
                fh.flush()
                logging.info("%d / %d runs", n, len(futures))
    return runs


def system_ranking(system: str, query: str, runs) -> list[str]:
    """The system's tool ids for the query, best first."""
    if system in SINGLE or system == API:
        return runs[(system, query)]
    members = HYBRID_3 if system == "hybrid_3" else SINGLE
    rankings = {r: [Hit(t, "", 0.0, "", "") for t in runs[(r, query)]] for r in members}
    weights, boosts = {}, {}
    if system in ("hybrid_4_boost", "hybrid_4_adaptive"):
        # name_match needs the names: re-read them from the cached names run.
        rankings[hybrid.NAMES] = names_hits(query, runs)
        weights, boosts = hybrid.name_weights(
            query, rankings, names_unmatched=0.0 if system == "hybrid_4_adaptive" else 1.0)
    return [h.tool_id for h in hybrid.fuse(rankings, K, weights=weights, hit_weights=boosts)]


_NAMES_TEXT: dict[str, str] = {}


def names_hits(query: str, runs) -> list[Hit]:
    """The names retriever's cached hits, with each tool's names for name_match."""
    ids = runs[(hybrid.NAMES, query)]
    missing = [t for t in ids if t not in _NAMES_TEXT]
    if missing:
        from qdrant_client import models
        flt = models.Filter(must=[models.FieldCondition(key="tool_id",
                                                        match=models.MatchAny(any=missing))])
        offset = None
        while True:
            points, offset = client().scroll(NAMES_COLLECTION, scroll_filter=flt, limit=1000,
                                             offset=offset, with_payload=["tool_id", "text"])
            _NAMES_TEXT.update({p.payload["tool_id"]: p.payload["text"] for p in points})
            if offset is None:
                break
    return [Hit(t, "", 0.0, "name", _NAMES_TEXT.get(t, "")) for t in ids]


def groups_of(row: dict) -> list[str]:
    kind = row["type"]
    out = ["all", kind, kind.split("_")[0]]
    if kind.startswith("task") and (row.get("own_query_sim") or 0) < NEAR_COPY:
        out.append("task, no near-copies")
    return out


def score(ranking: list[str], relevant: set[str]) -> dict[str, float]:
    first = next((i for i, t in enumerate(ranking[:K], 1) if t in relevant), None)
    return {"R@1": float(first == 1), "R@5": float(first is not None and first <= 5),
            "R@10": float(first is not None), "MRR@10": 1.0 / first if first else 0.0}


def plot(table, n_by_group, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["R@1", "R@10"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True, facecolor=SURFACE)
    cmap = plt.get_cmap("Blues")
    for ax, metric in zip(axes, metrics):
        data = np.array([[table[(s, g)][metric] for g in PLOT_GROUPS] for s in SYSTEMS])
        ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(len(SYSTEMS)):
            for j in range(len(PLOT_GROUPS)):
                v = data[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if v > 0.6 else INK)
        ax.set_xticks(range(len(PLOT_GROUPS)),
                      [f"{g.replace('_', ' ')}\nn={n_by_group[g]}" for g in PLOT_GROUPS],
                      fontsize=8.5, color=INK2)
        ax.set_yticks(range(len(SYSTEMS)), [s.replace("_", " ") for s in SYSTEMS],
                      fontsize=9, color=INK2)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        # White gaps between cells, and a rule above the hybrids and the API.
        ax.set_xticks(np.arange(-.5, len(PLOT_GROUPS)), minor=True)
        ax.set_yticks(np.arange(-.5, len(SYSTEMS)), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
        for y in (len(SINGLE) - .5, len(SYSTEMS) - 1.5):
            ax.axhline(y, color=MUTED, linewidth=1)
        ax.set_title(metric, fontsize=11, color=INK, loc="left")
    fig.suptitle("Finding the right tool: share of queries with it in first place (R@1) "
                 "and in the top 10 (R@10)", fontsize=11.5, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--queries", type=Path, default=QUERIES_DIR / "queries.jsonl")
    p.add_argument("--runs", type=Path, default=QUERIES_DIR / "runs.jsonl",
                   help="cache of every retriever's hit lists (resumed)")
    p.add_argument("--strict", action="store_true",
                   help="count only the query's own tool, not its same-name records")
    p.add_argument("--workers", type=int, default=8, help="parallel Qdrant queries")
    p.add_argument("--out-dir", type=Path, default=EVAL)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ("urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    rows = [json.loads(line) for line in args.queries.open(encoding="utf-8")]
    runs = run_all(sorted({r["query"] for r in rows}), args.runs, args.workers)

    sums = defaultdict(lambda: defaultdict(float))
    n_by_group = defaultdict(int)
    for row in rows:
        relevant = {row["tool_id"]} if args.strict else set(row["relevant"])
        groups = groups_of(row)
        for g in groups:
            n_by_group[g] += 1
        for system in SYSTEMS:
            s = score(system_ranking(system, row["query"], runs), relevant)
            for g in groups:
                for m, v in s.items():
                    sums[(system, g)][m] += v
    table = {key: {m: v / n_by_group[key[1]] for m, v in ms.items()} for key, ms in sums.items()}

    tag = "strict" if args.strict else "lenient"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"retriever_comparison_{tag}"
    with open(f"{stem}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "group", "n", "R@1", "R@5", "R@10", "MRR@10"])
        for g in GROUPS:
            for s in SYSTEMS:
                m = table[(s, g)]
                w.writerow([s, g, n_by_group[g]] + [round(m[k], 4)
                                                    for k in ("R@1", "R@5", "R@10", "MRR@10")])
    summary = {
        "queries": str(args.queries), "n_queries": len(rows), "relevance": tag, "k": K,
        "depth_per_retriever": DEPTH, "rrf_k": hybrid.RRF_K, "name_boost": hybrid.NAME_BOOST,
        "index_counts": {COLLECTION: client().count(COLLECTION).count,
                         NAMES_COLLECTION: client().count(NAMES_COLLECTION).count},
        "observatory_api": observatory_api.URL,
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_by_group": dict(n_by_group),
        "results": {g: {s: {m: round(v, 4) for m, v in table[(s, g)].items()} for s in SYSTEMS}
                    for g in GROUPS},
    }
    Path(f"{stem}.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(table, n_by_group, Path(f"{stem}.png"))

    print(f"MRR@10 ({tag}), {len(rows)} queries")
    head = ["all", "name", "task", "task, no near-copies"]
    print(f"{'system':18}" + "".join(f"{g:>22}" for g in head))
    for s in SYSTEMS:
        print(f"{s:18}" + "".join(f"{table[(s, g)]['MRR@10']:>22.3f}" for g in head))
    print(f"-> {stem}.csv / .json / .png")


if __name__ == "__main__":
    main()
