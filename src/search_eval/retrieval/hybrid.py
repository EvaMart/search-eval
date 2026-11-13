"""
Hybrid retrieval: the individual retrievers fused with Reciprocal Rank Fusion.

Each retriever returns its top `depth` tools; a tool's fused score is

    sum over retrievers r of  weight_r(tool) / (rrf_k + rank_r)

(rank from 1; a retriever that did not return the tool adds nothing; the weight
is the retriever's, unless that hit has its own). RRF uses
ranks only, so the retrievers' scores, which are not comparable (cosine vs
unbounded BM25), never meet. The retrievers search two collections
(`tool_search`, `tool_search_exp_names`), so the fusion is done here rather than
in one Qdrant prefetch query.

Each hit of the names retriever whose name *is* the query (exactly or nearly,
`bm25_names.name_match`) is weighted `name_boost` (default 3) instead of 1: a
query that is a tool's name is a lookup, and the other retrievers, which return
tools *about* that name, should not outvote it. Only the matching hits: boosting
the whole names list would raise "samtools_stats" with "samtools". Near matches
count only when no hit matches exactly, so "bwa mem" boosts bwa_mem, not
bwa-mem2 and BWA-MEME as well. With the
default, a matching hit first in the names list counts 3/61, as much as being
first in the three other retrievers together.

When no hit matches, the names retriever only adds noise (names that share an
ordinary word with the query: "Diploid Constructor" for a query about diploid
assemblies); `names_unmatched` sets its weight for such a query (default 0:
dropped). On the evaluation queries (analysis.retriever_comparison) dropping it
keeps the name results and raises task MRR@10 from 0.465 to 0.511.

`fuse` is pure: the comparison of retrievers can run each once per query, then
`name_boosts` and `fuse` those same lists, rather than querying again.

Usage:
    python -m search_eval.retrieval.hybrid "remove haplotigs from a diploid assembly"
    python -m search_eval.retrieval.hybrid --show-ranks "samtools"
    python -m search_eval.retrieval.hybrid --name-boost 1 "samtools"   # no boost
    python -m search_eval.retrieval.hybrid --names-unmatched 1 "align long reads"   # keep it
    python -m search_eval.retrieval.hybrid --retrievers dense_description bm25_description \\
        --weight bm25_description=0.5 "align long reads"
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

from search_eval.retrieval import bm25_description, bm25_names, dense_description, dense_queries
from search_eval.retrieval.index import Hit

RETRIEVERS: dict[str, Callable[[str, int], list[Hit]]] = {
    "dense_description": dense_description.search,
    "bm25_description": bm25_description.search,
    "bm25_names": bm25_names.search,
    "dense_queries": dense_queries.search,
}
RRF_K = 60  # the usual constant (Cormack et al. 2009)
DEPTH = 100
NAMES = "bm25_names"
NAME_BOOST = 3.0
NAMES_UNMATCHED = 0.0


@dataclass(frozen=True)
class FusedHit:
    tool_id: str
    name: str
    score: float
    ranks: dict[str, int]  # rank in each retriever that returned the tool


def fuse(rankings: dict[str, list[Hit]], k: int = 10, rrf_k: float = RRF_K,
         weights: dict[str, float] | None = None,
         hit_weights: dict[str, dict[str, float]] | None = None) -> list[FusedHit]:
    """
    The k best tools of several rankings (retriever name -> hits, best first).
    `weights` are per retriever (default 1); `hit_weights` (retriever -> tool id
    -> weight) override them for single hits.
    """
    weights, hit_weights = weights or {}, hit_weights or {}
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    names: dict[str, str] = {}
    for retriever, hits in rankings.items():
        w = weights.get(retriever, 1.0)
        own = hit_weights.get(retriever, {})
        for rank, h in enumerate(hits, 1):
            scores[h.tool_id] = scores.get(h.tool_id, 0.0) + own.get(h.tool_id, w) / (rrf_k + rank)
            ranks.setdefault(h.tool_id, {})[retriever] = rank
            names.setdefault(h.tool_id, h.name)
    # Ties (same ranks in different retrievers) broken by tool id, to be deterministic.
    order = sorted(scores, key=lambda t: (-scores[t], t))[:k]
    return [FusedHit(t, names[t], scores[t], ranks[t]) for t in order]


def name_boosts(query: str, rankings: dict[str, list[Hit]],
                weights: dict[str, float] | None = None,
                name_boost: float = NAME_BOOST) -> dict[str, dict[str, float]]:
    """
    `hit_weights` for `fuse`: `name_boost` for each hit of the names retriever that
    is an exact match of the whole query, or, when none is, a near match (never
    below that retriever's own weight).
    """
    w = max((weights or {}).get(NAMES, 1.0), name_boost)
    matches = {h.tool_id: bm25_names.name_match(query, h) for h in rankings.get(NAMES, [])}
    level = "exact" if "exact" in matches.values() else "near"
    boosted = {t: w for t, m in matches.items() if m == level}
    return {NAMES: boosted} if boosted else {}


def name_weights(query: str, rankings: dict[str, list[Hit]],
                 weights: dict[str, float] | None = None, name_boost: float = NAME_BOOST,
                 names_unmatched: float = NAMES_UNMATCHED) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """
    (`weights`, `hit_weights`) for `fuse`: the name matches boosted, and the names
    retriever weighted `names_unmatched` when it has none.
    """
    boosts = name_boosts(query, rankings, weights, name_boost)
    weights = dict(weights or {})
    if not boosts:
        weights[NAMES] = names_unmatched
    return weights, boosts


def run(query: str, retrievers: list[str] | None = None, depth: int = DEPTH) -> dict[str, list[Hit]]:
    """Each retriever's top `depth` tools for the query."""
    return {r: RETRIEVERS[r](query, depth) for r in retrievers or RETRIEVERS}


def search(query: str, k: int = 10, retrievers: list[str] | None = None, depth: int = DEPTH,
           rrf_k: float = RRF_K, weights: dict[str, float] | None = None,
           name_boost: float = NAME_BOOST,
           names_unmatched: float = NAMES_UNMATCHED) -> list[FusedHit]:
    """The k best tools of the fused retrievers (default: all of them)."""
    rankings = run(query, retrievers, depth)
    return fuse(rankings, k, rrf_k,
                *name_weights(query, rankings, weights, name_boost, names_unmatched))


def _weight(s: str) -> tuple[str, float]:
    name, value = s.split("=", 1)
    if name not in RETRIEVERS:
        raise argparse.ArgumentTypeError(f"unknown retriever {name!r}")
    return name, float(value)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("query")
    p.add_argument("-k", type=int, default=10, help="number of tools (default 10)")
    p.add_argument("--retrievers", nargs="+", choices=list(RETRIEVERS),
                   help="retrievers to fuse (default: all)")
    p.add_argument("--depth", type=int, default=DEPTH,
                   help=f"tools taken from each retriever (default {DEPTH})")
    p.add_argument("--rrf-k", type=float, default=RRF_K, help=f"RRF constant (default {RRF_K})")
    p.add_argument("--weight", type=_weight, action="append", default=[],
                   metavar="RETRIEVER=W", help="weight of a retriever (default 1)")
    p.add_argument("--name-boost", type=float, default=NAME_BOOST,
                   help=f"weight of a {NAMES} hit that is an exact or near name match "
                        f"(default {NAME_BOOST}; 1 to switch off)")
    p.add_argument("--names-unmatched", type=float, default=NAMES_UNMATCHED,
                   help=f"weight of {NAMES} when none of its hits is a name match "
                        f"(default {NAMES_UNMATCHED:g}: dropped; 1 to keep it)")
    p.add_argument("--show-ranks", action="store_true",
                   help="print each tool's rank in every retriever")
    args = p.parse_args()

    retrievers = args.retrievers or list(RETRIEVERS)
    hits = search(args.query, args.k, retrievers, args.depth, args.rrf_k, dict(args.weight),
                  args.name_boost, args.names_unmatched)
    for rank, h in enumerate(hits, 1):
        print(f"{rank:>3}  {h.score:.4f}  {h.name}  ({h.tool_id})")
        if args.show_ranks:
            print("       " + "  ".join(f"{r}={h.ranks.get(r, '-')}" for r in retrievers))


if __name__ == "__main__":
    main()
