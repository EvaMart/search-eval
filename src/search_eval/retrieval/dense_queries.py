"""
Retriever 4: the dense vector of the synthetic-query points.

Cosine similarity between the query and the synthetic queries the LLM wrote for
each tool (about five per tool), embedded with gte-modernbert-base like the
description points. A tool has several query points, so hits are grouped by
`tool_id`: a tool scores as its best-matching query. Tools whose queries were
`insufficient` have no query points and can't be found here.

Usage:
    python -m search_eval.retrieval.dense_queries "remove haplotigs from a diploid assembly"
    python -m search_eval.retrieval.dense_queries -k 20 --show-text "align long reads"
"""

from __future__ import annotations

import argparse

from qdrant_client import models

from search_eval.retrieval.index import COLLECTION, Hit, client, encode, kind_filter, to_hit


def search(query: str, k: int = 10, prompt_version: str | None = None) -> list[Hit]:
    """
    The k tools with the synthetic query closest to this one, best first. Each
    hit's text is that best-matching query. `prompt_version` keeps only queries
    written by that prompt (the index holds only "2" today).
    """
    flt = kind_filter("synthetic_query")
    if prompt_version is not None:
        flt.must.append(models.FieldCondition(
            key="prompt_version", match=models.MatchValue(value=prompt_version)))
    result = client().query_points_groups(
        COLLECTION,
        query=encode(query),
        using="dense",
        query_filter=flt,
        group_by="tool_id",
        group_size=1,
        limit=k,
        with_payload=["tool_id", "name", "kind", "text"],
    )
    return [to_hit(g.hits[0]) for g in result.groups]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("query")
    p.add_argument("-k", type=int, default=10, help="number of tools (default 10)")
    p.add_argument("--prompt-version", help="only queries written by this prompt version")
    p.add_argument("--show-text", action="store_true",
                   help="print the synthetic query each tool matched on")
    args = p.parse_args()

    for rank, h in enumerate(search(args.query, args.k, args.prompt_version), 1):
        print(f"{rank:>3}  {h.score:.3f}  {h.name}  ({h.tool_id})")
        if args.show_text:
            print("       " + h.text[:200])


if __name__ == "__main__":
    main()
