"""
Retriever 1: the dense vector of the description points.

Cosine similarity between the query and each tool's description text (name,
description and EDAM terms, embedded with gte-modernbert-base). There is one
description point per tool, so no grouping by tool is needed.

Usage:
    python -m search_eval.retrieval.dense_description "remove haplotigs from a diploid assembly"
    python -m search_eval.retrieval.dense_description -k 20 --show-text "align long reads"
"""

from __future__ import annotations

import argparse

from search_eval.retrieval.index import COLLECTION, Hit, client, encode, kind_filter, to_hit


def search(query: str, k: int = 10) -> list[Hit]:
    """The k tools whose description is closest to the query, best first."""
    result = client().query_points(
        COLLECTION,
        query=encode(query),
        using="dense",
        query_filter=kind_filter("description"),
        limit=k,
        with_payload=["tool_id", "name", "kind", "text"],
    )
    return [to_hit(p) for p in result.points]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("query")
    p.add_argument("-k", type=int, default=10, help="number of tools (default 10)")
    p.add_argument("--show-text", action="store_true",
                   help="print the start of each matched description")
    args = p.parse_args()

    for rank, h in enumerate(search(args.query, args.k), 1):
        print(f"{rank:>3}  {h.score:.3f}  {h.name}  ({h.tool_id})")
        if args.show_text:
            print("       " + h.text.split("\n")[0][:200])


if __name__ == "__main__":
    main()
