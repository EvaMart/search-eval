"""
Baseline: the Observatory's current search (its public API), for comparison.

The API returns tools in pages of 10, best first, with no score; a hit's score is
minus its rank. Its `id` is the tool's `_id` in `toolsDev`, the same as the
index's `tool_id`, so its results are judged against the same relevant tools.
It is not one of the hybrid's retrievers.

Usage:
    python -m search_eval.retrieval.observatory_api "PyMut"
    python -m search_eval.retrieval.observatory_api -k 20 "align long reads"
"""

from __future__ import annotations

import argparse
import math

import requests

from search_eval.retrieval.index import Hit

URL = "https://observatory.openebench.bsc.es/api/search"
SEARCH_IN = "name,description,topics,operations"  # as the web search sends it
PAGE_SIZE = 10


def search(query: str, k: int = 10, timeout: float = 60) -> list[Hit]:
    """The API's first k tools for the query (fewer if it finds fewer)."""
    hits = []
    for page in range(math.ceil(k / PAGE_SIZE)):
        r = requests.get(URL, params={"page": page, "q": query, "searchIn": SEARCH_IN},
                         timeout=timeout)
        r.raise_for_status()
        tools = r.json().get("tools") or []
        for t in tools:
            name = t.get("preferred_label") or t.get("name") or ""
            if isinstance(name, list):
                name = name[0] if name else ""
            hits.append(Hit(str(t["id"]), name, -float(len(hits) + 1), "api", ""))
        if len(tools) < PAGE_SIZE:
            break
    return hits[:k]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("query")
    p.add_argument("-k", type=int, default=10, help="number of tools (default 10)")
    args = p.parse_args()
    for rank, h in enumerate(search(args.query, args.k), 1):
        print(f"{rank:>3}  {h.name}  ({h.tool_id})")


if __name__ == "__main__":
    main()
