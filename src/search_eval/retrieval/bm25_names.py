"""
Retriever 3: BM25 over tool names (the experimental `tool_search_exp_names`).

Keyword match between the query and each tool's name and labels only, so a name
match is the whole signal rather than diluted by descriptions. Stemming and
stopwords are off (the pipeline's NAME_BM25_OPTIONS): "STAR" does not match
"stars", and a name that is an English word still counts. The query is
tokenised with the same options as the index, or the hashed tokens would not
line up. One point per tool, so no grouping by tool is needed.

`name_match` tells whether a hit's name or a label *is* the query (exactly, or
but for case, punctuation, spacing or a character or two): the hybrid search
weights this retriever higher for such a query.

The collection is built by the pipeline (`rsetl search-index --names`).

Usage:
    python -m search_eval.retrieval.bm25_names "samtools"
    python -m search_eval.retrieval.bm25_names -k 20 --show-text "purge dups"
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from difflib import SequenceMatcher

from qdrant_client import models

from infrastructure.db.qdrant.tool_search_index import NAME_BM25_OPTIONS
from search_eval.retrieval.index import NAMES_COLLECTION, Hit, client, to_hit


def search(query: str, k: int = 10) -> list[Hit]:
    """The k tools whose name or labels best match the query's words, best first."""
    result = client().query_points(
        NAMES_COLLECTION,
        query=models.Document(text=query, model="qdrant/bm25", options=NAME_BM25_OPTIONS),
        using="bm25",
        limit=k,
        with_payload=["tool_id", "name", "kind", "text"],
    )
    return [to_hit(p) for p in result.points]


NEAR_MATCH = 0.9  # SequenceMatcher ratio of the normalised strings
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalise(s: str) -> str:
    """Lowercase ASCII letters and digits only: "Purge_Dups", "purge dups" -> "purgedups"."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return _NON_ALNUM.sub("", s.lower())


def name_match(query: str, hit: Hit, near: float = NEAR_MATCH) -> str | None:
    """
    "exact" if one of the hit's names or labels normalises to the query, "near" if
    one is within `near` similarity of it (a version digit, a plural: "bwa mem" vs
    "BWA-MEM2"), else None. Short strings must match exactly: one character is
    too much of "STAR" to differ.
    """
    q = normalise(query)
    if not q:
        return None
    best = None
    for name in hit.text.split("\n"):
        n = normalise(name)
        if n == q:
            return "exact"
        if min(len(n), len(q)) >= 5 and SequenceMatcher(None, n, q).ratio() >= near:
            best = "near"
    return best


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("query")
    p.add_argument("-k", type=int, default=10, help="number of tools (default 10)")
    p.add_argument("--show-text", action="store_true",
                   help="print the names and labels each tool matched on")
    args = p.parse_args()

    for rank, h in enumerate(search(args.query, args.k), 1):
        match = name_match(args.query, h)
        print(f"{rank:>3}  {h.score:6.2f}  {h.name}  ({h.tool_id})"
              + (f"  [{match} name match]" if match else ""))
        if args.show_text:
            print("        " + " | ".join(h.text.split("\n"))[:200])


if __name__ == "__main__":
    main()
