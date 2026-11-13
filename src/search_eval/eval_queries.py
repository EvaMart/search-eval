"""
Build the evaluation query set for the retrievers (search_eval.retrieval): a seeded
sample of tools and, for each, queries whose right answer is that tool.

Stages, each writing to --out-dir:

    sample    sample.json   the sampled tools ({tool_id, name, names, sources,
                            type, n_queries}) and the parameters
    names     name_queries.jsonl   queries built from the tool's display name,
                            no LLM: the exact name, a spelling variant (other
                            separators, lowercase) and a one-edit typo
    tasks     task_queries.jsonl   queries an LLM writes from the tool's record
                            (as the pipeline formats it, build_tool_context):
                            one keyword search, one task description, one
                            question; never the tool's name
              task_completions.jsonl   every completion, so a rerun asks only
                            for the missing or failed ones
              task_calls.jsonl   one row per Gepeto call, every run
    finish    queries.jsonl   name and task queries together; task queries get
                            `own_query_sim`, their cosine to the closest of their
                            tool's indexed synthetic queries (the leakage check)
              dataset.json   parameters, counts and the leakage summary

Task queries are written by a different model (glm5.3-fp8) and a different prompt
(TASK_PROMPT) from the indexed synthetic queries (Qwen3.6-27B, prompt v2), so the
`dense_queries` retriever is not matching queries against near copies of
themselves. Some overlap is unavoidable: both read the same record.

Tools are sampled from the index itself (`tool_search_exp_names`, one point per
tool, with the tool's name and labels), so every one is searchable; tools without
synthetic queries (`n_queries` 0) are kept, as a user may look for them too.

Each query row is {query, type, tool_id, relevant}. `relevant` is the tool plus
every other tool with the same normalised display name (duplicate records such as
the several "STAR"s), so finding a duplicate is not counted a miss; `tool_id`
alone gives the strict score.

Usage:
    python -m search_eval.eval_queries sample --n 300 --seed 20261112
    python -m search_eval.eval_queries names
    python -m search_eval.eval_queries tasks          # Gepeto; resumes
    python -m search_eval.eval_queries finish
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from search_eval.common import DATA, provenance
from search_eval.retrieval.bm25_names import normalise
from search_eval.retrieval.index import COLLECTION, NAMES_COLLECTION, client, kind_filter

OUT_DIR = DATA / "eval_queries"
MIN_TYPO_LEN = 6  # a typo in a shorter name makes another name ("STAR" -> "STAT")
_SEPARATORS = re.compile(r"[\s_\-.]+")

TASK_MODEL = "glm5.3-fp8"
TASK_TEMPERATURE = 0.7
TASK_TYPES = {"keyword": "task_keyword", "description": "task_description",
              "question": "task_question"}
# Not the pipeline's prompt (SYSTEM_PROMPTS): other styles and lengths, so the
# evaluation queries are not the indexed synthetic queries rewritten.
TASK_PROMPT = """You simulate researchers searching a registry of life-science \
research software. Given a tool record, write the three searches below, each one \
a different person who needs this tool but has never heard of it.

- "keyword": 2-4 words, typed as a terse search-box query (e.g. "variant calling \
nanopore").
- "description": 6-14 words describing the task they want done, in their own \
words, not the record's (e.g. "find structural variants in long reads from a \
tumour sample").
- "question": a question or problem statement as they would ask a colleague \
(e.g. "how can I tell which of my contigs are bacterial contamination?").

Rules:
- Never use the tool's name, labels or any part of them.
- Describe the need, not the tool: do not copy phrases from the description.
- Only needs this tool really meets, according to the record.
- If the record is too vague to say what the tool does, return null for all three.

Respond with JSON only: {"keyword": "...", "description": "...", "question": "..."}"""


def index_tools() -> list[dict]:
    """Every tool of the names collection: {tool_id, name, names, sources, type}."""
    tools, offset = [], None
    while True:
        points, offset = client().scroll(
            NAMES_COLLECTION, limit=10000, offset=offset, with_vectors=False,
            with_payload=["tool_id", "name", "text", "sources", "type"])
        for p in points:
            tools.append({"tool_id": p.payload["tool_id"], "name": p.payload["name"],
                          "names": p.payload["text"].split("\n"),
                          "sources": p.payload.get("sources", []),
                          "type": p.payload.get("type", [])})
        if offset is None:
            return sorted(tools, key=lambda t: t["tool_id"])


def query_counts() -> dict[str, int]:
    """Synthetic-query points per tool."""
    result = client().facet(COLLECTION, key="tool_id", facet_filter=kind_filter("synthetic_query"),
                            limit=1_000_000, exact=True)
    return {h.value: h.count for h in result.hits}


def variant(name: str) -> str | None:
    """
    The name as a user might type it differently: separators as spaces, lowercase
    ("purge_dups" -> "purge dups", "Purge-Haplotigs" -> "purge haplotigs"). None
    when that is only the name lowercased, which BM25 sees as the name itself.
    CamelCase is not split: "MetMashR" -> "met mash r" is no one's spelling.
    """
    v = _SEPARATORS.sub(" ", name).strip().lower()
    return v if v != name.lower() else None


def typo(name: str, rng: random.Random) -> str | None:
    """
    One edit inside the name (deleted letter, swapped neighbours, doubled letter),
    never on the first letter. None for names shorter than MIN_TYPO_LEN letters.
    """
    letters = [i for i, c in enumerate(name) if c.isalpha() and i > 0]
    if len(normalise(name)) < MIN_TYPO_LEN or len(letters) < 2:
        return None
    i = rng.choice(letters)
    edit = rng.choice(["delete", "swap", "double"])
    if edit == "delete":
        return name[:i] + name[i + 1:]
    if edit == "double":
        return name[:i] + name[i] + name[i:]
    j = i + 1 if i + 1 < len(name) and name[i + 1].isalpha() else i - 1
    if j < 1 or name[i].lower() == name[j].lower():
        return name[:i] + name[i + 1:]  # nothing to swap: delete instead
    a, b = sorted((i, j))
    return name[:a] + name[b] + name[a] + name[b + 1:]


def sample(args) -> None:
    tools = index_tools()
    counts = query_counts()
    rng = random.Random(args.seed)
    picked = sorted(rng.sample(tools, args.n), key=lambda t: t["tool_id"])
    for t in picked:
        t["n_queries"] = counts.get(t["tool_id"], 0)
    meta = {
        "n": args.n,
        "seed": args.seed,
        "sampled_from": f"{NAMES_COLLECTION} ({len(tools)} tools)",
        "index_counts": {COLLECTION: client().count(COLLECTION).count,
                         NAMES_COLLECTION: client().count(NAMES_COLLECTION).count},
        "tools_with_queries_in_sample": sum(t["n_queries"] > 0 for t in picked),
        "sampled_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tools": picked,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sample.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(f"{args.n} tools ({meta['tools_with_queries_in_sample']} with synthetic queries) "
          f"-> {args.out_dir / 'sample.json'}")


def relevant_sets(tools: list[dict]) -> dict[str, list[str]]:
    """Per sampled tool: itself and every tool with the same normalised display name."""
    by_name: dict[str, set[str]] = defaultdict(set)
    for t in index_tools():
        by_name[normalise(t["name"])].add(t["tool_id"])
    return {t["tool_id"]: sorted(by_name[normalise(t["name"])] | {t["tool_id"]})
            for t in tools}


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def names(args) -> None:
    meta = json.loads((args.out_dir / "sample.json").read_text())
    relevant_of = relevant_sets(meta["tools"])
    rng = random.Random(meta["seed"])
    rows = []
    for t in meta["tools"]:
        relevant = relevant_of[t["tool_id"]]
        for kind, q in (("name_exact", t["name"]), ("name_variant", variant(t["name"])),
                        ("name_typo", typo(t["name"], rng))):
            if q:
                rows.append({"query": q, "type": kind, "tool_id": t["tool_id"],
                             "relevant": relevant})
    path = args.out_dir / "name_queries.jsonl"
    write_rows(path, rows)
    by_type = defaultdict(int)
    for r in rows:
        by_type[r["type"]] += 1
    print(f"{len(rows)} name queries {dict(by_type)} -> {path}")


def tool_contexts(tool_ids: list[str]) -> dict[str, str | None]:
    """Each tool's record as the pipeline shows it to the query LLM (None: too thin)."""
    from bson import ObjectId

    from application.services.semantic_search.synthetic_queries import (
        CONTEXT_PROJECTION, build_tool_context)
    from infrastructure.config import PipelineConfig
    from infrastructure.db.repositories import from_config

    repos = from_config(PipelineConfig.from_env())
    out = {t: None for t in tool_ids}
    for tool in repos.tools.iter_projected(
            {"_id": {"$in": [ObjectId(t) for t in tool_ids]}}, CONTEXT_PROJECTION,
            batch_size=500):
        out[str(tool["_id"])] = build_tool_context(tool.get("data") or {})
    return out


def parse_task_queries(raw: str, names: list[str]) -> dict[str, str | None] | None:
    """
    {keyword, description, question} from a completion, or None if unparseable.
    A query that names the tool (the pipeline's `_mentions`) becomes None.
    """
    from application.services.semantic_search.synthetic_queries import _THINK, _mentions

    text = _THINK.sub("", raw or "").strip()
    start, end = text.find("{"), text.rfind("}")
    try:
        answer = json.loads(text[start:end + 1])
    except (ValueError, TypeError):
        return None
    if not isinstance(answer, dict):
        return None
    out = {}
    for key in TASK_TYPES:
        q = answer.get(key)
        q = q.strip() if isinstance(q, str) else None
        out[key] = q if q and not _mentions(q, names) else None
    return out


def tasks(args) -> None:
    from infrastructure.config import Credentials
    from infrastructure.external.gepeto import GepetoClient

    meta = json.loads((args.out_dir / "sample.json").read_text())
    tools = {t["tool_id"]: t for t in meta["tools"]}
    contexts = tool_contexts(list(tools))
    gepeto = GepetoClient(Credentials.from_env().require("gepeto_api_key").gepeto_api_key,
                          timeout=args.timeout)

    path = args.out_dir / "task_completions.jsonl"
    done = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            if r["model"] == args.model and r["prompt"] == TASK_PROMPT and not r.get("error"):
                done[r["tool_id"]] = r
    todo = [t for t, c in contexts.items() if c and t not in done]
    logging.info("%d tools: %d cached, %d without a usable record, %d to ask",
                 len(tools), len(done), sum(c is None for c in contexts.values()), len(todo))

    def ask(tool_id: str) -> tuple[dict, dict]:
        messages = [{"role": "system", "content": TASK_PROMPT},
                    {"role": "user", "content": contexts[tool_id]}]
        row = {"tool_id": tool_id, "model": args.model, "prompt": TASK_PROMPT}
        t0 = time.perf_counter()
        try:
            raw, usage = gepeto.query(messages, model=args.model, temperature=args.temperature)
            error = None if parse_task_queries(raw, tools[tool_id]["names"]) is not None \
                else (usage.get("error") or "unparseable completion")
        except Exception as exc:
            raw, usage, error = None, {}, f"{type(exc).__name__}: {exc}"
        call = {"tool_id": tool_id, "model": args.model, "wall_s": round(time.perf_counter() - t0, 2),
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "error": error}
        return {**row, "completion": raw, "usage": usage, "error": error}, call

    with path.open("a", encoding="utf-8") as fh, \
            (args.out_dir / "task_calls.jsonl").open("a", encoding="utf-8") as calls, \
            ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(ask, t) for t in todo]
        failed = 0
        for n, future in enumerate(as_completed(futures), 1):
            row, call = future.result()
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            calls.write(json.dumps(call) + "\n")
            calls.flush()
            if row["error"]:
                failed += 1
            else:
                done[row["tool_id"]] = row
            if n % 25 == 0 or n == len(todo):
                logging.info("%d / %d asked, %d failed", n, len(todo), failed)
    # One row per tool, the latest good answer winning.
    write_rows(path, list(done.values()))

    relevant_of = relevant_sets(meta["tools"])
    rows, empty = [], 0
    for tool_id, r in sorted(done.items()):
        queries = parse_task_queries(r["completion"], tools[tool_id]["names"])
        if not any(queries.values()):
            empty += 1
        for key, kind in TASK_TYPES.items():
            if queries[key]:
                rows.append({"query": queries[key], "type": kind, "tool_id": tool_id,
                             "relevant": relevant_of[tool_id]})
    write_rows(args.out_dir / "task_queries.jsonl", rows)
    meta["tasks"] = {
        "model": args.model, "temperature": args.temperature, "prompt": TASK_PROMPT,
        "tools_without_record": sum(c is None for c in contexts.values()),
        "tools_answered": len(done), "tools_judged_too_vague": empty,
        "tools_failed": len(todo) - sum(t in done for t in todo),
        "queries": len(rows),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **provenance(),
    }
    (args.out_dir / "sample.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(f"{len(rows)} task queries from {len(done) - empty} tools "
          f"({meta['tasks']['tools_failed']} failed, {empty} too vague, "
          f"{meta['tasks']['tools_without_record']} without a record) "
          f"-> {args.out_dir / 'task_queries.jsonl'}")


NEAR_COPY = 0.95  # cosine above which a task query is a paraphrase of an indexed one


def indexed_query_vectors(tool_ids: list[str]):
    """The dense vectors of these tools' synthetic-query points, as (tool ids, matrix)."""
    import numpy as np
    from qdrant_client import models

    flt = models.Filter(must=[
        models.FieldCondition(key="kind", match=models.MatchValue(value="synthetic_query")),
        models.FieldCondition(key="tool_id", match=models.MatchAny(any=tool_ids))])
    owners, vectors, offset = [], [], None
    while True:
        points, offset = client().scroll(COLLECTION, scroll_filter=flt, limit=1000, offset=offset,
                                         with_payload=["tool_id"], with_vectors=["dense"])
        for pt in points:
            owners.append(pt.payload["tool_id"])
            vectors.append(pt.vector["dense"])
        if offset is None:
            return np.array(owners), np.array(vectors, dtype=np.float32)


def finish(args) -> None:
    import numpy as np

    from search_eval.retrieval.index import encoder

    meta = json.loads((args.out_dir / "sample.json").read_text())
    read = lambda name: [json.loads(line) for line in (args.out_dir / name).open(encoding="utf-8")]
    name_rows, task_rows = read("name_queries.jsonl"), read("task_queries.jsonl")

    # Leakage: how close each task query is to its own tool's indexed queries,
    # against the closest indexed query of any *other* sampled tool.
    owners, matrix = indexed_query_vectors([t["tool_id"] for t in meta["tools"]])
    q = encoder().encode([r["query"] for r in task_rows], normalize_embeddings=True,
                         batch_size=64, convert_to_numpy=True)
    sims = q @ matrix.T
    own, other = [], []
    for r, row in zip(task_rows, sims):
        mine = owners == r["tool_id"]
        r["own_query_sim"] = round(float(row[mine].max()), 4) if mine.any() else None
        if mine.any():
            own.append(r["own_query_sim"])
            other.append(float(row[~mine].max()))
    own_a, other_a = np.array(own), np.array(other)
    leakage = {
        "task_queries_with_indexed_queries": len(own),
        "own_query_sim": {"median": round(float(np.median(own_a)), 3),
                          "p90": round(float(np.percentile(own_a, 90)), 3)},
        "best_other_tool_query_sim": {"median": round(float(np.median(other_a)), 3),
                                      "p90": round(float(np.percentile(other_a, 90)), 3)},
        f"share_near_copies_ge_{NEAR_COPY}": round(float((own_a >= NEAR_COPY).mean()), 3),
        "by_type_median_own_sim": {
            kind: round(float(np.median([r["own_query_sim"] for r in task_rows
                                         if r["type"] == kind and r["own_query_sim"] is not None])), 3)
            for kind in TASK_TYPES.values()},
    }

    rows = name_rows + task_rows
    write_rows(args.out_dir / "queries.jsonl", rows)
    by_type = defaultdict(int)
    for r in rows:
        by_type[r["type"]] += 1
    dataset = {
        "description": "Evaluation queries for the tool_search retrievers: name queries built "
                       "from display names, task queries written by an LLM from the tool's "
                       "record. Each row's `relevant` is the tool and its same-name records.",
        "files": {"queries": "queries.jsonl", "sample": "sample.json"},
        "n_tools": meta["n"], "seed": meta["seed"], "index_counts": meta["index_counts"],
        "queries_by_type": dict(sorted(by_type.items())),
        "tasks": {k: v for k, v in meta.get("tasks", {}).items() if k != "prompt"},
        "task_prompt": meta.get("tasks", {}).get("prompt"),
        "leakage": leakage,
        "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (args.out_dir / "dataset.json").write_text(json.dumps(dataset, indent=2, ensure_ascii=False) + "\n")
    print(f"{len(rows)} queries {dict(sorted(by_type.items()))} -> {args.out_dir / 'queries.jsonl'}")
    print("leakage:", json.dumps(leakage, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("sample", help="sample the tools")
    s.add_argument("--n", type=int, default=300)
    s.add_argument("--seed", type=int, default=20261112)
    s.set_defaults(func=sample)
    s = sub.add_parser("names", help="name queries for the sampled tools")
    s.set_defaults(func=names)
    s = sub.add_parser("tasks", help="LLM task queries for the sampled tools (Gepeto)")
    s.add_argument("--model", default=TASK_MODEL)
    s.add_argument("--temperature", type=float, default=TASK_TEMPERATURE)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--timeout", type=float, default=300, help="seconds per Gepeto call")
    s.set_defaults(func=tasks)
    s = sub.add_parser("finish", help="merge the queries and check leakage")
    s.set_defaults(func=finish)
    for s in sub.choices.values():
        s.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()
    load_dotenv(override=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ("urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args.func(args)


if __name__ == "__main__":
    main()
