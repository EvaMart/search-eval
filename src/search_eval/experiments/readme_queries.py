"""
Do synthetic queries change when the LLM also sees the tool's GitHub README?

Generates prompt-version-2 queries (README_PROMPT_VERSION: the usual record plus a
cleaned README excerpt) for a random sample of tools with a GitHub repository,
online through Gepeto, and writes them next to the tools' current (version 1)
queries for comparison. Nothing is written to MongoDB: the `semantic_search`
collection keeps its v1 queries.

Sampled tools have exactly one GitHub repository, used by no other tool (the
README of a hub repository such as ucscgenomebrowser/kent describes the hub, not
the tool), and already have a v1 document.

READMEs come from the Observatory's github-metadata-api
(GitHubClient.get_readme_via_metadata) and are cached, as is every completion:
re-running with the same --out-dir resumes, retrying only failures.

Writes to --out-dir:
    sample.json        the sampled tool ids and the parameters
    readmes.jsonl      {id, owner, repo, readme}           raw README (null: none)
    requests.jsonl     same shape as `rsetl synthetic-queries export`
    results.jsonl      same shape as the MN5 job's output ({id, completion, ...})
    calls.jsonl        one row per Gepeto call, every run (timing, tokens, errors)
    timing_summary_<UTC>.json   latency percentiles, throughput, tokens, errors
    comparison.jsonl   {id, tool_name, repo, readme_chars, context,
                        stored: {...}, generated: {...}}  (stored = the v1
                        queries in MongoDB; generated = this run)

requests.jsonl / results.jsonl can be fed to search_eval.analysis.*
(--requests / --results). Do NOT `rsetl synthetic-queries import` them unless you
mean to replace the stored v1 queries.

--prompt-version 1 runs the current prompt (no README) with the same model on
the same sample: the baseline that separates the README's effect from the
model's. The sample is the same for every run with the same --seed.

Usage:
    python -m search_eval.experiments.readme_queries
    python -m search_eval.experiments.readme_queries --prompt-version 1 \\
        --out-dir data/v1_glm
    python -m search_eval.experiments.readme_queries --n-tools 500 \\
        --model glm5.3-fp8 --out-dir data/readme_v2
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from application.services.semantic_search.synthetic_queries import (
    CONTEXT_PROJECTION,
    QUERY_SCHEMA,
    PROMPT_VERSION,
    README_PROMPT_VERSION,
    STATUS_GENERATED,
    SYSTEM_PROMPTS,
    VOCABULARY_PROJECTION,
    build_request,
    clean_readme,
    result_from_completion,
    vocabulary,
)
from infrastructure.config import Credentials, PipelineConfig
from infrastructure.db.repositories import from_config
from infrastructure.external.gepeto import GepetoClient
from infrastructure.external.github import GitHubClient

from search_eval.common import DATA, provenance

_GITHUB = re.compile(r"github\.com[/:]([^/\s#?]+)/([^/\s#?]+)", re.IGNORECASE)


def github_repos(tool: dict) -> set[tuple[str, str]]:
    """The tool's GitHub repositories as (owner, repo), lowercased, without .git."""
    repos = set()
    for item in (tool.get("data") or {}).get("repository") or []:
        match = _GITHUB.search((item or {}).get("url") or "")
        if match:
            repo = re.sub(r"\.git$", "", match.group(2))
            repos.add((match.group(1).lower(), repo.lower()))
    return repos


def read_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_tools(repos, n_tools: int, seed: int) -> tuple[list[dict], dict]:
    projection = {**CONTEXT_PROJECTION, "data.repository.url": 1}
    tools, per_repo = {}, Counter()
    for tool in repos.tools.iter_projected(
        {"data.repository.url": {"$regex": "github\\.com", "$options": "i"}},
        projection,
        batch_size=1000,
    ):
        found = github_repos(tool)
        per_repo.update(found)
        tools[str(tool["_id"])] = (tool, found)

    v1 = {
        doc["tool_id"]: doc
        for doc in repos.semantic_search.db_adapter.find(
            repos.semantic_search.collection_name,
            {"tool_id": {"$in": list(tools)}},
            projection={"_id": 0},
            batch_size=5000,
        )
    }
    eligible = sorted(
        tool_id
        for tool_id, (_, found) in tools.items()
        if len(found) == 1 and per_repo[next(iter(found))] == 1 and tool_id in v1
    )
    stats = {
        "tools_with_github": len(tools),
        "eligible": len(eligible),
        "sampled": min(n_tools, len(eligible)),
    }
    chosen = random.Random(seed).sample(eligible, stats["sampled"])
    sample = []
    for tool_id in chosen:
        tool, found = tools[tool_id]
        sample.append({"tool": tool, "repo": next(iter(found)), "v1": v1[tool_id]})
    return sample, stats


def fetch_readmes(sample, github: GitHubClient, path: Path, workers: int) -> dict:
    cached = read_jsonl(path)
    todo = [s for s in sample if str(s["tool"]["_id"]) not in cached]
    logging.info(f"READMEs: {len(cached)} cached, {len(todo)} to fetch")

    def fetch(entry):
        owner, repo = entry["repo"]
        return {
            "id": str(entry["tool"]["_id"]),
            "owner": owner,
            "repo": repo,
            "readme": github.get_readme_via_metadata(owner, repo),
        }

    with (
        path.open("a", encoding="utf-8") as handle,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        for n, future in enumerate(
            as_completed(pool.submit(fetch, s) for s in todo), 1
        ):
            row = future.result()
            cached[row["id"]] = row
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if n % 50 == 0:
                logging.info(f"READMEs fetched: {n}/{len(todo)}")
    return cached


def _call_record(request, model, workers, started_at, wall_s, meta, outcome, error):
    """One row of calls.jsonl: what a single Gepeto call cost and how it ended."""
    meta = meta or {}
    details = meta.get("completion_tokens_details") or {}
    completion_tokens = meta.get("completion_tokens")
    latency = meta.get("latency_s")
    return {
        "id": request.tool_id,
        "model": model,
        "workers": workers,
        "started_at": started_at,
        "wall_s": round(wall_s, 3),
        "latency_s": latency,
        "total_s": meta.get("total_s"),
        "attempts": meta.get("attempts"),
        "http_status": meta.get("http_status"),
        "prompt_tokens": meta.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
        "reasoning_tokens": details.get("reasoning_tokens"),
        "completion_tokens_per_s": (
            round(completion_tokens / latency, 2)
            if completion_tokens and latency
            else None
        ),
        "context_chars": len(request.context or ""),
        "outcome": outcome,
        "error": error,
    }


def generate(
    requests, gepeto, model, temperature, out_dir: Path, workers: int
) -> tuple[dict, list[dict], float]:
    """
    Completions keyed by tool id; only missing or failed ones are (re)asked.

    Also returns this run's call records and its wall-clock duration. Every call
    is appended to calls.jsonl, which is never rewritten: it is the log of the
    service across runs.
    """
    path = out_dir / "results.jsonl"
    done = {
        k: v
        for k, v in read_jsonl(path).items()
        if not v.get("error") and v.get("model") == model
    }
    todo = [r for r in requests if r.tool_id not in done]
    logging.info(f"Completions: {len(done)} cached, {len(todo)} to generate")

    def ask(request):
        row = {
            "id": request.tool_id,
            "tool_name": request.tool_name,
            "names": request.names,
            "input_hash": request.input_hash,
            "prompt_version": request.prompt_version,
            "model": model,
        }
        started_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        t0 = time.perf_counter()
        try:
            raw, meta = gepeto.query(
                request.messages(), model=model, temperature=temperature
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            call = _call_record(
                request,
                model,
                workers,
                started_at,
                time.perf_counter() - t0,
                None,
                "transport_error",
                error,
            )
            return {**row, "completion": None, "error": error}, call
        wall = time.perf_counter() - t0
        if not raw:
            error = meta.get("error") or "empty response"
            outcome, row = "http_error", {**row, "completion": raw, "error": error}
        elif result_from_completion(request, raw) is None:
            error = "unparseable completion"
            outcome, row = "unparseable", {**row, "completion": raw, "error": error}
        else:
            error, outcome = None, "ok"
            row = {**row, "completion": raw, "usage": meta}
        call = _call_record(
            request, model, workers, started_at, wall, meta, outcome, error
        )
        return row, call

    calls = []
    run_start = time.perf_counter()
    # Appended as they arrive, so an interrupted run keeps what it got...
    with (
        path.open("a", encoding="utf-8") as handle,
        (out_dir / "calls.jsonl").open("a", encoding="utf-8") as call_log,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        futures = [pool.submit(ask, r) for r in todo]
        for n, future in enumerate(as_completed(futures), 1):
            row, call = future.result()
            done[row["id"]] = row
            calls.append(call)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            call_log.write(json.dumps(call) + "\n")
            call_log.flush()
            if n % 50 == 0:
                ok = [c["wall_s"] for c in calls if c["outcome"] == "ok"]
                logging.info(
                    f"Completions: {n}/{len(todo)} -- failed "
                    f"{len(calls) - len(ok)}, median {_pct(ok, 50)}s, "
                    f"p95 {_pct(ok, 95)}s"
                )
    # ...then rewritten with one row per tool, the latest attempt winning.
    write_jsonl(path, done.values())
    return done, calls, time.perf_counter() - run_start


def _pct(values, q):
    """The q-th percentile (nearest rank), or None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    return round(
        ordered[min(len(ordered) - 1, max(0, round(q / 100 * len(ordered)) - 1))], 3
    )


def timing_summary(calls: list[dict], run_s: float, workers: int) -> dict:
    """Latency, throughput, tokens and errors of one run's calls."""
    ok = [c for c in calls if c["outcome"] == "ok"]

    def dist(key, rows):
        values = [c[key] for c in rows if c.get(key) is not None]
        if not values:
            return None
        return {
            "n": len(values),
            "mean": round(sum(values) / len(values), 3),
            "min": round(min(values), 3),
            **{f"p{q}": _pct(values, q) for q in (50, 90, 95, 99)},
            "max": round(max(values), 3),
        }

    def total(key):
        return sum(c.get(key) or 0 for c in ok)

    return {
        "calls": len(calls),
        "workers": workers,
        "run_wall_s": round(run_s, 1),
        "calls_per_min": round(len(calls) / run_s * 60, 2) if run_s else None,
        "outcomes": dict(Counter(c["outcome"] for c in calls)),
        "http_status": dict(Counter(str(c["http_status"]) for c in calls)),
        "retried_calls": sum(1 for c in calls if (c.get("attempts") or 1) > 1),
        "errors": dict(Counter(c["error"][:120] for c in calls if c["error"])),
        "wall_s_ok": dist("wall_s", ok),
        "latency_s_ok": dist("latency_s", ok),
        "wall_s_failed": dist("wall_s", [c for c in calls if c["outcome"] != "ok"]),
        "prompt_tokens": dist("prompt_tokens", ok),
        "completion_tokens": dist("completion_tokens", ok),
        "reasoning_tokens": dist("reasoning_tokens", ok),
        "completion_tokens_per_s": dist("completion_tokens_per_s", ok),
        "tokens_total": {
            "prompt": total("prompt_tokens"),
            "completion": total("completion_tokens"),
            "reasoning": total("reasoning_tokens"),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--env-file", default=".env")
    p.add_argument("--n-tools", type=int, default=500)
    p.add_argument("--seed", type=int, default=20261013)
    p.add_argument("--model", default="glm5.3-fp8", help="Gepeto model.")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=4, help="Concurrent LLM calls.")
    p.add_argument(
        "--timeout", type=float, default=300, help="Seconds per Gepeto HTTP attempt."
    )
    p.add_argument(
        "--fetch-workers", type=int, default=4, help="Concurrent README fetches."
    )
    p.add_argument(
        "--prompt-version",
        default=README_PROMPT_VERSION,
        choices=sorted(SYSTEM_PROMPTS),
        help=f"{README_PROMPT_VERSION}: with README (default); {PROMPT_VERSION}: the "
        "current prompt, no README fetched -- a same-model baseline.",
    )
    p.add_argument("--out-dir", type=Path, default=DATA / "readme_v2")
    args = p.parse_args()
    with_readme = args.prompt_version == README_PROMPT_VERSION

    load_dotenv(args.env_file, override=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ("urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger().handlers[0].addFilter(
        lambda r: not r.getMessage().startswith("Sending request to Gepeto")
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    credentials = Credentials.from_env().require(
        "gepeto_api_key", *(["github_token"] if with_readme else [])
    )
    repos = from_config(PipelineConfig.from_env())

    sample, stats = sample_tools(repos, args.n_tools, args.seed)
    (args.out_dir / "sample.json").write_text(
        json.dumps(
            {
                **vars(args),
                **provenance(),
                "out_dir": str(args.out_dir),
                **stats,
                "tool_ids": [str(s["tool"]["_id"]) for s in sample],
            },
            indent=2,
        )
    )
    logging.info(f"Sample: {stats}")

    readmes = (
        fetch_readmes(
            sample,
            GitHubClient(credentials.github_token),
            args.out_dir / "readmes.jsonl",
            args.fetch_workers,
        )
        if with_readme
        else {}
    )

    # The name filter's vocabulary, over every tool, as the stage computes it.
    vocab = vocabulary(
        tool.get("data")
        for tool in repos.tools.iter_projected(
            {}, VOCABULARY_PROJECTION, batch_size=2000
        )
    )
    requests = [
        build_request(
            entry["tool"],
            (readmes.get(str(entry["tool"]["_id"])) or {}).get("readme"),
            prompt_version=args.prompt_version,
            vocab=vocab,
        )
        for entry in sample
    ]
    write_jsonl(
        args.out_dir / "requests.jsonl",
        (
            {
                "id": r.tool_id,
                "tool_name": r.tool_name,
                "names": r.names,
                "input_hash": r.input_hash,
                "prompt_version": r.prompt_version,
                "messages": r.messages(),
                "json_schema": QUERY_SCHEMA,
            }
            for r in requests
        ),
    )

    # A record with no context at all would have been stored insufficient without
    # a call; none should reach here, since every sampled tool had a v1 document.
    askable = [r for r in requests if r.context]
    results, calls, run_s = generate(
        askable,
        GepetoClient(credentials.gepeto_api_key, timeout=args.timeout),
        args.model,
        args.temperature,
        args.out_dir,
        args.workers,
    )
    if calls:
        summary = timing_summary(calls, run_s, args.workers)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (args.out_dir / f"timing_summary_{stamp}.json").write_text(
            json.dumps(summary, indent=2)
        )
        print(json.dumps(summary, indent=2))

    counts = Counter()
    rows = []
    for entry, request in zip(sample, requests):
        tool_id = request.tool_id
        raw_readme = (readmes.get(tool_id) or {}).get("readme")
        cleaned = clean_readme(raw_readme)
        completion = results.get(tool_id) or {}
        result = (
            None
            if completion.get("error") or not completion
            else result_from_completion(request, completion["completion"])
        )
        status = result.status if result else "failed"
        counts[status] += 1
        counts["with_readme" if cleaned else "without_readme"] += 1
        stored = entry["v1"]
        rows.append(
            {
                "id": tool_id,
                "tool_name": request.tool_name,
                "repo": "/".join(entry["repo"]),
                "readme_raw_chars": len(raw_readme or ""),
                "readme_chars": len(cleaned or ""),
                "stored": {
                    "status": stored.get("status"),
                    "queries": stored.get("queries") or [],
                    "model": stored.get("model"),
                    "prompt_version": stored.get("prompt_version"),
                },
                "generated": {
                    "status": status,
                    "queries": result.queries if result else [],
                    "model": args.model,
                    "prompt_version": args.prompt_version,
                },
                "context": request.context,
            }
        )
    write_jsonl(args.out_dir / "comparison.jsonl", rows)

    stored_status = Counter(r["stored"]["status"] for r in rows)
    print(f"\nSample: {stats}")
    if with_readme:
        print(
            f"READMEs: {counts['with_readme']} usable, "
            f"{counts['without_readme']} none"
        )
    print(f"stored: {dict(stored_status)}")
    print(
        f"v{args.prompt_version} ({args.model}): generated={counts[STATUS_GENERATED]} "
        f"insufficient={counts['insufficient']} failed={counts['failed']}"
    )
    print(f"Written to {args.out_dir}")


if __name__ == "__main__":
    main()
