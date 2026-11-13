"""
Snapshot the queries stored in MongoDB (`semantic_search`) into a query set under data/.

The analyses only read files, so a set that lives in MongoDB is first copied here.
Every document of one prompt version is written as stored (the queries already went
through `parse_queries` at import), so a later import does not change the snapshot.
Documents of other prompt versions are left out and counted in dataset.json:
after a partial import the collection mixes versions.

Writes to --out-dir:
    results.jsonl   one stored document per line ({tool_id, tool_name, status,
                    queries, input_hash, prompt_version, model, generated_at});
                    read by the analyses like an MN5 results file
    dataset.json    counts by prompt version / model / status, and the
                    pipeline's system prompt for that version. Fill in the
                    description and the rest by hand, as for the other sets.

Usage:
    python -m search_eval.snapshot_queries --prompt-version 2 --out-dir data/v2_rev_qwen
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from application.services.semantic_search.synthetic_queries import SYSTEM_PROMPTS
from infrastructure.config import PipelineConfig
from infrastructure.db.repositories import from_config

from search_eval.common import provenance


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--prompt-version", required=True, choices=sorted(SYSTEM_PROMPTS))
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    load_dotenv()
    repos = from_config(PipelineConfig.from_env())
    coll = repos.semantic_search
    docs = coll.db_adapter.find(coll.collection_name, {}, projection={"_id": 0},
                                batch_size=5000)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts, kept = Counter(), 0
    with (args.out_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for doc in docs:
            counts[(doc.get("prompt_version"), doc.get("model"), doc.get("status"))] += 1
            if doc.get("prompt_version") == args.prompt_version:
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
                kept += 1

    meta = {
        "name": "",
        "description": "",
        "prompt_version": args.prompt_version,
        "source": f"MongoDB collection `{coll.collection_name}`, documents with "
                  f"prompt_version {args.prompt_version!r}",
        "snapshot_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generated_by": "python -m search_eval.snapshot_queries "
                        f"--prompt-version {args.prompt_version} --out-dir {args.out_dir}",
        "files": {"results": "results.jsonl"},
        "counts": {
            "documents_in_collection": sum(counts.values()),
            "documents_kept": kept,
            "by_prompt_version_model_status": [
                {"prompt_version": v, "model": m, "status": s, "n": n}
                for (v, m, s), n in counts.most_common()],
        },
        **provenance(),
        "system_prompt": SYSTEM_PROMPTS[args.prompt_version],
    }
    (args.out_dir / "dataset.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta["counts"], indent=2))
    print(f"wrote {args.out_dir}/results.jsonl / dataset.json")


if __name__ == "__main__":
    main()
