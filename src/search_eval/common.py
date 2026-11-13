"""
What the analyses share: data paths, loading the pipeline's request/result files,
record text, embeddings, neighbours, provenance and the plot style.

The prompt, the context format and the query filters are the pipeline's
(research-software-etl, `application.services.semantic_search.synthetic_queries`);
nothing here re-implements them.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from importlib import metadata
from pathlib import Path

import numpy as np

from application.services.semantic_search.synthetic_queries import parse_queries

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EVAL = DATA / "eval"
PIPELINE_DIST = "research-software-observatory-pipeline"
DEFAULT_MODEL = "Alibaba-NLP/gte-modernbert-base"  # same as the similarity stage

# Plot style: surface, ink, secondary ink, muted, grid, axis; categorical order.
SURFACE, INK, INK2, MUTED, GRID, AXIS = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7")
BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd0"

# Context lines that follow the description block (build_tool_context).
_TERM_LINE = re.compile(r"^(Topics|Operations|Input formats|Output formats): ")


def source_text(context: str) -> tuple[str, str]:
    """The description block of an LLM context, or its EDAM terms when it has none."""
    desc, terms, in_desc = [], [], False
    for line in context.split("\n"):
        if line == "Description:":
            in_desc = True
        elif _TERM_LINE.match(line):
            in_desc = False
            terms.append(line)
        elif in_desc:
            desc.append(line)
    if desc:
        return "\n".join(desc).strip(), "description"
    return "\n".join(terms).strip(), "edam_terms"


def load(results_path: Path, requests_path: Path):
    """(completions by id, (record text, kind) by id, tool name by id)."""
    completions = {}
    for line in results_path.open():
        r = json.loads(line)
        completions[r["id"]] = r
    sources, names = {}, {}
    for line in requests_path.open():
        r = json.loads(line)
        context = r["messages"][1]["content"]
        if context:
            sources[r["id"]] = source_text(context)
            names[r["id"]] = r["tool_name"]
    return completions, sources, names


def read_completions(path: Path, tool_ids: set[str]) -> dict[str, list[str]]:
    """
    The stored queries of these tools: each completion through `parse_queries`.

    A snapshot of MongoDB documents (`snapshot_queries`) holds queries that
    already went through it at import, so those are taken as they are.
    """
    out = {}
    for line in path.open():
        r = json.loads(line)
        if "tool_id" in r:
            if r["tool_id"] in tool_ids:
                out[r["tool_id"]] = r.get("queries") or []
        elif r["id"] in tool_ids:
            out[r["id"]] = parse_queries(r["completion"], r["names"]) or []
    return out


def all_tool_ids(requests_path: Path) -> list[str]:
    """Every tool of an export that has a context: the whole corpus, sorted."""
    ids = []
    for line in requests_path.open():
        r = json.loads(line)
        if r["messages"][1]["content"]:
            ids.append(r["id"])
    return sorted(ids)


def parse_sets(sets: list[str] | None, default: list[str]) -> list[list[str]]:
    """`LABEL=RESULTS` arguments as [label, path] pairs."""
    return [s.split("=", 1) for s in (sets or default)]


def select_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name: str):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name, device=select_device())
    # As in the similarity stage (compute_embeddings.load_embedder).
    model.max_seq_length = 512
    return model


def record_embeddings(model, model_name: str, sources: dict, cache_dir: Path):
    """
    Embeddings of every record's source text, as (ids, matrix).

    Nearest neighbours need the whole corpus (~49k records, minutes on a laptop),
    so the matrix is cached, keyed on the model and a hash of every id and text:
    a changed requests file or model recomputes it.
    """
    ids = sorted(sources)
    texts = [sources[i][0] for i in ids]
    h = hashlib.sha256(model_name.encode())
    for i, t in zip(ids, texts):
        h.update(i.encode() + b"\0" + t.encode() + b"\0")
    key = h.hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"record_embeddings_{model_name.split('/')[-1]}_{key}.npy"
    if path.exists():
        return ids, np.load(path)
    matrix = model.encode(texts, normalize_embeddings=True, batch_size=32,
                          show_progress_bar=True, convert_to_numpy=True)
    matrix = matrix.astype(np.float32)
    np.save(path, matrix)
    return ids, matrix


def nearest_neighbours(tool_id: str, ids: list[str], matrix: np.ndarray,
                       sources: dict, k: int) -> list[str]:
    """
    The k records whose source text is closest to this tool's own.

    Records with the *same* text as the tool are skipped: they are copies of it
    (the same description under another record), not a different tool, and would
    score as "same" by construction.
    """
    index = {i: n for n, i in enumerate(ids)}
    own = sources[tool_id][0].strip().lower()
    sims = matrix @ matrix[index[tool_id]]
    out = []
    for n in np.argsort(-sims):
        other = ids[n]
        if other != tool_id and sources[other][0].strip().lower() != own:
            out.append(other)
            if len(out) == k:
                break
    return out


def nearest_neighbours_many(tool_ids: list[str], ids: list[str], matrix: np.ndarray,
                            sources: dict, k: int, batch: int = 512) -> dict[str, list[str]]:
    """
    `nearest_neighbours` for many tools at once: one matrix product per batch,
    and a sort of only the top candidates. Falls back to the full sort for a tool
    with too many copies of its own text among them.
    """
    index = {i: n for n, i in enumerate(ids)}
    texts = [sources[i][0].strip().lower() for i in ids]
    m = min(len(ids), k + 64)
    out = {}
    for start in range(0, len(tool_ids), batch):
        chunk = tool_ids[start:start + batch]
        rows = [index[t] for t in chunk]
        sims = matrix[rows] @ matrix.T
        top = np.argpartition(-sims, m - 1, axis=1)[:, :m]
        for t, row, s, cand in zip(chunk, rows, sims, top):
            found = [n for n in cand[np.argsort(-s[cand])]
                     if n != row and texts[n] != texts[row]][:k]
            if len(found) < k:
                found = [index[o] for o in nearest_neighbours(t, ids, matrix, sources, k)]
            out[t] = [ids[n] for n in found]
    return out


def provenance() -> dict:
    """
    Which pipeline code produced the queries' parsing and prompts.

    The package version, plus the git commit (and whether the tree was dirty)
    when the pipeline is installed from a checkout, so every summary says which
    prompt and parser it measured.
    """
    import application

    out = {"pipeline_version": metadata.version(PIPELINE_DIST)}
    src = Path(application.__file__).resolve().parent
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=src, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=src,
                               capture_output=True, text=True, check=True).stdout
        out.update(pipeline_commit=sha, pipeline_dirty=bool(dirty.strip()))
    except (OSError, subprocess.CalledProcessError):
        try:  # a pip install from git records the commit here
            direct = json.loads(metadata.distribution(PIPELINE_DIST)
                                .read_text("direct_url.json") or "{}")
            out["pipeline_commit"] = direct.get("vcs_info", {}).get("commit_id")
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    return out


def style_axis(ax) -> None:
    """Recessive axes: no frame but the baseline, a light y grid."""
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
