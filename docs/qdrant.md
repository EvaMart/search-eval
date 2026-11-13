# Qdrant: the `tool_search` vector index

The semantic-search index lives in a Qdrant collection, `tool_search`. It is built
by the pipeline (`rsetl search-index` in research-software-etl) from the `tools` and
`semantic_search` MongoDB collections. This repo **reads** it. Don't write to it from
here (see [Experiments](#experiments-own-collections)).

## Access

Qdrant (v1.20) runs on a remote host and is reached through an SSH tunnel to local
port 6333. While the tunnel is up:

- REST API: `http://localhost:6333`
- Dashboard: `http://localhost:6333/dashboard`

Every request needs the API key. Put it in `.env`, with the same variable names the
pipeline uses:

```bash
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=...            # ask the maintainer; never commit it
```

```python
import os
from qdrant_client import QdrantClient

client = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    api_key=os.getenv("QDRANT_API_KEY"),
    cloud_inference=True,  # needed to send text for BM25 (see below)
    timeout=60,  # the 5 s default can trip on the first query after a while idle
)
```

Two things to watch for:

- **Always pass `url`.** With an API key and `url=None`, `qdrant-client` defaults to
  HTTPS. The tunnel speaks plain HTTP, so that fails with
  `SSL: WRONG_VERSION_NUMBER`.
- The client warns `Api key is used with an insecure connection`. That's expected
  over the local tunnel.

If requests start failing with `Connection refused` or `Connection reset by peer`,
the tunnel is down.

## What is in it

Snapshot of 2026-11-04 (prompt version 2 queries, `Qwen3.6-27B`):

| `kind` | Points | One per | Vectors |
|---|---|---|---|
| `description` | 50,314 | tool (every tool in `toolsDev`) | `dense` + `bm25` |
| `synthetic_query` | 250,817 | stored synthetic query | `dense` only |

Tools whose queries are `insufficient` in `semantic_search` (about 5k) have a
description point but no query points.

### Vectors

- **`dense`**: 768 dimensions, cosine, L2-normalised, from
  `Alibaba-NLP/gte-modernbert-base` (the same model as `common.DEFAULT_MODEL` and the
  similarity stage). No query/document prefix is used. Encode search queries the same
  way: `model.encode(q, normalize_embeddings=True)`. Inputs are capped at 512 tokens.
- **`bm25`**: sparse, IDF modifier, **computed by Qdrant itself** (`qdrant/bm25`,
  English stemming and stopwords). Send text, not tokens:
  `models.Document(text=q, model="qdrant/bm25")`. Only description points have it.

### What text each point was embedded from

**Description point, `dense`:**

```
<preferred_label or name>: <longest distinct description, max 2000 chars>
Topics: <EDAM topic terms>.
Operations: <EDAM operation terms>.
Input: <EDAM input format terms>.
Output: <EDAM output format terms>.
```

If there is no description, the `help` documentation text is used instead. Empty
lines are left out. Caveat: 2,000 characters is close to the 512-token cap, so for
the longest descriptions the EDAM lines at the end may be truncated away.

**Description point, `bm25`:** every name the tool goes by (preferred label, name,
labels, other names), every distinct description, then the same EDAM lines.

**Query point, `dense`:** the query text alone.

The exact code is `description_dense_text`, `description_sparse_text` and
`tool_points` in the pipeline's
`application/services/semantic_search/search_index.py`. Import them rather than
copying them, as with the other pipeline functions.

### Payload

| Field | On | Meaning |
|---|---|---|
| `tool_id` | all | the tool's `_id` in `toolsDev`, as a string (`ObjectId(tool_id)` to join with MongoDB) |
| `kind` | all | `description` or `synthetic_query` |
| `name` | all | display name (`preferred_label`, else `name`) |
| `type` | all | the tool's `data.type` list, e.g. `["cmd"]`, `["web", "lib"]` |
| `sources` | all | registries the tool was merged from (`data.source`), e.g. `["bioconda_recipes", "biotools", "github"]` |
| `text` | all | the dense text above (for a query point, the query) |
| `prompt_version` | queries | prompt that wrote the query (`"2"`) |
| `model` | queries | LLM that wrote the query |
| `text_hash`, `payload_hash` | all | used by the pipeline for incremental updates; ignore them |

Keyword payload indexes exist on `tool_id`, `kind`, `type`, `sources` and
`prompt_version`, so filtering and grouping on those are fast.

Point ids are UUIDs derived from the tool id and the text. Don't key anything on
them; use `tool_id`.

## Querying

A tool has several points, so group the results by `tool_id`. Hybrid search
(dense over both kinds, plus BM25 over descriptions, fused with RRF):

```python
from qdrant_client import models
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Alibaba-NLP/gte-modernbert-base")
q = "remove haplotigs from a diploid assembly"
vector = model.encode(q, normalize_embeddings=True).tolist()

result = client.query_points_groups(
    "tool_search",
    prefetch=[
        models.Prefetch(query=vector, using="dense", limit=200),
        models.Prefetch(
            query=models.Document(text=q, model="qdrant/bm25"),
            using="bm25",
            limit=200,
        ),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    group_by="tool_id",
    group_size=3,  # best-matching points per tool, to see which text matched
    limit=10,
    with_payload=["name", "kind", "text"],
)
for group in result.groups:
    print(group.id, [(h.payload["kind"], round(h.score, 3)) for h in group.hits])
```

Variants, for comparing retrieval strategies:

- **Queries only / descriptions only:** add
  `filter=models.Filter(must=[models.FieldCondition(key="kind", match=models.MatchValue(value="synthetic_query"))])`
  to the dense `Prefetch` (or to `query_points_groups` itself, without prefetch).
- **By tool type or registry:** `FieldCondition(key="type", match=models.MatchAny(any=["cmd", "lib"]))`,
  and the same with `key="sources"`.
- **BM25 alone:** `client.query_points_groups("tool_search", query=models.Document(text=q, model="qdrant/bm25"), using="bm25", group_by="tool_id", limit=10)`.

## Keeping it current

The pipeline updates the collection with `rsetl search-index`, which is incremental:
it re-embeds only changed text, rewrites changed payloads and deletes points of
retired tools and replaced queries. It is not yet a stage of `rsetl run`, so the
index is as fresh as its last manual run. Results can change between runs; record
`client.count("tool_search")` and the date alongside any evaluation result.

## Experiments: own collections

To try a different embedding model, a different description text, or v1 queries,
build a **separate** collection (e.g. `tool_search_exp_<name>`) rather than
changing `tool_search`, which is the one the search uses. Reuse the pipeline's
point builders where possible, so the only difference is the thing being tested.
Delete experiment collections when done.
