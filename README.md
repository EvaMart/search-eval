# search-eval

Experiments on the Software Observatory's search. The pipeline that builds the data
lives in [research-software-etl](https://github.com/inab/research-software-etl);
this repo samples tools, generates experimental variants, compares them, scores them
and plots the results.

**Split rule:** anything a pipeline stage or `rsetl` command runs stays in the pipeline.
Anything that samples, compares, scores or plots belongs here.

Two topics so far:

- **Retrieval** (current): which way of searching the Qdrant index finds the right
  tool, and how it compares with the Observatory's current search. A hybrid of four
  retrievers is ~2.4× better overall and ~16× better on task descriptions. See
  [Retrieval evaluation](#retrieval-evaluation).
- **Synthetic queries**: the search queries an LLM writes for each tool, used for
  semantic search. The questions are whether showing the LLM the tool's GitHub
  README (prompt v2) gives better queries than the record alone (prompt v1), and
  how much of any difference comes from the model. See [Datasets](#datasets) to
  [Results](#results).

- [Install](#install)
- [Retrieval evaluation](#retrieval-evaluation): the retrievers, the hybrid, the test queries and the results
- [Datasets](#datasets): which query sets exist, and how they were made
- [Prompt versions](#prompt-versions): exactly how v1, v2 and v2 revised differ
- [Experiments](#experiments): generating a new query set
- [Analyses](#analyses): what each analysis measures, its inputs and outputs
- [Results](#results)
- [Qdrant index](docs/qdrant.md): accessing the `tool_search` vector index and what is in it

## Install

The prompt, the context built from a tool and the query filters (`parse_queries`,
`build_request`, `clean_readme`, `vocabulary`) are imported from the pipeline,
never copied, so an evaluation measures exactly what the pipeline does.

```bash
pip install -e ../research-software-etl   # local checkout (development)
pip install -e .
# or, from git (pin the ref to a commit or tag first):
pip install -e ".[pipeline]"
```

Only the experiment script and `snapshot_queries` need credentials: MongoDB, Gepeto and GitHub. Copy
`.env.example` to `.env` and fill it in. The analyses only read files under `data/`.

The pipeline installs generic top-level packages (`application`, `domain`,
`infrastructure`, `adapters`). Don't create modules with those names here.

The embedding model is `Alibaba-NLP/gte-modernbert-base`, the same one the pipeline's
similarity stage uses. Once it's cached, `HF_HUB_OFFLINE=1` avoids reaching huggingface.co.

## Layout

```
src/search_eval/
├── common.py                         # paths, loading, record text, embeddings, neighbours, provenance, plot style
├── snapshot_queries.py               # copy one prompt version's queries from MongoDB (semantic_search) into data/
├── eval_queries.py                   # build the retrieval test queries (sample, names, tasks, finish)
├── retrieval/
│   ├── index.py                      # Qdrant client, encoder, Hit
│   ├── dense_description.py          # dense vectors of tool descriptions
│   ├── bm25_description.py           # BM25 on tool descriptions
│   ├── bm25_names.py                 # BM25 on names and labels (tool_search_exp_names), name matching
│   ├── dense_queries.py              # dense vectors of the synthetic queries, grouped by tool
│   ├── hybrid.py                     # RRF fusion of the four, with the name rule
│   └── observatory_api.py            # the current Observatory search, as a baseline
├── experiments/
│   └── readme_queries.py             # generate a query set online (Gepeto) for the 500-tool sample
└── analysis/
    ├── retriever_comparison.py       # score every retriever, the hybrids and the API on the test queries
    ├── query_source_similarity.py    # one query set, random sample of the whole corpus: query vs own record
    ├── query_versions_similarity.py  # several query sets on the 500-tool sample: query vs own record
    └── query_diversity.py            # several query sets on the 500-tool sample: queries of a tool vs each other
data/                                 # gitignored, see Datasets
```

Run any module with `python -m search_eval.<package>.<module>`, and add `--help` for its options.

## Retrieval evaluation

The full methodology, with the reasons for each choice, is in the
[methodology doc](https://claude.ai/artifact/Lg2MjkavtvYKStL8ZUFBnT). This is the summary.

### What is searched

The `tool_search` collection in Qdrant ([docs/qdrant.md](docs/qdrant.md)) holds, for
50,314 tools, a description point (dense vector and BM25) and 250,817 synthetic
query points (dense). `tool_search_exp_names` holds one point per tool with BM25 on
its name and labels (built by `rsetl search-index --names`). The dense model is
gte-modernbert-base.

### The retrievers and the hybrid

| Retriever | Searches | Catches |
|---|---|---|
| `dense_description` | description vectors | paraphrases of what a tool does |
| `bm25_description` | description words (BM25) | rare terms, formats, names in descriptions |
| `bm25_names` | names and labels (BM25, no stemming or stopwords) | lookups by name |
| `dense_queries` | synthetic query vectors, best per tool | how users phrase a task |

`hybrid` fuses the four with Reciprocal Rank Fusion: each takes its top 100 and a
tool scores Σ w / (60 + rank). Ranks only, so cosine and BM25 scores are never
compared. On top of that, the **name rule**:

- a `bm25_names` hit whose name *is* the query (exact after normalising, or near
  with ratio ≥ 0.9 when nothing is exact) gets weight 3, so "samtools" returns
  samtools before samtools_stats;
- when no hit matches, `bm25_names` gets weight 0: for a task query it only matches
  ordinary words. This is the default (`hybrid_4_adaptive` below).

```bash
python -m search_eval.retrieval.hybrid --show-ranks "remove haplotigs from a diploid assembly"
```

### Test queries

There were no real user queries, so `eval_queries` builds them for 300 tools sampled
uniformly from the index (seed 20261112), into `data/eval_queries/`:

- **596 name queries**, by rule: the exact name (300), a one-edit typo for names of
  6+ letters (236), a variant with separators as spaces and lowercase (60).
- **820 task queries** for 275 tools, written by glm5.3-fp8 from the tool's record:
  a keyword search, a short description and a question, never the name. A different
  model and prompt from the indexed synthetic queries.
- **Leakage check**: a task query's cosine to its own tool's closest indexed query
  has median 0.845 (0.741 for other tools); 3% are near-copies (≥ 0.95) and are also
  scored apart.

A query is answered if its tool, or a record with the same normalised name, is in
the top 10 (`--strict`: the tool only; it changes MRR by ≤ 0.011).

```bash
python -m search_eval.eval_queries sample --n 300 --seed 20261112
python -m search_eval.eval_queries names
python -m search_eval.eval_queries tasks
python -m search_eval.eval_queries finish
python -m search_eval.analysis.retriever_comparison            # and --strict
```

`retriever_comparison` caches each retriever's hits in `data/eval_queries/runs.jsonl`
and writes `data/eval/retriever_comparison_{lenient,strict}.{csv,json,png}`
(Recall@1/5/10 and MRR@10 per query type).

### Results, 2026-11-12

MRR@10 (1 = the right tool always first):

| System | All | Name queries | Task queries |
|---|---|---|---|
| **hybrid_4_adaptive (default)** | **0.628** | 0.790 | 0.511 |
| hybrid_4_boost (names never dropped) | 0.603 | 0.792 | 0.465 |
| hybrid_4 (no name rule) | 0.563 | 0.696 | 0.465 |
| hybrid_3 (no `bm25_names`) | 0.526 | 0.548 | 0.510 |
| bm25_description | 0.511 | 0.676 | 0.391 |
| dense_description | 0.424 | 0.525 | 0.350 |
| dense_queries | 0.355 | 0.126 | 0.521 |
| bm25_names | 0.330 | 0.728 | 0.041 |
| observatory_api (current search) | 0.264 | 0.585 | 0.031 |

Conclusions:

- **Each retriever has a blind spot.** `dense_queries` is best on tasks but can't find
  names (the synthetic queries never contain them); `bm25_names` is the reverse.
- **The hybrid keeps both strengths.** With the name rule it is best on names and
  within 0.01 of the best single retriever on tasks.
- **Both halves of the name rule matter.** Boosting matches lifts name queries
  (0.696 → 0.792); dropping the names list when nothing matches lifts task queries
  (0.465 → 0.511) at no cost on names.
- **Against the current search**: 2.4× overall, 35% better on names and 16× on task
  queries (0.031 → 0.511).

Limits: the queries are synthetic (LLM- and rule-made), and only the tool and its
same-name records count as right, so equally good alternatives count as misses.
Both affect every system alike; the next step is to re-test on real Observatory
search queries.

## Datasets

All under `data/`. There are five query sets: two for the full
corpus, and three for a fixed 500-tool sample.

**Data files.** Most of `data/` is too big for git and is ignored. Only the small
reference files are tracked (about 1.4 MB; the exceptions are in `.gitignore`):

- every `dataset.json` and `sample.json`, plus `results.dataset.json`;
- the retrieval test queries: `eval_queries/{queries,name_queries,task_queries}.jsonl`;
- the analysis summaries: `eval/*.json`, `eval/*.png`, `eval/retriever_comparison_*.csv`.

The query sets, `query_requests.jsonl` and `eval_queries/runs.jsonl` are not in
git. The embedding cache and the large per-query CSVs are rebuilt by running the
analyses again.

| Query set | Where | Prompt | Model | Run on | Tools | Generated |
|---|---|---|---|---|---|---|
| **v1 stored (Qwen)** | `results.jsonl` | v1 | Qwen3.6-27B | MN5 (batch job) | 49,134 (whole corpus) | before 2026-10-08 |
| **v1 GLM** | `v1_glm/` | v1 | glm5.3-fp8 | Gepeto (online) | the 500-tool sample | 2026-10-13 |
| **v2 GLM** | `readme_v2/` | v2 (first version) | glm5.3-fp8 | Gepeto (online) | the 500-tool sample | 2026-10-13 |
| **v2 revised GLM** | `readme_v2_rev/` | v2 revised | glm5.3-fp8 | Gepeto (online) | the 500-tool sample | 2026-10-13 |
| **v2 revised (Qwen)** | `v2_rev_qwen/` | v2 revised | Qwen3.6-27B | MN5 (batch job) | 50,247 (whole corpus) | 2026-10-16 |

Each set has a metadata file: `dataset.json` in its folder, or
`data/results.dataset.json` for the stored set, whose files sit at the top of
`data/`. It holds a plain-language description, the prompt and prompt version,
the model, provider and parameters, the date, tool counts, and the exact system
prompt sent. Write one for every new set.

Comparisons between pairs of sets:

- v1 stored vs v1 GLM: the **model** changes, the prompt stays the same.
- v1 GLM vs v2 GLM / v2 revised GLM: the **prompt and the README** change, the model stays the same.
- v2 GLM vs v2 revised GLM: only the **v2 revision** changes. See [Prompt versions](#prompt-versions).
- v1 stored vs v2 revised GLM: both the model and the prompt change. This is the
  comparison made so far, see [Results](#results).

### The full corpus: v1 stored (Qwen)

The pipeline produced these files, and they are the queries in MongoDB today.
They're refreshed by a new export or MN5 run.

- `query_requests.jsonl`: output of `rsetl synthetic-queries export`, holding every
  tool's v1 prompt and context (49,134 tools). The analyses also take each record's
  text from here, and the set of records used for neighbours and random controls.
- `results.jsonl`: output of the MN5 job for that export, holding the stored v1 queries.
- `query_requests_test1000.jsonl`: a 1000-tool test export. No analysis uses it.

### The full corpus: v2 revised (Qwen)

The v2 revised prompt run on MN5 for the 2026-10-15 export (49,154 tools, 8,263 of
them with a README excerpt), with the same model as v1 stored. So v1 stored vs this
set changes only the prompt, plus the README where there is one. These are the
queries in MongoDB today: every tool in `toolsDev` has exactly one prompt v2
document. `v2_rev_qwen/` holds two files:

- `results.jsonl`: a snapshot of the `semantic_search` collection (50,247
  documents, 2026-10-27), made with `python -m search_eval.snapshot_queries
  --prompt-version 2 --out-dir data/v2_rev_qwen` (it reads MongoDB credentials
  from `.env`). The queries are stored as imported, already through `parse_queries`.
- `mn5_results.jsonl`: the MN5 job's raw output that was imported, copied from the
  pipeline repo (`data/semantic_search/results.jsonl`, 49,154 completions, no errors).

Of the 50,247 tools, 44,882 have queries (8,089 of them with a README). The other
5,365 are stored as `insufficient` with no queries: 1,093 were too thin to prompt,
so they were never exported, and for 4,272 the model's answer left no usable query.
`dataset.json` has the counts.

An earlier MN5 run was incomplete (45,056 completions), and its import skipped
every tool with a README. Both were superseded on 2026-10-16. That run's output is
kept in the pipeline repo, under `data/semantic_search/v2-incomplete/`.

### The 500-tool sample: v1 GLM, v2 GLM, v2 revised GLM

`experiments/readme_queries` generated these three sets. They all use **the same
500 tools** (seed 20261013, identical `tool_ids` in every `sample.json`), so they
can be compared tool by tool, with each other and with the stored v1 queries of
the same tools.

A tool is eligible if it has exactly one GitHub repository, no other tool shares
that repository, and it already has a stored v1 document. The shared-repository
rule exists because a hub's README, such as ucscgenomebrowser/kent's, describes
the hub rather than the tool. Of 10,303 tools with a GitHub repository, 8,750 are
eligible. All three runs used temperature 0.5.

Each folder holds:

| File | Contents |
|---|---|
| `sample.json` | the sampled tool ids and the run's parameters |
| `readmes.jsonl` | raw README per tool, `null` when there is none (v2 runs only; identical in both) |
| `requests.jsonl` | the prompts sent, in the same shape as `rsetl synthetic-queries export` |
| `results.jsonl` | the completions, in the same shape as the MN5 output. **This is the query set the analyses read** |
| `comparison.jsonl` | per tool: the context, the stored v1 queries and this run's queries side by side, for reading by hand |
| `calls.jsonl` | one row per Gepeto call (timing, tokens, errors). Rows are appended, never rewritten |
| `timing_summary_<UTC>.json` | latency percentiles, throughput, tokens |

`data/<set>.log` is each run's console log. The `out_dir` in `sample.json` says
`data/semantic_search/...` because the runs happened in the pipeline repo before
the files moved here.

Tools that got queries, out of 500:

| v1 stored (Qwen) | v1 GLM | v2 GLM | v2 revised GLM |
|---|---|---|---|
| 476 | 491 | 495 | 496 |

**Do not `rsetl synthetic-queries import` these files.** That would replace the
stored v1 queries in MongoDB.

**Telling the sets apart.** The folder is the only reliable label. Both v2 runs
record `prompt_version: "2"`, so the field doesn't separate them. These
`sample.json` files also predate the provenance fields (`pipeline_version`,
`pipeline_commit`). `requests.jsonl` holds the exact prompt each run sent.

### Analysis outputs: `data/eval/`

CSV, JSON summary and PNG per analysis run. The names are
`<analysis>_n<tools>_seed<seed>.*` and **don't include the query sets**, so running
an analysis again with other sets but the same `n` and seed overwrites the
previous output. Each JSON lists the sets it used and the pipeline provenance.

| File stem | Analysis | Query sets |
|---|---|---|
| `query_source_similarity_n50_seed20261009` | query_source_similarity | v1 stored (Qwen), 50 tools sampled from the whole corpus |
| `query_versions_similarity_n500_seed20261013` | query_versions_similarity | v1 stored (Qwen) vs v2 revised GLM, 500-tool sample |
| `query_diversity_n500_seed20261013` | query_diversity | v1 stored (Qwen) vs v2 revised GLM, 500-tool sample |
| `query_versions_similarity_n49134_seed20261013` | query_versions_similarity | v1 stored (Qwen), v2 revised Mongo and MN5 output (Qwen), whole corpus (`--all-tools`) |
| `query_diversity_n49134_seed20261013` | query_diversity | same three sets, whole corpus (`--all-tools`) |

`record_embeddings_<model>_<hash>.npy` is the cached embedding of every record's text
(~49k records, minutes to compute). Every analysis uses it to find nearest
neighbours. It's rebuilt automatically when `query_requests.jsonl` or the model
changes, and it has to stay in `--out-dir`, otherwise it's recomputed.

## Prompt versions

The prompts are defined in the pipeline
(`application/services/semantic_search/synthetic_queries.py`: `SYSTEM_PROMPTS`,
`build_tool_context`, `clean_readme`). Every version asks for up to 6 queries of
3-15 words, never containing the tool's name.

- **v1**: the tool record only, meaning its description (or Galaxy help text) and
  its EDAM topics, operations and formats. The stored queries and v1 GLM use the
  same v1 prompt.
- **v2**: v1 plus a cleaned excerpt of the GitHub README, at most 3000 characters,
  and one rule added to the prompt: *use the README for specifics (data types, file
  formats, methods, organisms, use cases); ignore installation steps, licensing,
  citation, badges and other boilerplate.*
- **v2 revised**: v2 with three changes.
  1. A length hint: "3-15 words each, *mostly 4-10, as a researcher would type them*".
  2. One more rule: *do not base queries on example datasets, benchmark figures,
     version numbers or implementation internals*.
  3. A stricter `clean_readme`, which also drops sections such as Documentation
     and Run Tests. This changed the context for 66 of the 500 tools, shortening
     those excerpts by ~16% on average. The raw READMEs are the same in both runs.

The current pipeline (`README_PROMPT_VERSION = "2"`) is v2 revised. The first v2
exists only in `readme_v2/requests.jsonl`.

## Experiments

`experiments/readme_queries` generates one query set for the 500-tool sample,
online through Gepeto. It never writes to MongoDB.

```bash
# v2 (current README prompt) with the default model
python -m search_eval.experiments.readme_queries --out-dir data/<new_set>
# v1 prompt (no README) with the same model, as a baseline
python -m search_eval.experiments.readme_queries --prompt-version 1 --out-dir data/<new_set>
# another model
python -m search_eval.experiments.readme_queries --model <gepeto-model> --out-dir data/<new_set>
```

The prompt it uses is whatever the installed pipeline defines for that version.
To try a new prompt, change it in the pipeline checkout and run into a **new**
`--out-dir`. Rerunning into an existing folder resumes it: only failed calls are
retried, and completions already there are kept. Keep the same `--seed` (default
20261013) so the set lines up with the others.

Then add a `dataset.json` to the folder (copy one from another set), add the set
to the [Datasets](#datasets) table here, and compare it with the existing sets
(see below).

## Analyses

Every analysis embeds queries and record texts with gte-modernbert-base and uses
cosine similarity. Queries are the **stored** ones: each completion goes through
the pipeline's `parse_queries`, as `rsetl synthetic-queries import` does, so
queries containing the tool's name and similar are dropped. The **record text** is
the description block of the tool's *v1* context, or its EDAM terms when there is
no description. It never includes the README. The tool name is left out.

### `query_versions_similarity`: do the queries stay close to their own record?

Use this to compare query sets on the 500-tool sample. Each query is scored against:

- **same**: its own tool's record text.
- **neighbour**: the 5 records most similar to the tool's own. These are hard
  negatives: same topic, different tool, which is what search has to tell apart.
- **random**: one fixed set of 500 random records, shared by every query of
  every set. These are easy negatives.

Grounded queries put "same" well above "random". How far "same" sits above
"neighbour" shows how *specific* they are. Reported per set: medians, the AUC of
same vs random and same vs neighbour, and the share of queries whose own record
beats all 5 neighbours.

```bash
python -m search_eval.analysis.query_versions_similarity   # default: v1 stored vs v2 revised GLM
python -m search_eval.analysis.query_versions_similarity \
    --set "v1 stored (Qwen)=data/results.jsonl" \
    --set "v1 GLM=data/v1_glm/results.jsonl" \
    --set "v2 GLM=data/readme_v2/results.jsonl" \
    --set "v2 revised GLM=data/readme_v2_rev/results.jsonl"
```

`--set LABEL=RESULTS` can be repeated, and the label is what appears in the plot
and summary. `--sample` picks the tool list, which defaults to
`data/readme_v2_rev/sample.json`, the same 500 tools as the other two.
`--all-tools` scores every tool in `--requests` (the whole corpus, 49,134) instead:

```bash
python -m search_eval.analysis.query_versions_similarity --all-tools \
    --set "v1 stored (Qwen)=data/results.jsonl" \
    --set "v2 revised, Mongo (Qwen)=data/v2_rev_qwen/results.jsonl" \
    --set "v2 revised, MN5 output (Qwen)=data/v2_rev_qwen/mn5_results.jsonl"
```

On the whole corpus with three sets, this takes ~20 minutes and ~3 GB of RAM on a
laptop, and the CSV is ~0.9 GB. A results file can be a MN5 output or a
`snapshot_queries` file.

Because the record text never includes the README, queries built on README-only
details score *lower* here even when they're correct. A drop for v2 means the
queries drift from the description, which isn't necessarily a bad thing.

### `query_diversity`: are a tool's queries different from each other?

Use this to compare query sets on the 500-tool sample. For every pair of queries
of the *same* tool it computes their cosine ("within"). As a baseline it takes the
same number of pairs from *different* tools ("between"). High within-tool
similarity means paraphrases. Pairs at or above `--dup-threshold` (0.9) count as
near-duplicates. Per set it reports medians, the share of near-duplicate pairs and
the share of tools with one, plus a paired comparison on tools that have queries
in every set.

```bash
python -m search_eval.analysis.query_diversity              # default: v1 stored vs v2 revised GLM
python -m search_eval.analysis.query_diversity --set "v1 GLM=data/v1_glm/results.jsonl" --set ...
```

Same `--set`, `--sample` and `--all-tools` options as above.

### `query_source_similarity`: one query set, sampled from the whole corpus

This is the same same / neighbour / random scoring, but for **one** query set
and a random sample drawn from that set's own tools. It's the analysis behind the
hand review of 2026-10-09. By default it uses v1 stored (Qwen), 50 tools, seed
20261009: 45 tools with queries plus 5 the LLM answered with none. The random
controls are 5 per query, drawn fresh, not a fixed 500.

```bash
python -m search_eval.analysis.query_source_similarity                        # the hand-reviewed sample
python -m search_eval.analysis.query_source_similarity --n-tools 500 --seed 7 # a bigger sample
```

`--results` accepts any set, but for comparing sets use `query_versions_similarity`
instead. This analysis samples tools per set, so two runs wouldn't cover the same
tools. Also, the output name has no set label, so a run with the default `n` and
seed overwrites the v1 output.

## Results

### Prompt v2 revised vs stored v1, 2026-10-13

Sets: **v1 stored (Qwen)** vs **v2 revised GLM**, on the 500-tool sample. Both the
model and the prompt change, so these numbers don't yet say which of the two
causes a difference.

Reading `comparison.jsonl` by hand: v2 picks up details only the README has, such
as file formats, supported engines and input types. No invented capabilities
turned up when checked against the context.

**Query vs record** (`query_versions_similarity`). The record text is the
description only, without the README.

| | v1 stored (Qwen) | v2 revised GLM |
|---|---|---|
| Median cosine to own record | 0.77 | 0.74 |
| Median cosine to 5 nearest neighbours | 0.72 | 0.71 |
| Median cosine to 500 random records | 0.54 | 0.54 |
| AUC, own record vs random | 0.988 | 0.976 |
| AUC, own record vs neighbours | 0.688 | 0.629 |
| Own record beats all 5 neighbours | 48% | 37% |

v2 queries drift a little from the description and are harder to tell apart from
neighbouring tools *by description*. The README details they add can't score here.

**Diversity within a tool** (`query_diversity`)

| | v1 stored (Qwen) | v2 revised GLM |
|---|---|---|
| Median same-tool pair cosine | 0.76 | 0.75 |
| Median different-tool pair cosine | 0.57 | 0.56 |
| Same-tool pairs ≥ 0.9 (near-duplicates) | 5.3% | 1.3% |
| Tools with a near-duplicate pair | 39% | 14% |

The medians barely move, but v2 has far fewer near-duplicate queries.

### Prompt v2 revised vs stored v1, whole corpus, same model, 2026-10-16

Sets: **v1 stored (Qwen)** vs **v2 revised (Qwen)**, both Qwen3.6-27B on MN5, over
the whole corpus (`--all-tools`, 49,134 records). Only the prompt changes, plus the
README where a tool has one. These numbers come from the **earlier, incomplete**
data: the MN5 output was the 45,056-completion run (now in the pipeline repo's
`data/semantic_search/v2-incomplete/`), and the MongoDB set came from the partial
import that had no README tools (see [the dataset](#the-full-corpus-v2-revised-qwen)).
They haven't been recomputed on the complete set. On tools without a README, the
MongoDB set gave the same numbers as the MN5 output.

**Whole sets** (each on its own tools)

| | v1 stored | v2 revised, MN5 output | v2 revised, MongoDB |
|---|---|---|---|
| Tools with queries | 45,562 | 41,016 | 33,833 |
| Queries | 261,872 | 226,454 | 184,428 |
| Median cosine to own record | 0.787 | 0.779 | 0.787 |
| Median cosine to 5 nearest neighbours | 0.727 | 0.723 | 0.725 |
| Median cosine to 500 random records | 0.537 | 0.533 | 0.532 |
| AUC, own record vs random | 0.990 | 0.987 | 0.990 |
| AUC, own record vs neighbours | 0.706 | 0.693 | 0.711 |
| Own record beats all 5 neighbours | 55% | 53% | 56% |
| Median same-tool pair cosine | 0.77 | 0.76 | 0.76 |
| Same-tool pairs ≥ 0.9 (near-duplicates) | 5.1% | 4.1% | 4.6% |
| Tools with a near-duplicate pair | 38% | 32% | 35% |

**Paired, v1 stored vs v2 revised MN5 output, tools with queries in both** (40,593),
split by whether the v2 context had a README:

| | README, v1 | README, v2 | no README, v1 | no README, v2 |
|---|---|---|---|---|
| Tools | 7,037 | 7,037 | 33,556 | 33,556 |
| Median cosine to own record | 0.776 | 0.746 | 0.789 | 0.787 |
| Median cosine to neighbours | 0.729 | 0.711 | 0.728 | 0.725 |
| AUC, own record vs neighbours | 0.674 | 0.623 | 0.710 | 0.710 |
| Own record beats all 5 neighbours | 46% | 38% | 56% | 56% |
| Median per-tool mean pair cosine | 0.766 | 0.735 | 0.768 | 0.766 |

Without a README, the revised prompt barely changes anything: the scores are the
same, and there are slightly fewer near-duplicates. All of the movement is on the 7k README tools.
Their queries move away from the description (own record 0.776 → 0.746, beats
all neighbours 46% → 38%) and become more diverse (0.766 → 0.735). This is the
same direction as the 500-tool GLM sample, at the scale of the corpus and with
the model held fixed. So it comes from the README, not from GLM. As before, a
lower score against the description doesn't show the queries are worse. The README
details they add can't score here.

The paired README split was computed from the two CSVs. It isn't part of either
analysis's JSON.

### Open questions

- **Model or prompt?** v1 GLM and v2 GLM exist but haven't been analysed yet. Run
  both analyses with all four sets (the `--set` example above) to separate the
  model change (Qwen to GLM) from the prompt and README change.
- **Did the v2 revision help?** That needs v2 GLM vs v2 revised GLM, which hasn't been run yet.
- **Scoring README-based queries.** Every score here uses the description as the
  record text. It can't reward a correct query built on README details. Retrieval
  against a real search index would.
