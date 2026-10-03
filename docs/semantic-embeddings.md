# Semantic embeddings

Bonus 1 of the subject: a **vector index** built with a lightweight CPU
model (`all-MiniLM-L6-v2`) **next to** the lexical BM25 index.

This document explains what embeddings are, how they retrieve, and how
to run the `--semantic` flag on `index` and `search`.

---

## What is an embedding?

An **embedding** is a list of numbers (a vector) that represents the
*meaning* of a piece of text.

Example (toy 3-D vectors, not real):

| Text | Vector (idea) |
|------|----------------|
| "kill a process" | `[0.21, 0.80, 0.11]` |
| "terminate a job" | `[0.19, 0.77, 0.09]` |
| "configure OpenAI server" | `[0.91, 0.05, 0.40]` |

The first two sentences share almost no keywords, but sit close in vector
space. The third is about something else, so it sits far away.

A **sentence-transformer** model (here: `all-MiniLM-L6-v2`) learned this
mapping from lots of text pairs: similar meanings → similar vectors.

Typical size for MiniLM: **384 dimensions** per chunk or query.

---

## How semantic retrieval works

Two phases, same as BM25 — but the “index” stores vectors, not term counts.

### 1. Index time (once, with `--semantic`)

For every chunk already produced by chonkie:

1. Feed `chunk.text` into MiniLM.
2. Get a 384-float vector.
3. Store the vector **together with the same metadata** BM25 uses
   (`file_path`, character offsets).

Saved as `data/processed/embeddings.npy`: shape `(n_chunks, 384)`.
Row `i` is the same chunk as `metadata.json[i]` / the BM25 corpus.

### 2. Query time (`search --semantic`)

1. Embed the question with the **same** model → one 384-float vector.
2. Compare it to every chunk vector with **cosine similarity**.
3. Return the top-k chunks whose vectors are closest to the query,
   as `file_path [start:end]` (same format as BM25 search).

```
query:  "How do I stop a running worker?"
         │
         ▼
    MiniLM embed
         │
         ▼
    q = [0.20, 0.79, …]          (384 floats)
         │
         ▼
    cosine(q, chunk_i)  for i in all chunks
         │
         ▼
    rank by similarity → top-k sources
```

**Cosine similarity** is 1.0 when two vectors point the same way (same
meaning) and ~0 when they are unrelated.

---

## How to run it

Default commands stay BM25 (moulinette / `search_dataset` unchanged).

```sh
# Install deps (sentence-transformers pulls PyTorch — large first sync)
uv sync

# Build BM25 + MiniLM vectors (CPU). First run downloads the model.
uv run python -m src index --max_chunk_size 2000 --semantic

# Single-query semantic search
uv run python -m src search "How to stop a worker?" --k 5 --semantic
```

Compare with lexical search (no flag):

```sh
uv run python -m src search "How to stop a worker?" --k 5
```

Hybrid (bonus 2) fuses both lists — see
[`hybrid-retrieval.md`](hybrid-retrieval.md):

```sh
uv run python -m src search "How to stop a worker?" --k 5 --hybrid
```

If you pass `--semantic` on `search` without having indexed with
`--semantic`, the CLI logs an error and returns no hits (no traceback):

```text
ERROR:retrieval.semantic:Semantic index not found at
data/processed/embeddings.npy. Re-run index with --semantic.
```

### What gets written

| File | Who |
|------|-----|
| BM25 `*.index.npy`, `corpus.jsonl`, `metadata.json`, … | always |
| `data/processed/embeddings.npy` | only with `index --semantic` |

Do not commit model weights or `embeddings.npy` (already under
`data/processed/`, gitignored).

### Tests (offline, model mocked)

```sh
make test
# or
uv run pytest tests/test_semantic.py -v
```

---

## Embeddings vs BM25

| | BM25 (lexical) | Embeddings (semantic) |
|---|---|---|
| Matches on | Shared **tokens** (after stemming) | Shared **meaning** |
| "kill process" vs "terminate job" | Weak / miss | Strong if the model knows they are similar |
| Exact identifier `PagedAttention` | Strong | Keyword search is often safer |
| CLI | `search` (default) | `search --semantic` |
| Graded by moulinette | **Yes** (`search_dataset`) | **No** (demo / bonus 1) |

They fail on **different** questions. Bonus 2 (`search --hybrid`) merges
both ranked lists with Reciprocal Rank Fusion. Semantic/hybrid search
must not replace BM25 for evaluation.

---

## The model: `all-MiniLM-L6-v2`

- Hugging Face
  [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- ~22M parameters, **CPU** (`device="cpu"`)
- 384 dimensions per text
- First run downloads weights into `~/.cache/huggingface/`
- Used only to encode chunks and queries — Qwen/Ollama still generate
  answers

Code: `src/retrieval/semantic.py` (`create_semantic_index`,
`search_semantic_ids`). Flag: `--semantic` on `index` and `search`.

---

## How to demo at the defence

1. Show BM25 on a paraphrased question (few shared keywords).
2. Same query with `--semantic`.
3. Same query with `--hybrid` (RRF of both lists).
4. Point at `embeddings.npy` next to `params.index.json`.
5. State that BM25 remains the default / moulinette path.

---

## Further reading

- [sentence-transformers MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [retrieving-methods.md](retrieving-methods.md) (TF-IDF vs BM25)
- [hybrid-retrieval.md](hybrid-retrieval.md) (bonus 2: RRF of BM25 + MiniLM)
- [incremental-indexing.md](incremental-indexing.md) (bonus 3: re-chunk changed files)
- Subject bonus 1: vector index *next to* the lexical index
