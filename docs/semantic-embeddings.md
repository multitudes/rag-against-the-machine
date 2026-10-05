# Semantic embeddings

Bonus 1 of the subject: a **vector index** built with a lightweight CPU
model (`all-MiniLM-L6-v2`) **next to** the lexical BM25 index.

What embeddings are, how we retrieve with them, and how we run
`--semantic` on `index` and `search`.

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

Each chunk becomes **one** vector of 384 floats, no matter how long
the text is. `2000` is only our max character length (moulinette).
A 40-character heading and a 2000-character page both become 384
numbers. Length is not the vector size. That is why
`embeddings.npy` is `(n_chunks, 384)`: one row per chunk, aligned
with BM25.

---

## Tokens, the 256-token window, and the 384 floats

The 384 floats are **not** “one number per token.” Tokens are the
model’s **input**. The 384 numbers are the **output**: one compressed
meaning vector for the whole chunk.

**1. Tokenize the chunk.** MiniLM does not read characters. A
tokenizer cuts the text into subword pieces (often a word, sometimes
`##ing`, `v`, `##LLM`):

```
"How do I stop a worker?"
  → [CLS] how do i stop a worker ? [SEP]
```

`[CLS]` / `[SEP]` are bookends the model always adds.

**2. 256-token window.** `all-MiniLM-L6-v2` only looks at the
**first 256 tokens** of that list (including the bookends). Extra
tokens are dropped. A short chunk fits. A dense 2000-character chunk
can be longer than 256 tokens, so the **tail of the chunk is
invisible** to MiniLM. BM25 still indexed the whole span; the vector
only “saw” the prefix.

**3. One hidden vector per token, then collapse.** The transformer
turns **each** token into a 384-float vector (same width as the
model). Internally we briefly have something like `(256, 384)` — a
stack of per-token meanings.

Then **mean pooling**: average those token vectors (usually ignoring
padding) into **one** 384-vector. We also L2-normalise it so cosine
is just a dot product.

```txt
chunk text
    → tokens (≤ 256)
    → 256 × 384 hidden states
    → average
    → 1 × 384  = one row in embeddings.npy
```

**What the floats correspond to.** Nothing we can name. Dimension 17
is not “verbs” or “vLLM.” They are **learned axes** in a meaning
space: training pushed similar sentences together and different ones
apart. We only care that two 384-lists that point the same way are
about the same thing.

So: tokens = how the model *reads*; 384 floats = one fingerprint of
*what it understood* from (at most) those 256 tokens.

English prose is often **~4–5 characters per token**, so 256 tokens
is roughly **1000 characters**. Code and markdown are usually denser
(punctuation, identifiers, URLs), so a 2000-character Python chunk
can blow past the window more easily. That makes 2000 a bit long for
MiniLM: the vector is a fingerprint of the **prefix**, and the tail
is dropped. **1000 would fit semantic search better**, but it is not
a free upgrade for hybrid. Hybrid ranks the **same** chunk list as
BM25 — we do not keep a 2000-char lexical index and a 1000-char
vector index. `2000` is the moulinette **ceiling** (any longer span
is rejected), and our graded Recall@5 was measured on that index.
Re-chunking the default path to 1000 would change BM25 IDF and split
some official spans, so we would need to re-run the moulinette on
docs and code before calling it a win. On the current 2000-char
chunks, hybrid already papers over the mismatch: BM25 still sees the
whole span (including the tail MiniLM never read), and RRF keeps a
hit if either list ranks it well. We keep **2000 for `make run` /
defence**. A 1000-char index is only a side experiment
(`index --max_chunk_size 1000 --semantic`, then `search --hybrid`)
until those recall numbers are checked.

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

If `search --semantic` runs before `index --semantic`, the CLI logs
an error and returns no hits (no traceback):

```text
ERROR:retrieval.semantic:Semantic index not found at
data/processed/embeddings.npy. Re-run index with --semantic.
```

### What gets written

| File | Who |
|------|-----|
| BM25 `*.index.npy`, `corpus.jsonl`, `metadata.json`, … | always |
| `data/processed/embeddings.npy` | only with `index --semantic` |

We do not commit model weights or `embeddings.npy` (already under
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
- 384 dimensions per text (after mean pooling; 256-token window)
- First run downloads weights into `~/.cache/huggingface/`
- Used only to encode chunks and queries — Qwen/Ollama still generate
  answers

Code: `src/retrieval/semantic.py` (`create_semantic_index`,
`search_semantic_ids`). Flag: `--semantic` on `index` and `search`.

---

## Defence notes

We show BM25 on a paraphrased question (few shared keywords), then
the same query with `--semantic` and `--hybrid`. `embeddings.npy`
sits next to `params.index.json`. BM25 stays the default / moulinette
path.

---

## Further reading

- [sentence-transformers MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [retrieving-methods.md](retrieving-methods.md) (TF-IDF vs BM25)
- [hybrid-retrieval.md](hybrid-retrieval.md) (bonus 2: RRF of BM25 + MiniLM)
- [rrf.md](rrf.md) (`rrf_fuse`: ranks, not raw scores)
- [bonus.md](bonus.md) (all five flags, index vs search)
- [incremental-indexing.md](incremental-indexing.md) (bonus 3: re-chunk changed files)
- [caching.md](caching.md) (bonus 4: index + query cache)
- [http-api.md](http-api.md) (bonus 5: local HTTP API)
- Subject bonus 1: vector index *next to* the lexical index
