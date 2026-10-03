# Hybrid retrieval (BM25 + MiniLM)

Bonus 2 of the subject: **fuse** the lexical BM25 ranking with the MiniLM
semantic ranking into one top-k list.

`--semantic` stays MiniLM-only. `--hybrid` is the combo. Default search
stays BM25 (moulinette / `search_dataset` unchanged).

If both `--hybrid` and `--semantic` are passed, **hybrid wins**.

---

## Why fuse two lists?

BM25 is strong when the question shares **identifiers** with the source
(`vllm`, function names, exact tokens). MiniLM is strong on
**paraphrases** (“stop a worker” vs “kill the process”).

They fail on different questions. Reciprocal Rank Fusion (RRF) keeps
chunks that either method likes, and boosts chunks that **both** like.

---

## Reciprocal Rank Fusion

Each retriever produces an ordered list of chunk ids (best first).
RRF scores a chunk as:

\[
\text{score}(d) = \sum_{r \in R} \frac{1}{k_{\text{rrf}} + \text{rank}_r(d)}
\]

- \(R\) is the set of rankings (here: BM25 and MiniLM).
- \(\text{rank}_r(d)\) is 1-based position in that list (missing → skip).
- \(k_{\text{rrf}} = 60\) (standard smoothing; see `RRF_K` in
  `src/core/config.py`).

A chunk ranked #1 by both lists beats a chunk ranked #1 by only one.
The fused list is unique ids sorted by this score, truncated to `k`.

Each method is asked for a **pool** of `max(k, HYBRID_POOL)` candidates
(default pool 20) so RRF can promote items that sat just outside the
final top-k of one list.

---

## How to run it

Index with `--semantic` first so `embeddings.npy` exists (same as bonus
1). Then search with `--hybrid`:

```sh
uv run python -m src index --max_chunk_size 2000 --semantic

uv run python -m src search "How to stop a worker?" --k 5
uv run python -m src search "How to stop a worker?" --k 5 --semantic
uv run python -m src search "How to stop a worker?" --k 5 --hybrid
```

If `embeddings.npy` is missing, `--hybrid` logs the semantic error and
falls back to BM25-only fusion (still prints hits).

`search_dataset` / `evaluate` stay on BM25 until we A/B Recall@5.

### Tests (offline, model mocked)

```sh
uv run pytest tests/test_hybrid.py tests/test_semantic.py -v
```
