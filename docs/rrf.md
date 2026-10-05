# Reciprocal Rank Fusion (`rrf_fuse`)

Bonus 2 helper in `src/retrieval/semantic.py`. It takes two (or more)
ranked lists of **chunk ids** and returns **one** top-k list. Hybrid
search (`search --hybrid`) calls it on the BM25 ranking and the MiniLM
ranking.

We do **not** mix BM25 scores with cosine scores. Those numbers live
on different scales (4.4 vs 0.82). We only use **rank**: 1st, 2nd,
3rd.

Default `search` / `search_dataset` / the moulinette never go through
this function.

How we run hybrid is in [hybrid-retrieval.md](hybrid-retrieval.md).

---

## The formula

RRF gives each chunk:

$$
\mathrm{score}(d) = \sum_{r \in R} \frac{1}{k_{\mathrm{rrf}} + \mathrm{rank}_r(d)}
$$

- $R$ is the set of rankings (here: BM25 and MiniLM).
- $\mathrm{rank}_r(d)$ is the 1-based position in that list. If the
  chunk is missing from a list, that list adds nothing.
- $k_{\mathrm{rrf}} = 60$ (`RRF_K` in `src/core/config.py`).

A chunk both retrievers like beats a chunk only one likes, even if
that one was #1.

---

## Worked example

| Rank | BM25 | MiniLM |
|------|------|--------|
| 1 | chunk A | chunk C |
| 2 | chunk B | chunk A |
| 3 | chunk C | chunk D |

- A: $1/(60+1) + 1/(60+2) = 1/61 + 1/62$ → in **both**, high
- C: $1/63 + 1/61$
- B: only BM25 #2 → $1/62$
- D: only MiniLM #3 → $1/63$

`rrf_fuse` sums those, sorts unique ids by the sum (best first), and
cuts to `k`.

---

## What the function does

```python
rrf_fuse(ranked_lists, k, rrf_k=RRF_K) -> list[int]
```

- `ranked_lists`: each inner list is chunk ids, best first.
- `k`: how many fused ids to return.
- `rrf_k`: smoothing in $1 / (k_{\mathrm{rrf}} + \mathrm{rank})$.

Hybrid asks each method for a **pool** of `max(k, HYBRID_POOL)`
candidates (default 20) so a chunk just outside one list’s final
top-k can still be promoted.

---

## Tests

```sh
uv run pytest tests/test_hybrid.py tests/test_semantic.py -v
```
