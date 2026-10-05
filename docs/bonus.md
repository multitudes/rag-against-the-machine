# Subject bonuses

The mandatory path is BM25 `index` + `search` / `search_dataset`.
The five bonuses are extra flags on that pipeline. The moulinette
still grades default BM25 retrieval.

---

## Map

| Bonus | Flag | When it runs |
|-------|------|----------------|
| 1 Semantic | `--semantic` | **Index** writes `embeddings.npy`. **Search** ranks with MiniLM. |
| 2 Hybrid | `--hybrid` | **Search** only (RRF of BM25 + MiniLM). Needs bonus 1’s `.npy`. |
| 3 Incremental | `--incremental` | **Index** only: re-chunk files that changed. |
| 4 Cache | `--cache` | **Search** (and `serve`): reuse index + query hits. |
| 5 HTTP API | `serve` | Local `/health`, `/search`, `/answer`. |

Bonus 1 is **both** index and search. Bonus 2 is search, but it
needs a previous `index --semantic`. Bonus 3 is index only.

```sh
# 1 — MiniLM next to BM25
uv run python -m src index --max_chunk_size 2000 --semantic
uv run python -m src search "How to stop a worker?" --k 5 --semantic

# 2 — fuse the two ranked lists
uv run python -m src search "How to stop a worker?" --k 5 --hybrid

# 3 — after a first index, re-chunk only what changed
uv run python -m src index --max_chunk_size 2000 --semantic --incremental

# 4 — second identical query skips BM25 load
uv run python -m src search "How to stop a worker?" --k 5 --cache

# 5
uv run python -m src serve --host 127.0.0.1 --port 8000
```

If `--hybrid` and `--semantic` are both set, **hybrid wins**.

---

## Bonus 4: two caches, one flag

1. **Query cache** (`lookup_query`) — same question, `k`, and flags
   already in `query_cache.json`? Print those sources and **return**
   (no BM25 load).
2. **Searcher cache** (`get_cached_searcher`) — miss: reuse a Searcher
   already loaded in this process instead of `BM25.load` again.

Without `--cache` we always construct `Searcher(index_dir=...)`.

After a real search (a miss) we `store_query(...)` so the next
identical call can hit step 1. A hit never reaches `store_query`.

`index` calls `clear_index_caches` after a rewrite so those files
cannot point at old chunks. Details: [caching.md](caching.md).

---

## Further reading

- [semantic-embeddings.md](semantic-embeddings.md) (bonus 1)
- [hybrid-retrieval.md](hybrid-retrieval.md) / [rrf.md](rrf.md) (bonus 2)
- [incremental-indexing.md](incremental-indexing.md) (bonus 3)
- [caching.md](caching.md) (bonus 4)
- [http-api.md](http-api.md) (bonus 5)
