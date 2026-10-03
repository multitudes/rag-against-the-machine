# Caching

Bonus 4 of the subject: **cache the index and query results** so cold
start and repeated questions are cheaper.

`--cache` sits next to `--semantic` / `--hybrid` on `search`. Default
search stays uncached (moulinette / `search_dataset` unchanged).

```sh
uv run python -m src search "How to stop a worker?" --k 5 --cache
uv run python -m src search "How to stop a worker?" --k 5 --cache
```

The first call loads BM25 (and MiniLM if `--semantic` / `--hybrid`) and
writes `data/processed/query_cache.json`. The second call with the same
query, `k`, and flags reprints the sources **without** loading the
index.

---

## What is cached

### Query results (on disk)

Key = SHA-256 of `{query, k, semantic, hybrid, index fingerprint}`.

Value = the `MinimalSource` list (`file_path` + character offsets).

The fingerprint is size + mtime of `metadata.json`, `params.index.json`,
`embeddings.npy`, and `files.json`. After `index` (full or
incremental) the fingerprint changes, so old hits miss. `index` also
deletes `query_cache.json` so the file does not grow stale entries.

### Index (in process)

`get_cached_searcher()` keeps one `Searcher` per `index_dir` while the
fingerprint is unchanged. A cache **miss** on `search --cache` reuses
that instance instead of calling `BM25.load` again. A one-shot CLI
process only benefits on the query-file hit (no BM25 load at all). The
local HTTP API (`serve`, bonus 5) reuses that in-process Searcher
across requests.

---

## How to demo at the defence

1. `search "…" --k 5` (no flag) — watch BM25 load.
2. Same query with `--cache` — first time still loads, writes
   `query_cache.json`.
3. Repeat `--cache` — log line `Query cache hit` and an instant print.
4. Re-run `index`, then `--cache` again — miss, because the fingerprint
   changed.

### Tests (offline)

```sh
uv run pytest tests/test_cache.py -v
```
