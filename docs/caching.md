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

`lookup_query` never looks at `_searchers`. It is a **different**
cache: `query_cache.json` on disk.

`search --cache` does:

1. **`lookup_query`** — “did we already print this exact search?” If
   yes, return sources and **never** construct a Searcher.
2. On a miss, **`get_cached_searcher`** — that is the dict of live
   Searchers.
3. After a real search, **`store_query`** writes the sources under
   the same key.

The SHA-256 is not “prove this Searcher is still valid.” It is the
**JSON key** in that file. `_entry_key` hashes
`{query, k, semantic, hybrid, fp}` because those five things must
all match or the stored `MinimalSource` list is the wrong answer:

| Change | Same query string, but… |
|---|---|
| `k=5` vs `k=10` | Different list length |
| `--semantic` vs BM25 | Different ranking |
| `--hybrid` vs not | Different ranking |
| `fp` after `index` | Chunks / offsets changed |

`fp` is the index fingerprint: size + mtime of `metadata.json`,
`params.index.json`, `embeddings.npy`, and `files.json`. Baking it
into the key means a rewrite cannot hit an old entry: the SHA
changes, `entries.get(key)` is a miss, we search again.

If we keyed only on the query text, “How to stop a worker?” after a
re-index would still return yesterday’s offsets. `index` also
deletes `query_cache.json` so the file does not grow stale entries.

Each JSON entry looks like:

```json
{
  "entries": {
    "<sha256>": {
      "query": "...",
      "k": 5,
      "semantic": false,
      "hybrid": false,
      "fp": "metadata.json:…|params.index.json:…",
      "sources": [
        { "file_path": "...", "start": 0, "end": 80 }
      ]
    }
  }
}
```

**Lookup only needs `sources`.** `lookup_query` hashes the current
`{query, k, semantic, hybrid, fp}`, does `entries.get(key)`, then
validates `sources`. It never reads `fp` (or `query` / `k` / flags)
out of the value.

So `fp` in the body is **not required for correctness**. The SHA
already binds the fingerprint: after `index`, the key changes and
that entry is unreachable. The extra fields are for humans: the key
is opaque hex, and opening the file in defence shows *which*
question, `k`, flags, and index snapshot produced that list. We
could store `{"sources": [...]}` only and the cache would still
work.

`get_cached_searcher` uses the same fingerprint as a **string
compare** (`cached[0] == fingerprint`), not as a SHA. Two jobs, two
places.

### Index (in process)

`_searchers` is `index_dir → (fingerprint, Searcher)`. After we
construct a `Searcher` on a `--cache` miss, `get_cached_searcher()`
stores it there. Without `--cache` we still construct one and drop it
when the process exits — nothing is stored.

Default search uses only `data/processed`, so the dict has **0 or 1**
entries:

| What you ran | What is in `_searchers` |
|---|---|
| `search` (no `--cache`) | Nothing. |
| `search --cache` **hit** | Still nothing (or whatever was there). We never construct a Searcher. |
| `search --cache` **miss** | One Searcher, keyed by that folder. |

A second `--cache` miss **in the same process** reuses that object
instead of `BM25.load` again. `uv run python -m src search …` then
exit throws the dict away, so the CLI barely benefits from Searcher
reuse — the query-file hit is what skips the load. `serve` keeps the
process alive, so that one Searcher stays warm across requests.

Two different `index_dir`s in one process would be two keys. We do
not do that in the Makefile / defence flow.

---

## Defence notes

We run `search "…" --k 5` without a flag (BM25 load), then the same
query with `--cache` (still loads, writes `query_cache.json`). A
repeat `--cache` logs `Query cache hit` and prints immediately.
After `index`, `--cache` misses because the fingerprint changed.

### Tests (offline)

```sh
uv run pytest tests/test_cache.py -v
```
