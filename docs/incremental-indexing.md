# Incremental indexing

Bonus 3 of the subject: when a file changes, **re-index only that file**
instead of chunking the whole corpus again.

Default `index` stays a full rebuild (Makefile / moulinette). Search is
unchanged.

```sh
uv run python -m src index --max_chunk_size 2000 --semantic
uv run python -m src index --max_chunk_size 2000 --semantic --incremental
```

`--incremental` sits next to `--semantic` on `index`.

---

## What actually updates

`bm25s` has no “replace one document” API, and BM25 IDF depends on the
whole corpus. After a file changes we still **rewrite** the sparse
matrix from the merged chunk list.

The win is skipping **chonkie** (and MiniLM encode) on ~2 700 unchanged
files. tqdm then shows `Chunking changed files: 1` instead of
`Chunking files: 2734`.

---

## How it works

1. Every successful `index` (full or incremental) writes `files.json`
   next to `metadata.json`: path → `{mtime_ns, size}` plus the
   `max_chunk_size` used.
2. `--incremental` diffs the current walk against that manifest.
3. Added / changed files are re-chunked. Deleted files drop their
   chunks. Unchanged files reuse text + offsets from
   `corpus.jsonl` + `metadata.json`.
4. BM25 is rebuilt from the merge. If `embeddings.npy` already exists,
   MiniLM rows for unchanged files are copied and only new chunk texts
   are encoded.

Fallback to a **full** index (and a log line) when:

- there is no index / no `files.json`
- `metadata.json` and `corpus.jsonl` disagree
- `--max_chunk_size` differs from the last run

If nothing changed, the CLI prints `Index already up to date` and does
not rewrite the matrix.

---

## How to demo at the defence

1. Full index once (`index --semantic`).
2. Edit one file under `data/raw/vllm-0.10.1/` (or `touch` it).
3. Re-run `index --semantic --incremental`.
4. Point at the log: `Incremental: 1 changed, 0 added, 0 deleted, …`
   and a short tqdm over changed files only.
5. `search` still finds both the edited file and the rest of the corpus.

### Tests (offline)

```sh
uv run pytest tests/test_incremental.py -v
```
