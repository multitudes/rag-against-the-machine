# Ingestion

Ingestion is how we turn the vLLM tree into a searchable index. The
CLI command is `index` (not a separate `ingest` mode).

## What we do

1. **Read** files under `data/raw/vllm-0.10.1/` (`get_all_files`).
2. **Chunk** each file with chonkie (`chunk_content`) so a hit is a
   short, located span rather than a whole file.
3. **Index** the chunk texts with BM25 (`bm25s` + Snowball stemmer).
4. **Store** the sparse matrix, `corpus.jsonl`, and `metadata.json`
   under `data/processed/`.

With `--semantic` we also write MiniLM vectors (`embeddings.npy`)
aligned with the same metadata rows. We do **not** use ChromaDB.

## How we run it

```sh
make run
# same as:
uv run python -m src index --max_chunk_size 2000
```

Optional flags on the same command: `--semantic`, `--incremental`,
`--repo_path`, `--index_dir`.

There is no `mode="full"` / `mode="selective"` switch. The moulinette
expects a full corpus index. For local experiments we can point
`--repo_path` at a smaller folder.

## After ingestion

`search` and `search_dataset` load `data/processed/` and never walk
the raw tree again, until the next `index`.
