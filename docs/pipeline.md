# Pipeline

Our RAG system has two phases: **index once**, then **search or answer**
as many times as needed.

## Phase 1 — Ingestion (`index`)

We prepare the knowledge base from `data/raw/vllm-0.10.1/`.

1. Walk the corpus (`get_all_files`) and skip binaries / hidden dirs.
2. Split each file with chonkie (`chunk_content`) into `ChunkSource`
   rows (text + `file_path` + character offsets).
3. Build the **BM25** index (`create_bm25_index`) and write it under
   `data/processed/` (`metadata.json`, sparse matrix, corpus).
4. Optionally, with `--semantic`, embed the same chunks with MiniLM
   and save `embeddings.npy` next to the BM25 files.

This is the slow step (a few minutes for the full tree). We re-run it
when the corpus or `--max_chunk_size` changes, or we use
`--incremental` to re-chunk only files that changed.

## Phase 2 — Retrieval and generation

This is what `search`, `search_dataset`, `answer`, and `serve` do.

1. Tokenise the question the same way as the corpus (stem + stopwords).
2. Rank chunks with BM25 (default), MiniLM (`--semantic`), or both
   fused (`--hybrid`).
3. Map chunk ids through `metadata.json` to `MinimalSource` locations.
4. For answers, read those character spans from the original files,
   put them in the prompt, and call Qwen via Ollama.

The two indexes are separate: BM25 is lexical, MiniLM is semantic.
Hybrid retrieval (bonus 2) merges the two ranked lists. The moulinette
grades the BM25 path (`search_dataset`).

See also [indexing_workflow.md](indexing_workflow.md),
[ingestion.md](ingestion.md), and [bonus.md](bonus.md) (the five
optional flags).
