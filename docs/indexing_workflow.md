Perfect — I have the full picture. Here's the complete indexing walkthrough:

---

## Indexing — full call chain

```
make run
  └─▶ uv run python -m src index --max_chunk_size 2000
        └─▶ RagCLI.index()              (__main__.py)
              ├─▶ get_all_files()        (file_processing.py)
              ├─▶ chunk_content()        (chunking.py)   ← called per file
              └─▶ create_bm25_index()   (indexing.py)
```

---

### Stage 1 — `get_all_files()` `file_processing.py`

Walks `data/raw/vllm-0.10.1/` with `os.walk`. It:
- **Skips** hidden dirs (`.github`, `.buildkite` …) and `__pycache__`, `node_modules`, `.git`
- **Skips** hidden files (starting with `.`)
- Returns a flat `list[str]` of ~2 734 file paths

---

### Stage 2 — `chunk_content()` `chunking.py` ← called for every file

For each file it does three things:

**1. Detect the extension** and skip binaries immediately via `IGNORE_EXTENSIONS` (`.pdf`, `.so`, `.png` …)

**2. Pick the right chunker** based on extension:

```
.py .pyi .cpp .h .sh .js …  →  CodeChunker    (splits on AST boundaries)
.md .html .rst              →  RecursiveChunker (splits on headings/paragraphs)
.txt .yaml .json .toml …   →  SentenceChunker  (splits on sentences, with overlap)
anything else               →  skip (return [])
```

**3. Build `ChunkSource` objects** — each chunk gets:
- `text` — the raw text slice
- `source.file_path` — exact path as stored (e.g. `data/raw/vllm-0.10.1/vllm/engine/llm_engine.py`)
- `source.first_character_index` / `last_character_index` — byte offsets in the original file

Then `_enforce_max_size()` runs as a hard safety net: if any chunk is still > 2000 chars (e.g. a massive function that `CodeChunker` couldn't split), it slices it by character while keeping the correct offsets.

---

### Stage 3 — `create_bm25_index()` `indexing.py`

Takes the full `list[ChunkSource]` (~16 000 chunks) and:

1. **Saves** `metadata.json` → array of `{file_path, first_character_index, last_character_index}` — this is how file positions are recovered at search time
2. **Tokenises** all chunk texts: lowercase → remove English stopwords → Snowball stem (e.g. `configuring` → `configur`)
3. **Builds** the BM25 sparse matrix with `bm25s.BM25()`
4. **Saves** the index to `data/processed/` — several files: the matrix, vocab, stopwords, corpus

---

### What lives in `data/processed/` after indexing

```
data/processed/
├── metadata.json        ← file_path + char ranges for every chunk
├── corpus.jsonl         ← raw text of every chunk
├── vocab.index.json     ← stemmed token → id mapping
├── stopwords.tokenizer.json
└── *.npy                ← the sparse BM25 matrix (scores)
```

At search time, `Searcher` loads all of this back and uses `metadata.json` to map a BM25 result index back to a real file location.
