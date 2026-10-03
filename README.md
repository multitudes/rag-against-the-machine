[![42](https://img.shields.io/badge/-Berlin-blue.svg?logo=data:image/svg%2bxml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4NCjwhLS0gR2VuZXJhdG9yOiBBZG9iZSBJbGx1c3RyYXRvciAxOC4xLjAsIFNWRyBFeHBvcnQgUGx1Zy1JbiAuIFNWRyBWZXJzaW9uOiA2LjAwIEJ1aWxkIDApICAtLT4NCjxzdmcgdmVyc2lvbj0iMS4xIg0KCSBpZD0iQ2FscXVlXzEiIHNvZGlwb2RpOmRvY25hbWU9IjQyX2xvZ28uc3ZnIiBpbmtzY2FwZTp2ZXJzaW9uPSIwLjQ4LjIgcjk4MTkiIHhtbG5zOnJkZj0iaHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyIgeG1sbnM6c3ZnPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6c29kaXBvZGk9Imh0dHA6Ly9zb2RpcG9kaS5zb3VyY2Vmb3JnZS5uZXQvRFREL3NvZGlwb2RpLTAuZHRkIiB4bWxuczpkYz0iaHR0cDovL3B1cmwub3JnL2RjL2VsZW1lbnRzLzEuMS8iIHhtbG5zOmNjPSJodHRwOi8vY3JlYXRpdmVjb21tb25zLm9yZy9ucyMiIHhtbG5zOmlua3NjYXBlPSJodHRwOi8vd3d3Lmlua3NjYXBlLm9yZy9uYW1lc3BhY2VzL2lua3NjYXBlIg0KCSB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB4PSIwcHgiIHk9IjBweCIgdmlld0JveD0iMCAtMjAwIDk2MCA5NjAiDQoJIGVuYWJsZS1iYWNrZ3JvdW5kPSJuZXcgMCAtMjAwIDk2MCA5NjAiIHhtbDpzcGFjZT0icHJlc2VydmUiPg0KPHBvbHlnb24gaWQ9InBvbHlnb241IiBwb2ludHM9IjMyLDQxMi42IDM2Mi4xLDQxMi42IDM2Mi4xLDU3OCA1MjYuOCw1NzggNTI2LjgsMjc5LjEgMTk3LjMsMjc5LjEgNTI2LjgsLTUxLjEgMzYyLjEsLTUxLjEgDQoJMzIsMjc5LjEgIi8+DQo8cG9seWdvbiBpZD0icG9seWdvbjciIHBvaW50cz0iNTk3LjksMTE0LjIgNzYyLjcsLTUxLjEgNTk3LjksLTUxLjEgIi8+DQo8cG9seWdvbiBpZD0icG9seWdvbjkiIHBvaW50cz0iNzYyLjcsMTE0LjIgNTk3LjksMjc5LjEgNTk3LjksNDQzLjkgNzYyLjcsNDQzLjkgNzYyLjcsMjc5LjEgOTI4LDExNC4yIDkyOCwtNTEuMSA3NjIuNywtNTEuMSAiLz4NCjxwb2x5Z29uIGlkPSJwb2x5Z29uMTEiIHBvaW50cz0iOTI4LDI3OS4xIDc2Mi43LDQ0My45IDkyOCw0NDMuOSAiLz4NCjwvc3ZnPg0K)](https://42berlin.de) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![version](https://img.shields.io/badge/version-1.0.0-blue) [![tests](https://github.com/multitudes/rage-against-the-machine/actions/workflows/test.yml/badge.svg?branch=dev)](https://github.com/multitudes/rage-against-the-machine/actions/workflows/test.yml) 

*This project has been created as part of the 42 curriculum by lbrusa.*

# RAG Against the Machine

## Description

**RAG Against the Machine** is a Retrieval-Augmented Generation (RAG) system that answers
questions about the [vLLM](https://github.com/vllm-project/vllm) codebase.

A language model only knows what it was trained on. Instead of retraining it every time
the codebase changes, RAG lets the model reach out to the right source files at answer
time. This project implements the full four-stage pipeline:

1. **Indexing** — chunk every file in the vLLM repository and persist a searchable index
2. **Retrieval** — given a question, find the top-k most relevant snippets using BM25
3. **Augmentation** — pass those snippets as context inside the model's prompt
4. **Generation** — produce a grounded natural-language answer with Qwen/Qwen3-0.6B

Retrieval quality is measured with **recall@k**: the share of reference sources that
appear in the top-k results. The system targets ≥ 80 % recall@5 on documentation
questions and ≥ 50 % on code questions.

---

## System Architecture

```txt
data/raw/vllm-0.10.1/
        │
        ▼
┌─────────────────────┐
│   File Processing   │  get_all_files() — walks the corpus, filters extensions
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│      Chunking       │  chunk_content() — CodeChunker / RecursiveChunker /
│   (src/ingestion)   │  SentenceChunker depending on file type
└────────┬────────────┘
         │  List[ChunkSource]
         ▼
┌─────────────────────┐
│   BM25 Indexing     │  create_bm25_index() — tokenises, stems, builds
│  (bm25s + stemmer)  │  sparse matrix; saves to data/processed/
└────────┬────────────┘
         │
         ▼  (query)
┌─────────────────────┐
│     Retrieval       │  Searcher.search_one() / search_dataset()
│   (src/retrieval)   │  returns MinimalSearchResults (file_path + char range)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Context Assembly   │  reads character slices from the original files
└────────┬────────────┘
         │  context string
         ▼
┌─────────────────────┐
│  Answer Generation  │  Qwen/Qwen3-0.6B via Ollama
│  (src/answering)    │  returns MinimalAnswer (sources + answer text)
└─────────────────────┘
```

All data models between stages are validated with **Pydantic** (`src/core/schemas.py`).
The CLI is built with **Python Fire** and every long-running step has a **tqdm** progress bar.

---

## Chunking Strategy

Files are dispatched to three different chunkers based on their extension:

| File type | Extensions | Chunker | Rationale |
|---|---|---|---|
| Python / C++ / JS / Shell | `.py`, `.pyi`, `.cu`, `.cpp`, `.h`, `.sh`, `.js`, … | `chonkie.CodeChunker` | Splits on AST node boundaries so chunks do not break in the middle of a function or class |
| Markdown / HTML / RST | `.md`, `.html`, `.rst` | `chonkie.RecursiveChunker` | Respects heading and paragraph structure so a doc chunk maps to one logical section |
| Plain text / TOML / YAML / JSON | `.txt`, `.toml`, `.yaml`, `.json`, `.env`, … | `chonkie.SentenceChunker` | Splits on sentence boundaries with configurable overlap |

Binary and non-text files (`.pdf`, `.zip`, `.so`, `.png`, …) are skipped entirely.

**Chunk size** is configurable via `--max_chunk_size` (default 2000 characters, hard
maximum enforced by the moulinette). Chunks never exceed this limit; shorter chunks
are always valid. Using smaller chunks (e.g. 500–1000 chars) increases recall but
also increases index size and retrieval noise — see *Performance Analysis* below.

Character indices (`first_character_index`, `last_character_index`) are stored
verbatim from chonkie and matched against the original file bytes at retrieval time.

---

## Retrieval Method

Retrieval uses **BM25** ([BM25S](https://github.com/xhluca/bm25s) implementation)
with an English stemmer (PyStemmer / Snowball).

### Why BM25?

BM25 is a classic probabilistic term-frequency model. It ranks documents by how
often query terms appear in them, adjusted for document length and corpus-wide term
frequency. It requires no GPU, indexes in seconds, and retrieves in milliseconds —
making it ideal for the 5-minute indexing and 90-second throughput constraints.

### Pipeline

1. **Tokenise** the corpus at index time: lowercase, remove English stopwords, stem
   with Snowball. Tokens and the sparse BM25 matrix are persisted under `data/processed/`.
2. **At query time**: apply the same tokenisation to the question, call
   `bm25s.BM25.retrieve()` for top-k results.
3. **Return** `MinimalSearchResults`: for each hit, store the `file_path` and the
   `(first_character_index, last_character_index)` from the metadata JSON.

The index is loaded with `mmap=True` to reduce RAM usage on large corpora.

---

## Instructions

We manage the project with [uv](https://docs.astral.sh/uv/). Answer
generation needs [Ollama](https://ollama.com/) with `qwen3:0.6b`.
Python 3.12 is what we pin locally (`.python-version`); 3.10+ is the
subject floor.

```sh
# uv, if the machine does not have it yet
curl -LsSf https://astral.sh/uv/install.sh | sh

# model weights for answer generation
ollama pull qwen3:0.6b
```

### Installation

```sh
make install
# or: uv sync
```

### Full Pipeline

```sh
# 1. Build the index (~2–5 min)
make run
# equivalent: uv run python -m src index --max_chunk_size 2000

# 2. Search the docs dataset
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions

# 3. Score with the moulinette (our search JSON first, then ground truth)
# see docs/moulinette.md
./moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 --max_context_length 2000

# 4. Generate answers (Ollama must be running)
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

### All CLI Commands

```
uv run python -m src index          [--max_chunk_size INT] [--repo_path PATH] [--index_dir PATH] [--semantic] [--incremental]
uv run python -m src search         QUERY [--k INT] [--index_dir PATH] [--semantic] [--hybrid] [--cache]
uv run python -m src search_dataset --dataset_path PATH [--k INT] [--save_directory PATH]
uv run python -m src answer         QUERY [--k INT] [--index_dir PATH]
uv run python -m src answer_dataset --student_search_results_path PATH [--save_directory PATH]
uv run python -m src evaluate       --student_search_results_path PATH --dataset_path PATH
uv run python -m src serve          [--host HOST] [--port INT] [--index_dir PATH]
```

### Makefile Shortcuts

| Command | Action |
|---|---|
| `make install` | Install dependencies with uv |
| `make run` | Index the corpus (max_chunk_size=2000) |
| `make debug` | Index via pdb |
| `make search_dataset` | Search docs dataset, save to UnansweredQuestions/ |
| `make evaluate` | Local recall@k check on docs results |
| `make answer` | Answer one hardcoded question |
| `make lint` | Run flake8 + mypy |
| `make serve` | Local HTTP API on 127.0.0.1:8000 |
| `make clean` | Remove .venv, caches, data/processed/ |

---

## Example Usage

```sh
# Single search query
uv run python -m src search "How does vLLM implement PagedAttention?" --k 5

# Example output
{
  "question_id": "...",
  "question": "How does vLLM implement PagedAttention?",
  "retrieved_sources": [
    {
      "file_path": "data/raw/vllm-0.10.1/vllm/core/block_manager.py",
      "first_character_index": 142,
      "last_character_index": 2000
    },
    ...
  ]
}

# Single answer (Ollama required)
uv run python -m src answer "What models does vLLM support?" --k 5
```

---

## Performance Analysis

Benchmarked on the public evaluation datasets with `--max_chunk_size 2000` and `--k 10`:

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Target | Status |
|---|---|---|---|---|---|---|
| Docs | 52.0 % | 78.0 % | **84.0 %** | 89.0 % | ≥ 80 % | ✅ |
| Code | 33.3 % | 50.5 % | **54.5 %** | 63.6 % | ≥ 50 % | ✅ |

**Observations:**
- Markdown chunking with `RecursiveChunker` significantly improves docs recall compared
  to sentence-based chunking, because doc sections stay coherent.
- Code chunking with `chonkie.CodeChunker` (AST-aware) prevents splitting function
  signatures from their bodies, which matters for exact-match scoring.
- BM25 with stemming outperforms raw token matching on paraphrased questions, because
  stemming maps `configuring` and `configuration` to the same root.
- Indexing 2 734 files / 16 251 chunks takes under 60 seconds on a modern laptop.
- Retrieval throughput for 200 questions: under 10 seconds.

**Tuning knobs:**
- Lower `--max_chunk_size` (e.g. 500) → higher recall, more noise per chunk
- Higher `--k` → higher recall@k ceiling, slower answer generation
- Adding `--semantic` / `--hybrid` (bonus) complements BM25 on paraphrases

---

## Design Decisions

**BM25 over TF-IDF:** BM25 applies document-length normalisation that TF-IDF lacks.
On a corpus where file sizes vary by orders of magnitude (from 10-line configs to
5 000-line Python modules), this normalisation prevents large files from dominating
every query.

**Per-extension chunking strategy:** A Python file and a Markdown page have different
natural boundaries. Treating them the same way wastes chunk budget — splitting a
docstring mid-sentence or a function mid-body reduces the quality of both retrieval
and answer generation.

**Character indices instead of line numbers:** The moulinette compares retrieved
sources using character ranges (IoU ≥ 5 %). Storing and returning exact character
offsets from chonkie directly avoids any off-by-one errors that re-computing line→char
conversions would introduce.

**`answer_dataset` consumes pre-computed search results:** Decoupling retrieval and
generation means we can iterate on recall (fast) without re-running the slow LLM
inference loop, and vice versa.

**Ollama for LLM serving:** Ollama manages model weights, quantisation, and the
inference server locally. The `qwen3:0.6b` model runs on CPU in a few seconds per
question, making the pipeline fully offline.

**Excluded directories:** Hidden directories (`.buildkite`, `.github`, `.gemini`)
are skipped during indexing. They contain CI/CD configuration with no relevance to
user-facing vLLM questions, and including them would add noise to retrieval.

---

## Challenges Faced

**Binary file detection:** The vLLM repository contains compiled `.so` files, font
files, images, and architecture-specific blobs. Without an explicit extension blocklist,
chonkie raises `UnicodeDecodeError`. The solution was to maintain `IGNORE_EXTENSIONS`
in `chunking.py` and skip those files before attempting to read them.

**macOS / scipy incompatibility:** Python 3.10 scipy wheels ship with a linker section
(`__thread_bss`) that macOS 27's dynamic linker rejects. Pinning to Python 3.12
(`.python-version`) resolves this without changing the `requires-python = ">=3.10"`
constraint for the evaluation Linux machine.

**Chunk size vs. recall trade-off:** Larger chunks are faster to index and contain
more context for the LLM, but risk exceeding the moulinette's 2 000-character hard
limit (which invalidates the entire output). The `--max_chunk_size` guard in the
CLI and a runtime check in `chunk_content` enforce this ceiling.

**Path verbatim matching:** The moulinette compares `file_path` fields verbatim.
Any path prefix mismatch (e.g. `./data/raw/` vs. `data/raw/`) causes a miss.
Paths are stored exactly as passed to `get_all_files()` — which walks from
`data/raw/vllm-0.10.1` — so the prefix is always consistent.

---

## Resources

### Documentation & Papers

- [BM25S — Orders of magnitude faster lexical search](https://arxiv.org/abs/2407.03618)
- [BM25S GitHub](https://github.com/xhluca/bm25s)
- [Chonkie chunking library](https://docs.chonkie.ai/)
- [Python Fire CLI](https://google.github.io/python-fire/using-cli/)
- [Pydantic v2 docs](https://docs.pydantic.dev/latest/)
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Ollama](https://ollama.com/)

### AI Usage

We used AI tools (Claude via Cursor) for:

- **Gap analysis:** comparing the existing codebase against the new subject specification
  to identify missing fields, wrong CLI signatures, and incorrect output paths
- **Refactoring assistance:** rewriting `__main__.py` to match the subject's exact CLI
  interface (`index`, `search`, `search_dataset`, `answer`,
  `answer_dataset`, `evaluate`, `serve`)
- **Type annotation fixes:** resolving mypy errors caused by chonkie's missing `__all__`
  exports and scipy's `Any`-typed return values
- **Debugging:** diagnosing the macOS 27 + Python 3.10 + scipy binary incompatibility

We reviewed, tested, and kept only the generated code we understand.
Retrieval, chunking, and architecture choices are ours.
