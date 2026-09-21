# CODE REVIEW — Subject v2.0 Gap Analysis & Action Plan

> Generated from inspection of the full codebase against `rag assets/en.subject-6.pdf` (v2.0).

---

## Summary

The existing implementation is a solid first version. It indexes the vLLM corpus with BM25,
retrieves chunks, and answers via Ollama. However, several **breaking mismatches** exist
between the current code and the new subject specification that will cause the automated
moulinette to reject output entirely. Below is the full gap analysis, ordered by severity.

---

## 🔴 CRITICAL — Will break the moulinette / automated evaluation

### 1. `MinimalSearchResults` is missing the `question` field

**Location:** `src/core/schemas.py` lines 40–44

**Current:**
```python
class MinimalSearchResults(BaseModel):
    question_id: str
    retrieved_sources: List[MinimalSource]
```

**Subject requires:**
```python
class MinimalSearchResults(BaseModel):
    question_id: str
    question: str          # ← MISSING
    retrieved_sources: List[MinimalSource]
```

The moulinette README (`rag assets/README.md`) shows exactly this format and validates it.
Every JSON output your system produces will be **invalid** without this field.
`MinimalAnswer` inherits from `MinimalSearchResults`, so it is also affected.

**Fix:** Add `question: str` to `MinimalSearchResults`. Update every place that constructs
a `MinimalSearchResults` or `MinimalAnswer` to pass the question text.
Affected call sites:
- `src/retrieval/search.py` — `search_one()` and `search_dataset()`
- `src/answering/answer.py` — `get_answer()`

---

### 2. Index is saved to the wrong path

**Location:** `src/__main__.py` line 97, `src/retrieval/search.py` line 18 and 130, `src/retrieval/search.py` line 18

**Current:** everything uses `"bm25s_indices/"` (repo root)

**Subject requires:** `data/processed/`

```
uv run python -m src index --max_chunk_size 2000
→ Ingestion complete! Indices saved under data/processed/
```

The automated evaluation scripts assume `data/processed/` for the index. The `bm25s_indices/`
directory will not be found.

**Fix:**
- Change default `index_dir` from `"bm25s_indices/"` to `"data/processed/"` everywhere.
- The path must also be **configurable**, but `data/processed/` must be the default.

---

### 3. CLI command signatures do not match the subject spec

The subject defines an exact interface. Below is the mapping of what is required vs. what exists:

| Subject command | Current command | Problem |
|---|---|---|
| `index --max_chunk_size <int>` | `index` (chunk_size in `__init__`) | `--max_chunk_size` is not a direct `index` arg |
| `search <query> --k <int>` | `search` | `k` comes from `__init__`, not the method |
| `search_dataset --dataset_path <path> --k <int> --save_directory <dir>` | `search_dataset` | missing `--k` and `--save_directory` args |
| `answer <query> --k <int>` | `answer_one` | wrong name; `k` not a direct arg |
| `answer_dataset --student_search_results_path <path> --save_directory <dir>` | `answer_dataset` | completely wrong signature and architecture |
| `evaluate --student_search_results_path <path> --dataset_path <path>` | `measure_recall_at_k_on_dataset` | wrong name |

**Fix:** Refactor `RagCLI` so each command method accepts its own parameters directly
(not via `__init__`). With Python Fire, method parameters become CLI flags.
Rename `answer_one` → `answer` and `measure_recall_at_k_on_dataset` → `evaluate`.

---

### 4. `answer_dataset` has the wrong architecture

**Location:** `src/__main__.py` lines 349–410

**Current behaviour:** `answer_dataset` re-runs the full search pipeline from scratch
and ignores any pre-existing search results.

**Subject requires:** `answer_dataset` takes a **pre-computed** `StudentSearchResults` JSON
(produced by `search_dataset`) and generates answers for each entry.

```
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

The workflow is strictly sequential: `index` → `search_dataset` → `answer_dataset`.
`answer_dataset` must load `StudentSearchResults`, retrieve source text for each result,
call the LLM, and output a `StudentSearchResultsAndAnswer` JSON.

---

### 5. Output paths are hardcoded and do not use `--save_directory`

**Locations:** `src/__main__.py`, `src/utils.py`

- `search_dataset` always saves to `"data/output/search_results"` (no scoping by dataset).
- `answer_dataset` always saves to `"data/output/search_results/"`.
- `answer_one` saves to `"data/output"` with a timestamp-based name.

**Subject requires** (section VI.7.1):
```
data/output/search_results/<DatasetScope>/
data/output/search_results_and_answer/<DatasetScope>/
```
where `<DatasetScope>` is passed via `--save_directory` at call time.
All paths must be **configurable CLI arguments, never hard-coded**.

---

## 🟠 HIGH — Will lose significant points

### 6. `search_dataset` in `Searcher` ignores the `k` parameter

**Location:** `src/retrieval/search.py` line 86

```python
def search_dataset(
        self,
        questions: List[UnansweredQuestion],
        k: int = 5          # ← default k=5, never overridden from CLI
) -> StudentSearchResults:
```

The `k` value from the CLI is never passed through to `search_dataset`.
In `__main__.py`, `search_dataset` calls `searcher.search_dataset(unanswered.rag_questions)`
without forwarding `k`.

**Fix:** Propagate `k` from the CLI method argument all the way down to `search_one`.

---

### 7. `RecursiveChunker` for Markdown ignores `--max_chunk_size`

**Location:** `src/ingestion/chunking.py` lines 99–101

```python
chunker = RecursiveChunker.from_recipe("markdown", lang="en")
```

This uses a pre-built recipe with a fixed chunk size, completely ignoring the
user-supplied `chunk_size` argument. The subject is explicit:

> "Chunk size is configurable through a CLI argument (--max_chunk_size), with a default
> of 2000 characters. Do not go above it: the moulinette rejects any retrieved source
> longer than 2000 characters."

Since docs questions have an 80% recall@5 target and they rely heavily on Markdown files,
this is a high-priority fix.

**Fix:** Instantiate `RecursiveChunker` with the `chunk_size` parameter explicitly.

---

### 8. README does not meet the 42 requirements

**Location:** `README.md`

The current README is a minimal usage guide. The subject (Chapter VIII) mandates:

- [ ] **First line** must be italicised and read:
  *"This project has been created as part of the 42 curriculum by `<login>`."*
- [ ] **Description** section — project goal and overview
- [ ] **Instructions** section — installation and execution
- [ ] **Resources** section — references AND description of AI usage
- [ ] **System architecture** — RAG pipeline components and interactions
- [ ] **Chunking strategy** — document segmentation approach
- [ ] **Retrieval method** — algorithm and ranking mechanism
- [ ] **Performance analysis** — recall@k scores and system performance
- [ ] **Design decisions** — key implementation choices
- [ ] **Challenges faced** — difficulties and solutions
- [ ] **Example usage** — clear examples of running the system
- [ ] Written in **English**

**Fix:** Rewrite `README.md` from scratch following the template above.

---

### 9. Makefile has multiple issues

**Location:** `makefile`

| Issue | Line(s) | Fix |
|---|---|---|
| `lint` target is **defined twice** | 29 and 71 | Remove the duplicate (line 71 runs bare `flake8 src`) |
| `run` target is **empty** | 22 | Add `$(PYTHON) -m src` or a sensible default |
| `debug` target is **empty** | 24 | Add `$(PYTHON) -m pdb -m src` |
| `lint` flags incomplete | 29–30 | Subject requires `mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs` |
| `lint-strict` is present but subject optional | 33–34 | Keep, but ensure the mandatory `lint` is correct first |
| Custom targets (`ingest`, `search`, etc.) use hardcoded values | 50–70 | Fine for dev convenience, but ensure the named commands work too |

Corrected `lint` rule:
```makefile
lint:
	$(FLAKE8) .
	$(MYPY) . --warn-return-any --warn-unused-ignores \
	          --ignore-missing-imports --disallow-untyped-defs \
	          --check-untyped-defs
```

---

### 10. `pyproject.toml` dependencies are stale

**Location:** `pyproject.toml`

| Package | Status | Action |
|---|---|---|
| `dspy` | Unused (no imports found) | Remove |
| `langchain` | Unused (no imports found) | Remove |
| `chromadb` | Unused (no imports found) | Remove |
| `pydantic` | Used everywhere, not listed | Add |
| `requests` | Used in `answer.py` and `__main__.py` | Add |
| `mypy` | Required by `lint` target | Add (dev dep) |
| `flake8` | Required by `lint` target | Add (dev dep) |
| `PyStemmer` | Listed but pip name is `PyStemmer` ✓ | Keep |

Also: `uv.lock` should be regenerated after cleaning up deps.

---

## 🟡 MEDIUM — Quality / robustness issues

### 11. LLM integration uses Ollama instead of direct `transformers`

**Location:** `src/core/config.py`, `src/answering/answer.py`, `src/core/ollama_request.py`

The subject says to use **`Qwen/Qwen3-0.6B`**. Using Ollama's `qwen3:0.6b` is functionally
similar, but it requires Ollama to be installed and running, which is a hidden dependency.
During evaluation, if the evaluator machine doesn't have Ollama running, `answer_dataset`
will silently fail. The subject approach likely assumes `transformers`/`vllm` directly.

**Recommended fix:** Add a fallback or switch to using `transformers` (via
`AutoModelForCausalLM` + `AutoTokenizer`) so the model runs without a separate server.
This also aligns with the bonus point about a "Local HTTP API" — you could expose your
own API endpoint instead of depending on Ollama.

---

### 12. `index` method does not print the expected success message

**Location:** `src/__main__.py` line 107

The subject walkthrough shows:
```
Ingestion complete! Indices saved under data/processed/
```
Current code prints `Created index in X.XX seconds`. While not strictly graded, matching
expected output reduces confusion during evaluation.

---

### 13. `search_dataset` does not print the expected success message

**Location:** `src/__main__.py` line 187

Subject expects:
```
Saved student_search_results to data/output/search_results/UnansweredQuestions/dataset_docs_public.json
```
Current code prints a generic `Saved to file: <path>`.

---

### 14. Edge case handling is incomplete for CLI

**Location:** `src/__main__.py`

The subject states:
> "handle degenerate inputs (empty query, nonsensical query, k=0, missing files,
> malformed JSON) gracefully. The CLI is tested with such edge cases and must never
> crash with an unhandled traceback."

Current gaps:
- `k=0` is not validated (BM25 retrieve with k=0 may raise an exception)
- Empty `search_string` is not caught before the BM25 query
- The `index` method silently continues even if `chunk_size > 2000` (raises ValueError
  in `__init__`, which Fire may not present cleanly)

---

### 15. `src/ingestion/file_processing.py` excludes `.` directories but keeps dotfiles in vLLM

**Location:** `src/ingestion/file_processing.py` line 53

```python
dirs[:] = [d for d in dirs if not d.startswith('.') and d not in excluded_dirs]
```

This skips `.buildkite`, `.github`, `.gemini` etc. — those might contain relevant docs.
The subject says "Read the files you judge useful". This is a design decision you should
document in your README.

---

### 16. Type hints are incomplete in several functions

**Locations:** `src/answering/answer.py` (`create_prompt`, `calling_llm`, `get_answer`
missing return types on some lines), `src/__main__.py` (methods return `None` but not
annotated).

The subject mandates mypy compliance. Run `make lint` and fix all warnings.

---

## 🟢 LOW — Cosmetic / nice-to-have

### 17. `src/ingestion/chunking.py` — `overlap` parameter not used for `RecursiveChunker`

The `overlap: int = 200` parameter in `chunk_content` is only used by `SentenceChunker`.
`CodeChunker` and `RecursiveChunker` don't receive it. This is likely intentional but
should be documented.

### 18. `src/__main__.py` — `answer_dataset` references `question` in the except block

**Location:** line 408

```python
logger.error(f"'{question.question}': {e}")
```

If the exception happens before `question` is assigned in the loop, this will raise a
`NameError`. Use a safer fallback.

### 19. Missing `uv.lock`-level reproducibility for evaluation

The subject says reviewers only run `uv sync`. Ensure `uv.lock` is committed and up to
date after adding/removing dependencies.

---

## Action Plan (Ordered by Priority)

| # | Task | Files to change | Effort |
|---|---|---|---|
| 1 | Add `question: str` to `MinimalSearchResults` + fix all constructors | `schemas.py`, `search.py`, `answer.py` | Small |
| 2 | Change index path to `data/processed/` everywhere | `__main__.py`, `search.py`, `indexing.py` | Small |
| 3 | Refactor CLI: rename commands, move params from `__init__` to methods | `__main__.py` | Medium |
| 4 | Rewrite `answer_dataset` to consume pre-computed `StudentSearchResults` | `__main__.py`, `answer.py` | Medium |
| 5 | Pass `--save_directory` properly in `search_dataset` and `answer_dataset` | `__main__.py`, `utils.py` | Small |
| 6 | Propagate `k` from CLI through to `Searcher.search_dataset` and `search_one` | `__main__.py`, `search.py` | Small |
| 7 | Fix `RecursiveChunker` to respect `chunk_size` | `chunking.py` | Small |
| 8 | Fix `makefile`: remove duplicate `lint`, fill `run`/`debug`, fix lint flags | `makefile` | Small |
| 9 | Clean up `pyproject.toml`: remove unused deps, add `pydantic`/`requests` | `pyproject.toml` | Small |
| 10 | Rewrite `README.md` to meet 42 requirements | `README.md` | Large |
| 11 | Add type hints and pass `mypy` lint | all `src/` files | Medium |
| 12 | Add edge case guards (`k=0`, empty query) | `__main__.py` | Small |
| 13 | (Optional) Replace Ollama with direct `transformers` for Qwen/Qwen3-0.6B | `answer.py`, `config.py`, `ollama_request.py` | Large |

---

## Bonus Checklist (only after mandatory part is solid)

| Bonus | Status | Notes |
|---|---|---|
| Semantic embeddings (all-MiniLM-L6-v2 vector index) | ❌ Not started | Add a `VectorIndexer` + `vector_search` |
| Hybrid retrieval (lexical + semantic) | ❌ Not started | Depends on bonus 1 |
| Incremental indexing (re-index only changed files) | ❌ Not started | Track file hashes in `data/processed/` |
| Caching (index + query results) | ❌ Not started | Could use `functools.lru_cache` or file-based cache |
| Local HTTP API (FastAPI or Flask) | ❌ Not started | Expose `search` and `answer` endpoints |
