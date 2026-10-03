# Running the moulinette

The official recall@k for the defence is the **moulinette** binary,
not our `evaluate` command. It only scores **retrieval**: our
`search_dataset` JSON against the **AnsweredQuestions** ground truth.

We do not import or call this binary from `src/`.

## What to run first

```sh
make run
# or: uv run python -m src index --max_chunk_size 2000

uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```

That write is a `StudentSearchResults` file:
`search_results` + `k`, one `retrieved_sources` list per question.

The same flow for code, with
`dataset_code_public.json` and
`data/output/search_results/UnansweredQuestions/` (or a scoped
code folder if we keep outputs separate).

## Command

Subject order — **our results first**, ground truth second:

```sh
./moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 --max_context_length 2000
```

Or `make moulinette` (docs dataset, same flags).

| Argument | File | JSON shape |
|----------|------|------------|
| 1 | Our `search_dataset` output | `{ "search_results": [...], "k": 10 }` |
| 2 | Ground truth | `{ "rag_questions": [..., "sources", "answer"] }` |
| `--k` | How many of our hits to score | Use the same *k* we searched with, or smaller (e.g. search 10, score 5) |
| `--max_context_length` | 2000 | Rejects any retrieved span longer than this |

The binary on disk may be named `moulinette-ubuntu` or
`moulinette-fedora` in the subject zip. We rename it to `moulinette`
and `chmod +x` it.

## What success looks like

```text
Student data is valid: True
Evaluation Results
========================================
Recall@1: … Recall@3: … Recall@5: … Recall@10: …
```

Targets: **docs ≥ 80 % recall@5**, **code ≥ 50 % recall@5**.
Figures in the subject PDF are only a format example.

## The error we hit with swapped files

```sh
./moulinette evaluate_student_search_results \
  data/datasets/UnansweredQuestions/dataset_docs_public.json \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --k 5
```

Pydantic then says `search_results` and `k` are missing, because
argument 1 was the **questions** file (`rag_questions`), not our
search output. `UnansweredQuestions` also has no `sources`, so it
cannot be the ground-truth file either.

Use **AnsweredQuestions** as argument 2.

## Ours vs the moulinette

`make evaluate` / `uv run python -m src evaluate` is a local check
with the same two files (results path, then ground truth). The
defence number is always the moulinette.

See [recall.md](recall.md) for how a hit is counted (exact `file_path`,
span overlap ≥ 5 %).
