# Recall@k

The moulinette (and our `evaluate` command) score **retrieval only**.
We do not generate answers to compute recall.

Recall@k asks: of the ground-truth sources for a question, how many
did we retrieve in our top-k?

## What counts as a hit

A retrieved `MinimalSource` matches a ground-truth source when:

- `file_path` is **exactly** the same string, and
- the character ranges overlap enough (IoU ≥ 0.05 / about 5 %).

A hit in the wrong file never counts. We do not need the spans to
match exactly — covering the right region of the right file is enough.

## Per-question score

\[
\text{recall@k} = \frac{\text{ground-truth sources found in top-}k}{\text{number of ground-truth sources}}
\]

Dataset recall is the average of those per-question scores. Docs must
reach **80 % recall@5**, code **50 %**.

## What we compare

| File | Role |
|------|------|
| `data/datasets/AnsweredQuestions/…` | Ground truth (`sources` on each question) |
| Our `search_dataset` JSON | `retrieved_sources` for the same `question_id`s |

Character indexes matter for the overlap check. The official number
during the defence comes from the **moulinette**, not from `evaluate`.

Answer quality (whether Qwen phrased things well) is a separate
judgement and is not part of recall@k.

See [retrieving-methods.md](retrieving-methods.md) for BM25 vs TF-IDF.
