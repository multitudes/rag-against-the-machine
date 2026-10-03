# TF-IDF vs BM25

The subject requires **at least one** classic lexical retrieval method:
**TF-IDF** or **BM25**. This project uses **BM25** (via
[`bm25s`](https://github.com/xhluca/bm25s)).

Both methods score how well a query matches a document (in our case, a
*chunk*) using **words**, not neural embeddings. They are fast, CPU-only,
and easy to index offline.

---

## Shared idea: “important words”

Lexical search answers:

> Which documents contain the query terms, and how strongly?

Two building blocks appear in both formulas:

| Idea | Meaning |
|------|---------|
| **Term frequency (TF)** | How often a query term appears *in this document* |
| **Inverse document frequency (IDF)** | How rare that term is *across the whole corpus* |

A word that appears many times in one chunk but almost nowhere else
(e.g. `tensorizer`) is a strong signal. A word that appears everywhere
(e.g. `the`, `function`) is weak.

---

## TF-IDF

**TF-IDF** = Term Frequency × Inverse Document Frequency.

### Intuition

- Documents that use the query terms a lot score higher (TF).
- Rare terms count more than common ones (IDF).

A simple form of the weight for term \(t\) in document \(d\) is:

\[
\text{tf-idf}(t, d) = \text{tf}(t, d) \times \log\frac{N}{\text{df}(t)}
\]

where \(N\) is the number of documents and \(\text{df}(t)\) is how many
documents contain \(t\).

At query time a TF-IDF system typically:

1. Turn the query and every document into TF-IDF vectors.
2. Rank documents by **cosine similarity** (or a similar vector score)
   between the query vector and each document vector.

### Strengths

- Simple to understand and implement (`sklearn.TfidfVectorizer`).
- Works well as a baseline keyword matcher.
- No GPU required.

### Weaknesses

- **No saturation:** repeating a term 50 times can dominate the score
  even if the document is not really “about” that term.
- **Weak length handling:** long documents accumulate more term hits by
  chance and can unfairly outrank short, focused ones.
- Often slightly weaker than BM25 on the same corpus.

---

## BM25

**BM25** (“Best Matching 25”) is a probabilistic ranking function from
the Okapi family. It keeps the TF × IDF intuition but adds two important
fixes.

### What BM25 improves over TF-IDF

1. **Term-frequency saturation**  
   Extra occurrences of the same word help less and less. Matching
   `vllm` three times is not three times better than matching it once.

2. **Document-length normalisation**  
   Scores are adjusted by how long the document is relative to the
   average document length, so long files do not win just by being long.

### Tunable parameters

| Parameter | Role (typical defaults) |
|-----------|-------------------------|
| **k1** | Controls TF saturation (often ~1.2–2.0) |
| **b** | Controls length normalisation (often ~0.75; `b=0` turns it off) |

We leave k1 and b at the `bm25s` defaults.

### Strengths

- Stronger default ranking than plain TF-IDF on most text corpora.
- Still pure lexical search: fast index, fast query, no GPU.
- Industry standard for keyword retrieval (Elasticsearch, Lucene, …).

### Weaknesses

- Still **lexical**: paraphrases without shared tokens
  (`“kill a process”` vs `"terminate a job"`) may miss each other.
- Scores are **relative** to the query, not probabilities or percentages.
  A top score of `4.3` is only meaningful compared to the next ranks for
  *that* query.

---

## Side-by-side

| | TF-IDF | BM25 |
|---|---|---|
| Core signal | TF × IDF | TF × IDF with saturation + length norm |
| Long documents | Can be over-favoured | Penalised / normalised |
| Repeated terms | Linear-ish growth | Saturates (diminishing returns) |
| Typical use | Classic IR baseline | Default lexical ranker in search engines |
| This project | Not used | **Used** (`bm25s` + English stemmer) |
| Subject recall hint* | Often lower bar | Often higher bar |

\*Exact moulinette thresholds depend on the subject version; the idea is
that BM25 is expected to retrieve a bit better than plain TF-IDF.

---

## How this project uses BM25

1. **Index time** (`ingestion/indexing.py`)  
   Chunks are tokenised (stopwords + stemming), then a BM25 sparse index
   is saved under `data/processed/` together with `metadata.json`
   (file path + character offsets for each chunk).

2. **Query time** (`retrieval/search.py`)  
   The question is tokenised the same way; BM25 returns the top-\(k\)
   chunk ids; metadata turns those ids into `MinimalSource` locations
   for the moulinette / LLM context.

MiniLM (`--semantic`) is a bonus layer on top of this, not a
replacement for the required lexical method.

---

## Why we did not use TF-IDF

TF-IDF is enough for a coursework baseline or a tiny `sklearn`
prototype. Our corpus mixes 10-line configs and multi-thousand-line
modules, and the questions are identifier-heavy, so **BM25 is the
better default** (length normalisation + term saturation).

---

## Further reading

- [BM25S paper](https://arxiv.org/abs/2407.03618) — fast BM25 in Python
- [bm25s on GitHub](https://github.com/xhluca/bm25s)
- Project notes: [bm25.md](bm25.md) (score interpretation),
  [recall.md](recall.md) (how recall@k is measured)
