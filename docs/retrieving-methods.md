# Retrieving Methods

## What are Retrieving Methods?

Retrieving methods are algorithms that find the most relevant document chunks/passages for a given query. They're the "R" (Retrieval) part of your RAG (Retrieval-Augmented Generation) system.

## Required Implementation

You must implement **at least one** of these basic retrieving methods:

### 1. TF-IDF (Term Frequency-Inverse Document Frequency)
- **What it does**: Measures how important a word is to a document relative to a collection of documents
- **How it works**: 
  - TF: How often a term appears in a document
  - IDF: How rare/common a term is across all documents
  - Score = TF × IDF
- **Good for**: Basic keyword matching
- **Performance target**: **65% recall@5** on English questions

### 2. BM25 (Best Matching 25)
- **What it does**: An improved version of TF-IDF with better handling of document length and term frequency saturation
- **How it works**: Uses a more sophisticated formula that considers:
  - Document length normalization
  - Term frequency saturation (diminishing returns for repeated terms)
  - Tunable parameters (k1, b)
- **Good for**: Better than TF-IDF, widely used in search engines
- **Performance target**: **75% recall@5** on English questions

## Performance Targets Explained

### What is Recall@5?
- **Recall@5**: Out of all the relevant documents for a question, what percentage appear in the top 5 search results?
- **Example**: If there are 10 relevant documents for a question, and 4 of them appear in your top 5 results, then recall@5 = 4/10 = 40%

### Your Targets:
- **BM25**: Must achieve ≥75% recall@5
- **TF-IDF**: Must achieve ≥65% recall@5

## Implementation with Your Libraries

You already have `bm25s` in your dependencies, which makes this easier:

```python
# BM25 implementation using bm25s
import bm25s

# For TF-IDF, you can use:
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

## Why This Matters

The retrieval quality directly impacts your final RAG system performance:
- **Poor retrieval** → Wrong context → Bad answers
- **Good retrieval** → Relevant context → Better answers

The 75%/65% targets ensure your retrieval is good enough to support quality answer generation.