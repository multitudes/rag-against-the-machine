# Chunks

A **chunk** is a short slice of one source file: the text we index,
plus where it came from.

chonkie returns objects with at least `text`, `start_index`, and
`end_index`. We wrap that in our Pydantic `ChunkSource`:

- `text` — the slice we tokenise / embed
- `source.file_path` — path as stored for the moulinette
- `source.first_character_index` / `last_character_index` — offsets
  in the original file

`_enforce_max_size` then guarantees every chunk is ≤
`--max_chunk_size` (default 2000) so a single hit cannot invalidate
the whole output.

Which chunker we pick depends on the extension
([extensions.md](extensions.md)): CodeChunker for code,
RecursiveChunker for Markdown, SentenceChunker for plain text.
