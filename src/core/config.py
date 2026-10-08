"""Configuration constants for the RAG system."""

# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_HEALTH_URL = "http://localhost:11434/api/tags"
# Generate can be slow on CPU. Health may wait while a model loads.
OLLAMA_GENERATE_TIMEOUT = 120
OLLAMA_HEALTH_TIMEOUT = 10

# Hard limit imposed by the moulinette: any retrieved source longer than
# this is rejected and invalidates the entire output file.
MAX_CHUNK_SIZE = 2000

# Default BM25 / MiniLM index directory (Makefile, CLI, Searcher).
DEFAULT_INDEX_DIR = "data/processed"

# Bonus 1 — MiniLM vector index stored next to the BM25 files.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDINGS_FILENAME = "embeddings.npy"

# Bonus 2 — Reciprocal Rank Fusion of BM25 + MiniLM lists.
RRF_K = 60
HYBRID_POOL = 20

# Bonus 3 — per-file fingerprints for incremental indexing.
FILES_MANIFEST_FILENAME = "files.json"

# Bonus 4 — on-disk query cache next to the BM25 files.
QUERY_CACHE_FILENAME = "query_cache.json"

# Bonus 5 — local HTTP API (loopback only by default).
API_DEFAULT_HOST = "127.0.0.1"
API_DEFAULT_PORT = 8000
