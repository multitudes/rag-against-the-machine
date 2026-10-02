"""Configuration constants for the RAG system."""

OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_HEALTH_URL = "http://localhost:11434/api/tags"

# Hard limit imposed by the moulinette: any retrieved source longer than
# this is rejected and invalidates the entire output file.
MAX_CHUNK_SIZE = 2000

# Bonus 1 — MiniLM vector index stored next to the BM25 files.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDINGS_FILENAME = "embeddings.npy"
