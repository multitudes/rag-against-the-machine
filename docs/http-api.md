# Local HTTP API

Bonus 5: **search** and **answer** over a small local HTTP server so
a script, `curl`, a notebook, or a tiny UI can drive the same pipeline
as the CLI.

The moulinette still uses the CLI (`search_dataset` /
`answer_dataset`). `serve` is for the defence demo and for local
tooling.

```sh
# index once (same as always)
uv run python -m src index --max_chunk_size 2000 --semantic

# API on loopback, port 8000
uv run python -m src serve
# or: make serve
```

It prints:

```text
RAG API at http://127.0.0.1:8000
  GET  /health
  GET  /search?query=...&k=5
  POST /search
  POST /answer
```

We leave that process running and call it from another terminal:

```sh
curl -s 'http://127.0.0.1:8000/health'

curl -s 'http://127.0.0.1:8000/search?query=How+to+stop+a+worker%3F&k=5'

curl -s http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"How to stop a worker?","k":5,"hybrid":true,"cache":true}'

# needs Ollama with qwen3:0.6b
curl -s http://127.0.0.1:8000/answer \
  -H 'Content-Type: application/json' \
  -d '{"query":"How to stop a worker?","k":5}'
```

Ctrl+C stops the server.

---

## Why an API?

Each CLI command is its own process: start Python, load BM25, print,
exit. That fits the moulinette. It is awkward for ten questions from
a browser or another program.

The HTTP server is **one long-lived process**. It loads the index once
(bonus 4’s in-process `Searcher` cache) and then answers requests.
`search --cache` still writes `query_cache.json`; here the same switch
is `"cache": true` in the JSON (or `?cache=true`).

Default bind is **127.0.0.1** — this machine only, not a public site.

---

## Endpoints

| Method | Path | Role |
|--------|------|------|
| GET | `/` | Lists the routes (handy in a browser) |
| GET | `/health` | Process is up; `index_ready` if `metadata.json` exists |
| GET | `/search?query=…&k=5` | Retrieve sources |
| POST | `/search` | Same, JSON body |
| GET/POST | `/answer` | Retrieve + generate with Qwen/Ollama |

### `/search` body / query string

| Field | Default | Meaning |
|-------|---------|---------|
| `query` (or `q`) | — | Question (required) |
| `k` | 5 | Top-k sources |
| `semantic` | false | MiniLM only (bonus 1) |
| `hybrid` | false | RRF BM25 + MiniLM (bonus 2) |
| `cache` | false | Query cache (bonus 4) |

`true` / `1` / `yes` all count as true.

Example response:

```json
{
  "question": "How to stop a worker?",
  "k": 5,
  "semantic": false,
  "hybrid": false,
  "cache": true,
  "cached": false,
  "retrieved_sources": [
    {
      "file_path": "data/raw/vllm-0.10.1/docs/...",
      "first_character_index": 120,
      "last_character_index": 400
    }
  ]
}
```

`cached` is `true` when this hit came from `query_cache.json`.

### `/answer`

Same fields, plus `"answer": "…"` from Qwen. If Ollama is down the
status is **503** (`Ollama is not running`). Empty query or bad `k`
is **400**. Missing index is **503**. Unknown path is **404**.

No extra pip package: stdlib `http.server` only.

---

## Defence notes

We `index` (if needed) then `serve`, open
`http://127.0.0.1:8000/` and `/health` in a browser, `curl` `/search`
(same sources as CLI `search`), repeat with `"cache": true` until
`"cached": true`, optionally `"hybrid"` / `"semantic"`, then `/answer`
with Ollama up. The moulinette still uses the CLI; this is the extra
interface.

### Tests (offline, Ollama mocked)

```sh
uv run pytest tests/test_api.py -v
```
