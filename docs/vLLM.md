# The vLLM corpus

Our knowledge base is the vLLM 0.10.1 tree under
`data/raw/vllm-0.10.1/` (not `assets/`). The evaluator provides this
checkout; we do not commit it.

It is mostly:

- **Python** — `vllm/`, examples, tests
- **Docs** — `docs/` (Markdown / RST / HTML)
- **Config** — YAML, JSON, TOML, Dockerfiles
- **Binaries we skip** — `.so`, images, fonts, archives (see
  [extensions.md](extensions.md))

`index` walks that tree, chunks the text files, and stores locations
as paths relative to the project root, e.g.
`data/raw/vllm-0.10.1/vllm/entrypoints/openai/api_server.py`. The
moulinette compares those strings verbatim.

A typical question (“How do we start the OpenAI-compatible server?”)
is answered by retrieving spans from those docs and entrypoints, then
(optionally) sending the spans to Qwen.
