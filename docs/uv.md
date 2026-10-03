# uv

The subject requires **uv** for install and run. The moulinette and
reviewers call `uv sync` from the repo root. We keep `pyproject.toml`
and `uv.lock` there.

## How we work in this repo

```sh
# one-time on a machine
curl -LsSf https://astral.sh/uv/install.sh | sh

# create .venv and install runtime + dev groups (pytest, flake8, mypy)
make install
# same as: uv sync

# run the CLI inside that env
uv run python -m src --help
uv run python -m src index --max_chunk_size 2000
```

`uv run` picks the project interpreter; we do not activate `.venv`
by hand.

## Dependencies

Runtime packages live in `[project] dependencies` in `pyproject.toml`
(`fire`, `bm25s`, `chonkie`, `pydantic`, `sentence-transformers`, …).
Lint and test tools are the `dev` group. After `uv add <pkg>` we
commit the updated lockfile.

We do not use `requirements.txt`. Python 3.12 is pinned in
`.python-version` (3.10+ is still what `requires-python` allows).

## References

- [uv documentation](https://docs.astral.sh/uv/)
- [pyproject.toml spec](https://packaging.python.org/en/latest/specifications/pyproject-toml/)
