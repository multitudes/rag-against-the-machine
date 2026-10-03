# pathlib in this repo

We use `pathlib.Path` for filesystem work (subject / flake8 style,
and Ruff PTH). `os.path` and `Path` do the same jobs; `Path` keeps
joins, exists-checks, and `open` on one object.

| We avoid | We use |
|---|---|
| `os.path.exists(p)` | `Path(p).exists()` |
| `os.makedirs(p, exist_ok=True)` | `Path(p).mkdir(parents=True, exist_ok=True)` |
| `os.path.join(a, b)` | `Path(a) / b` |
| `os.path.basename(p)` | `Path(p).name` |
| `open(p, …)` | `Path(p).open(…)` |

When `index_dir` is a `str`, we write `Path(index_dir) / "metadata.json"`,
never `Path(index_dir / "…")`.

```python
save_dir = Path(base) / "output" / "results"
save_dir.mkdir(parents=True, exist_ok=True)
output_path = save_dir / Path(dataset_path).name
```
