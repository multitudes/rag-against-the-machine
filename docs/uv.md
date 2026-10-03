# Using `uv` with Python Projects

`uv` is a fast Python package manager and virtual environment tool. In this project, we are required to use `uv` for dependency management and running our code.

## Setting Up the Environment

1. **Install `uv`** (if not already installed):
```zsh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Initialize a new uv project** (modern approach):
```zsh
uv init
```
This creates a `pyproject.toml` file for dependency management.

3. **Add dependencies:**
```zsh
uv add torch transformers huggingface-hub
uv add --dev flake8  # for development dependencies
```

4. **Install/sync dependencies:**
```zsh
uv sync
```
This installs all dependencies from `pyproject.toml` and creates/updates `uv.lock`.

## Running Your Project

To run your main script as required by the project:

```zsh
uv run python -m src
```

This command will:
- Automatically use the correct Python interpreter and virtual environment
- Run the `src` module as the entry point
- No need to manually activate the virtual environment

## Common `uv` Commands

- **Add a package:**
```zsh
uv add <package>
```
- **Add development dependency:**
```zsh
uv add --dev <package>
```
- **Remove a package:**
```zsh
uv remove <package>
```
- **List installed packages:**
```zsh
uv tree
```
- **Run scripts:**
```zsh
uv run python <script.py>
```
- **Sync dependencies:**
```zsh
uv sync
```

## Project Structure

With modern `uv`, your project uses:
- `pyproject.toml` - Project configuration and dependencies
- `uv.lock` - Locked dependency versions for reproducibility
- `.venv/` - Virtual environment (auto-created)

No `requirements.txt` file is needed.

## Notes
- `uv run` automatically manages the virtual environment - no manual activation needed
- The `uv.lock` file ensures reproducible builds across different machines
- For this project, all classes must use `pydantic` for validation, and you may use `numpy` and `json`
- Do **not** use forbidden packages (see README for details)

## References
- [uv documentation](https://docs.astral.sh/uv/)
- [pyproject.toml specification](https://packaging.python.org/en/latest/specifications/pyproject-toml/)