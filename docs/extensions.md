# File extensions

How we treat extensions under `data/raw/vllm-0.10.1/` in
`chunk_content` / `IGNORE_EXTENSIONS`.

| Extension | Description | Ingestion Strategy |
| :--- | :--- | :--- |
| `.py` | Python source code files. | **Code** |
| `.md` | Markdown files, used for documentation. | **Markdown** |
| `.txt` | Plain text files. | **Text** |
| `.sh` | Shell script files for automation. | **Code** |
| `.cu` | CUDA C++ source files for GPU programming. | **Code** |
| `.cuh` | CUDA C++ header files. | **Code** |
| `.cpp` | C++ source code files. | **Code** |
| `.h` | C/C++ header files. | **Code** |
| `.cmake` | CMake build system script files. | **Code** |
| `CMakeLists.txt` | The primary configuration file for a CMake project. | **Code** |
| `.toml` | TOML (Tom's Obvious, Minimal Language) configuration files. | **Text** |
| `.yaml` / `.yml` | YAML (YAML Ain't Markup Language) configuration files. | **Text** |
| `.json` | JSON (JavaScript Object Notation) data files. | **Text** |
| `.pyi` | Python interface stub files for type hinting. | **Code** |
| `Dockerfile` | Instructions for building a Docker container image. | **Code** |
| `.pdf` | Portable Document Format. | **Ignore** (Binary) |
| `.zip` | Compressed archive file. | **Ignore** (Binary) |
| `.so` | Shared Object, a compiled library file. | **Ignore** (Binary) |
| `LICENSE` | Software license file. | **Text** |
| `DCO` | Developer Certificate of Origin. | **Text** |
| `MANIFEST.in` | A manifest file specifying files to include in a Python source distribution. | **Text** |
| `.svg` / `.png` | Scalable Vector Graphics / Portable Network Graphics image files. | **Ignore** (Binary) |
| `.eot` / `.ttf` / `.woff` / `.woff2` | Web font files. | **Ignore** (Binary) |
| `.html` | HyperText Markup Language files. | **Markdown** (Can be treated as structured text) |
| `.css` | Cascading Style Sheets for web styling. | **Code** |
| `.rst` | reStructuredText, a format for textual data used primarily in the Python community. | **Markdown** |
| `.in` | Template files used by build systems to generate other files. | **Text** |
| `.pylintrc` | Configuration file for the Pylint linter. | **Ignore** (Config) |
| `.j2` | Jinja2 template file. | **Text** |