# rag-against-the-machine

## Usage

### Install dependencies
```sh
make install
```
Installs all Python dependencies using [uv](https://github.com/astral-sh/uv) and initializes the project if needed.

### Index your data
```sh
make ingest
```
Chunks and indexes your data for retrieval (creates BM25 indices).

### Search for an answer
```sh
make search
```
Runs a search for the default query ("OpenAI compatible server") and saves the results.

### Search using a dataset of questions
```sh
make search_dataset
```
Runs search for all questions in the provided dataset file.

### Evaluate recall@k
```sh
make evaluate
```
Measures recall@k on your search results using a ground truth dataset.

### Generate answers for a dataset
```sh
make generate
```
Runs the RAG pipeline to generate answers for all questions in the dataset.

### Answer a single question
```sh
make answer
```
Answers a single question using the RAG pipeline.

### Clean up all generated files
```sh
make clean
```
Removes virtual environments, all `__pycache__` folders, `.egg-info`, and BM25 indices.

### Lint your code
```sh
make lint
```
Runs flake8 on the `src` directory.

### Show CLI help
```sh
make help
```
Shows all available CLI commands and options.

## Citations
```bibtex
@misc{bm25s,
      title={BM25S: Orders of magnitude faster lexical search via eager sparse scoring}, 
      author={Xing Han Lù},
      year={2024},
      eprint={2407.03618},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2407.03618}, 
}
```


## resources
https://google.github.io/python-fire/using-cli/#calling-a-function  
https://www.anthropic.com/engineering/contextual-retrieval  
https://bm25s.github.io  
https://github.com/xhluca/bm25s  
https://github.com/xhluca/bm25s?tab=readme-ov-file  
https://github.com/xhluca/bm25s/blob/main/examples/index_nq.py  
https://docs.chonkie.ai/python-sdk/chunkers/code-chunker  
https://github.com/chonkie-inc/chonkie?tab=readme-ov-file
