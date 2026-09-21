# Variables
UV      := uv
PYTHON  := $(UV) run python
MYPY    := $(UV) run mypy
FLAKE8  := $(UV) run flake8
PYTEST  := $(UV) run pytest

# ── Mandatory rules ────────────────────────────────────────────────────────────

install:
	@command -v $(UV) >/dev/null 2>&1 || { \
		echo "Error: 'uv' is required but not installed."; \
		echo "Please install uv before running setup (see README.md)."; \
		exit 1; \
	}
	@echo "uv version: $$($(UV) --version)"
	@if [ ! -f pyproject.toml ]; then \
		echo "Initializing new uv project..."; \
		$(UV) init; \
	fi
	$(UV) sync

run:
	$(PYTHON) -m src index --max_chunk_size 2000

debug:
	$(PYTHON) -m pdb -m src index --max_chunk_size 2000

lint:
	$(FLAKE8) .
	$(MYPY) .

lint-strict:
	$(FLAKE8) .
	$(MYPY) . --strict

clean:
	@echo "Removing .venv"
	@rm -rf .venv
	@echo "Removing __pycache__"
	@find . -type d -name '__pycache__' -not -path './.venv/*' -exec rm -rf {} +
	@echo "Removing .mypy_cache"
	@rm -rf .mypy_cache
	@echo "Removing data/processed (index)"
	@rm -rf data/processed

fclean: clean
	@echo "Removing Hugging Face Hub model cache (~/.cache/huggingface/hub)..."
	rm -rf "$(HOME)/.cache/huggingface/hub"

# ── Convenience shortcuts (not part of the graded interface) ──────────────────

index:
	$(PYTHON) -m src index --max_chunk_size 2000

search:
	$(PYTHON) -m src search "OpenAI compatible server" --k 10

search_dataset:
	$(PYTHON) -m src search_dataset \
		--dataset_path data/datasets/UnansweredQuestions/Dataset_2025-09-21_valid_unanswered.json \
		--k 10 \
		--save_directory data/output/search_results/UnansweredQuestions

evaluate:
	$(PYTHON) -m src evaluate \
		--student_search_results_path data/output/search_results/UnansweredQuestions/Dataset_2025-09-21_valid_unanswered.json \
		--dataset_path data/datasets/AnsweredQuestions/Dataset_2025-09-21_valid_answered.json

answer:
	$(PYTHON) -m src answer "How to configure OpenAI server?" --k 10

help:
	$(PYTHON) -m src --help

.PHONY: install run debug lint lint-strict clean fclean \
        index search search_dataset evaluate answer help
