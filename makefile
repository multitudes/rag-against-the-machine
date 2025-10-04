install:
	@command -v uv >/dev/null 2>&1 || { \
		echo "uv not found. Installing..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	}
	@echo "uv version: $$(uv --version)"
	@if [ ! -f pyproject.toml ]; then \
		uv init; \
		echo "uv project initialized. Edit pyproject.toml if needed"; \
	else \
		echo "uv project already initialized"; \
	fi
	UV_LINK_MODE=copy uv sync

index:
	@uv run python -m src index 

search:
	@uv run python -m src search "OpenAI compatible server" --k 10

search_dataset:
	@uv run python -m src search_dataset \
	data/datasets/UnansweredQuestions/Dataset_2025-09-21_valid_unanswered.json

evaluate:
	@uv run python -m src evaluate \
	data/output/search_results/Dataset_2025-09-21.json \
	data/datasets/Dataset_2025-09-21_valid_answered.json

generate:
	@uv run python -m src generate \
	data/output/search_results/Dataset_2025-09-21_valid.json

answer: 
	@uv run python -m src answer "How to configure OpenAI server?" --k 10

debug:
	@uv run python -m pdb -m src

clean:
	@echo "Removing .venv"
	@rm -rf .venv
	@echo "Removing __pycache__"
	@rm -rf src/__pycache__
	@rm -rf llm_sdk/__pycache__

lint:
	flake8 src

help:
	@uv run python -m src --help

PHONY: install run debug clean lint