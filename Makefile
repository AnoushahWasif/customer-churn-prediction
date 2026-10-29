.PHONY: install data train reports lint format test

install:
	uv sync

data:
	uv run python -m churn.data

train:
	uv run churn train

reports:
	uv run churn reports

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest
