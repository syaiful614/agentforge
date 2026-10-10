.PHONY: install lint type test eval all
install:
	pip install -e ".[dev]"
lint:
	ruff check . && ruff format --check .
type:
	mypy src
test:
	pytest -q --cov=agentforge --cov-report=term-missing
eval:
	python evals/run_evals.py
all: lint type test
