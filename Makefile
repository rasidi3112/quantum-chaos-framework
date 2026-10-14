PYTHON ?= python

.PHONY: install test test-slow coverage lint typecheck figures validate docs clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

test-slow:
	$(PYTHON) -m pytest -m slow

coverage:
	$(PYTHON) -m pytest --cov=quantum_chaos --cov-fail-under=85

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy

figures:  ## regenerate every figure from configs/figures/*.yaml (~1 min)
	$(PYTHON) scripts/make_figures.py

validate:  ## level-statistics validation tables (~40 min)
	$(PYTHON) scripts/validate_level_statistics.py

docs:
	mkdocs build --strict

clean:
	rm -rf site .pytest_cache .mypy_cache .ruff_cache .coverage
