PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
UV ?= uv

install:
	$(UV) pip install --python .venv/bin/python --extra-index-url https://download.pytorch.org/whl/cpu -e ".[dev]"

lint:
	$(PY) -m ruff format --check . && $(PY) -m ruff check .

test:
	$(PY) -m pytest -q

serve:
	$(PY) -m docrag serve

.PHONY: install lint test serve
