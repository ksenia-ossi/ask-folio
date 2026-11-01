PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
UV ?= uv

install:
	@if [ ! -x .venv/bin/python ]; then $(UV) venv .venv; fi
	$(UV) pip install --python .venv/bin/python --extra-index-url https://download.pytorch.org/whl/cpu -e ".[dev]"

lint:
	$(PY) -m ruff format --check . && $(PY) -m ruff check .

test:
	$(PY) -m pytest -q

serve:
	$(PY) -m askfolio serve

.PHONY: install lint test serve
