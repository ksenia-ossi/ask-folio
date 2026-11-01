PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
UV ?= uv

install:
	@if [ ! -x .venv/bin/python ]; then $(UV) venv .venv; fi
	$(UV) pip install --python .venv/bin/python --extra-index-url https://download.pytorch.org/whl/cpu -e ".[dev]"

lint:
	$(PY) -m ruff format --check . && $(PY) -m ruff check .

test:
	$(PY) -m pytest -q

eval-data:
	mkdir -p eval/docs
	curl -fsSL -o eval/docs/IAR25.pdf "https://www.pernod-ricard.com/sites/default/files/inline-files/Pernod_Ricard_IAR25_ENGLISH.pdf"
	curl -fsSL -o eval/docs/FY26.pdf "https://www.pernod-ricard.com/system/files/2026-08/Pernod%20Ricard%20FY26%20Financial%20Communication.pdf"

eval: eval-data
	ASKFOLIO_INDEX=eval/.index $(PY) -m askfolio ingest eval/docs
	ASKFOLIO_INDEX=eval/.index $(PY) -m askfolio sweep --questions eval/questions.jsonl
	ASKFOLIO_INDEX=eval/.index $(PY) -m askfolio eval --questions eval/questions.jsonl
	cp eval/.index/eval/*.json* eval/results/

serve:
	$(PY) -m askfolio serve

.PHONY: install lint test eval-data eval serve
