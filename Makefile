.PHONY: setup data baseline test lint
setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && pip install -e .
data:
	python scripts/fetch_laps.py --config configs/races.yaml
baseline:
	python scripts/run_tyre_baseline.py
test:
	pytest -q
lint:
	ruff check .
