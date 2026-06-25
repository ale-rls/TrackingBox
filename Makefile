.PHONY: install install-ml dev test lint demo serve benchmark

install:
	pip install -e .

install-ml:
	pip install -e ".[ml]"

dev:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests

demo:
	audience-tracker demo --people 24 --frames 200

serve:
	audience-tracker serve --backend mock --port 8000

benchmark:
	audience-tracker benchmark --frames 300 --output benchmarks/report.json
