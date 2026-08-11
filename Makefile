.PHONY: run test lint format docker-up docker-down benchmark

run:
	PYTHONPATH=src uvicorn semantic_cache.main:app --reload --host 0.0.0.0 --port 8000

test:
	PYTHONPATH=src pytest -q

lint:
	ruff check src tests load_tests scripts
	ruff format --check src tests load_tests scripts

format:
	ruff format src tests load_tests scripts
	ruff check --fix src tests load_tests scripts

docker-up:
	docker compose up --build

docker-down:
	docker compose down

benchmark:
	PYTHONPATH=src python scripts/run_benchmark.py
