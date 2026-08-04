.PHONY: install test lint eval eval-full ingest serve docker clean

PY ?= python3
export PYTHONPATH := src

install:
	pip install -e ".[dev,serve]"

test:
	FINRAG_EMBEDDING_BACKEND=hash $(PY) -m pytest tests -q

lint:
	ruff check src tests

eval:
	FINRAG_EMBEDDING_BACKEND=hash $(PY) -m finrag.eval.run_eval

eval-full:
	FINRAG_EMBEDDING_BACKEND=sentence-transformers $(PY) -m finrag.eval.run_eval --rerank

ingest:
	$(PY) -m finrag.ingest.cli --tickers $(TICKERS) --form 10-K --years 3

serve:
	uvicorn finrag.api.main:app --reload --port 8000

docker:
	docker build -f docker/Dockerfile -t finrag:local .

clean:
	rm -rf artifacts .pytest_cache **/__pycache__
