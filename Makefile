.PHONY: install lint format typecheck test eval compare check ingest serve docker-build docker-up clean

install:  ## Install all extras + dev tools from the lockfile
	uv sync --all-extras
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest --cov

eval:  ## Golden-set evaluation with the same gate CI uses
	uv run retail-rag eval --retriever hybrid \
		--fail-under hit_rate=0.95 --fail-under recall=0.93 --fail-under mrr=0.90 \
		--fail-under refusal_accuracy=0.25 --fail-over false_refusal_rate=0.0

compare:  ## Evaluate every retriever and refresh the committed baselines
	for r in tfidf bm25 dense hybrid; do \
		uv run retail-rag eval --retriever $$r --output-dir evals/baselines --stem $${r}_local || exit 1; \
	done

check: lint typecheck test eval  ## Everything CI runs, locally

ingest:
	uv run retail-rag ingest

serve: ingest
	uv run retail-rag serve

docker-build:
	docker build -t australian-retail-rag:local .

docker-up:
	docker compose up --build

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov reports data/index
