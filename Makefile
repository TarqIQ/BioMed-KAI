.PHONY: setup dev test up pilot

setup:
	pip install -e .

dev:
	uvicorn biomedkai.api.server:app --reload --port 8000

test:
	pytest -q

up:
	docker compose -f docker/docker-compose.yml up --build

pilot:
	biomedkai eval multilingual --langs hi,es,zh --prompts-dir experiments/multilingual_pilot/prompts --out results/multilingual
