PY ?= .venv/bin/python
export MLFLOW_DISABLE_AGENT_HINT=1

.PHONY: setup data models golden eval eval-llm test serve mlflow results all

setup:            ## create the venv and install
	uv venv .venv --python 3.11 && uv pip install --python $(PY) -e ".[dev]"

data:             ## generate the SYNTHETIC dataset (seed 42)
	$(PY) -m copilot.data.generate --seed 42 --out data/saas.duckdb

models:           ## survival, XGBoost propensity, power analysis + A/B simulation -> MLflow
	$(PY) -m copilot.models.run_all

golden:           ## rebuild golden + held-out question sets from the reference implementation
	$(PY) -m copilot.agent.golden --set golden
	$(PY) -m copilot.agent.golden --set holdout

eval:             ## agent eval, no API key, no LLM
	$(PY) -m copilot.agent.evaluate --backend offline --golden eval/golden_set.jsonl
	$(PY) -m copilot.agent.evaluate --backend offline --golden eval/holdout_set.jsonl

eval-llm:         ## agent eval with a local Ollama model and LLM-as-a-Judge (no API key; needs `ollama pull llama3.2`)
	$(PY) -m copilot.agent.evaluate --backend ollama --judge ollama --tool-schema v1 --no-grounding --golden eval/golden_set.jsonl
	$(PY) -m copilot.agent.evaluate --backend ollama --judge ollama --tool-schema v2 --no-grounding --golden eval/golden_set.jsonl
	$(PY) -m copilot.agent.evaluate --backend ollama --judge ollama --tool-schema v2 --golden eval/golden_set.jsonl
	$(PY) -m copilot.agent.evaluate --backend ollama --judge ollama --tool-schema v2 --golden eval/holdout_set.jsonl

test:
	$(PY) -m pytest -q

results:          ## print the README results table from artifacts/
	$(PY) scripts/results_table.py

serve:            ## run the API (set the variables from .env.example first)
	$(PY) -m uvicorn copilot.api.main:app_factory --factory --port 8000

mlflow:           ## browse the logged runs
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db

all: data models golden eval eval-llm test results
