.PHONY: help install install-ml corpus index demo ask ui test eval clean docker docker-run

PY ?= python

help:           ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:        ## install runtime dependencies + package
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install --no-deps -e .

install-ml:     ## add neural embeddings + cross-encoder reranking (optional, free)
	$(PY) -m pip install -r requirements-ml.txt

corpus:         ## generate the sample PDFs
	$(PY) -m scripts.make_sample_pdfs

index:          ## index everything in data/raw
	$(PY) cli.py ingest data/raw

demo:           ## full end-to-end demonstration with proof images
	$(PY) cli.py demo

ask:            ## ask a question: make ask Q="what is the rent?"
	$(PY) cli.py ask "$(Q)" --trace

ui:             ## launch the Streamlit interface
	streamlit run app.py

test:           ## run the test suite
	pytest -q

eval:           ## grade retrieval, answers and refusals
	$(PY) cli.py eval --json data/eval_report.json

clean:          ## remove generated artefacts (keeps the sample PDFs)
	rm -rf data/index data/proof_cache data/uploads .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

docker:         ## build the container image
	docker build -t verirag .

docker-run:     ## run the UI in a container on :8501
	docker run --rm -p 8501:8501 verirag
