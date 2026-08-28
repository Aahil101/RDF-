# ---------------------------------------------------------------------------
# VeriRAG container. Slim, non-root, no build toolchain in the final image.
# Everything inside is free/open-source; no API key is required to run.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONIOENCODING=utf-8 \
    VERIRAG_DATA_DIR=data

WORKDIR /app

# Dependencies first so the layer caches independently of source changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY app.py cli.py ./

RUN pip install --no-cache-dir --no-deps -e .

# Build the sample corpus and index it so the image is demo-ready offline.
RUN python -m scripts.make_sample_pdfs \
    && python cli.py --provider extractive ingest data/raw

# Run as a non-root user; keep data writable for uploads and new sessions.
RUN useradd --create-home --uid 10001 verirag \
    && chown -R verirag:verirag /app
USER verirag

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8501/_stcore/health').read()"

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
