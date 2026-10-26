# FinSight images, built by docker compose:
#   api    — the FastAPI service and demo page (default target)
#   ingest — the one-shot data pipeline (SEC download -> Postgres)
FROM python:3.14-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/huggingface

WORKDIR /app

# CPU-only PyTorch: the default wheel bundles CUDA and adds several GB.
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY src ./src
RUN pip install .

# Bake the embedding model into the image so the container never downloads it.
RUN python -c "from finsight.retrieval import _model; _model()"
ENV HF_HUB_OFFLINE=1

RUN useradd --create-home --uid 1000 finsight


FROM base AS ingest

RUN pip install ".[ingest]"
COPY scripts ./scripts
USER finsight
CMD ["sh", "scripts/ingest.sh"]


FROM base AS api

USER finsight
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=8)"

# Hosting platforms pass the port in $PORT.
CMD ["sh", "-c", "exec uvicorn finsight.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
