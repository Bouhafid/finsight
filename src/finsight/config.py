"""Settings shared by the whole pipeline, read once from the environment."""

import os
from pathlib import Path

from dotenv import load_dotenv
from psycopg.conninfo import make_conninfo

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

COMPANY = "Apple"
FORM_TYPE = "10-K"
AVAILABLE_YEARS = (2023, 2024, 2025)
PIPELINE_VERSION = "structured-v1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# XBRL metric label stored in financial_facts -> code used in source IDs.
METRICS = {
    "Total net sales": "REVENUE",
    "Net income": "NET_INCOME",
    "Operating income": "OPERATING_INCOME",
    "Total assets": "ASSETS",
    "Total liabilities": "LIABILITIES",
}

# Document claims are verified in parallel; keep 1 for a local Ollama.
VERIFY_WORKERS = int(os.getenv("VERIFY_WORKERS", "4"))


def database_conninfo() -> str:
    """DATABASE_URL wins (hosted Postgres); otherwise the local DB_* variables."""
    url = os.getenv("DATABASE_URL")
    if url:
        return url

    return make_conninfo(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=os.getenv("DB_PORT", "5433"),
        dbname=os.getenv("DB_NAME", "finsight"),
        user=os.getenv("DB_USER", "finsight"),
        password=os.getenv("DB_PASSWORD", ""),
    )
