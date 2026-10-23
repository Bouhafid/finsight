"""HTTP API and demo page: uvicorn finsight.api:app"""

import logging
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from threading import Lock

import psycopg
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from finsight.config import AVAILABLE_YEARS, database_conninfo
from finsight.facts import DataIntegrityError
from finsight.models import Answer
from finsight.pipeline import MAX_TOP_K, answer_question
from finsight.retrieval import RetrievalError

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)

STATIC = Path(__file__).parent / "static"

# Each question costs several LLM calls: cap them per client on the public demo.
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_HOUR", "20"))
CACHE_SIZE = 200

app = FastAPI(
    title="FinSight",
    description=(
        "Evidence-grounded answers on Apple's 10-K filings. Figures are computed "
        "from XBRL data; text claims are kept only when backed by verbatim quotes."
    ),
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    years: list[int] = Field(default=list(AVAILABLE_YEARS), min_length=1)
    top_k: int = Field(default=2, ge=1, le=MAX_TOP_K)

    def cache_key(self) -> tuple:
        return (" ".join(self.question.lower().split()), tuple(sorted(set(self.years))), self.top_k)


class RateLimiter:
    def __init__(self, limit: int, window: float = 3600):
        self.limit = limit
        self.window = window
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = Lock()

    def allow(self, client: str) -> bool:
        now = time.monotonic()
        with self.lock:
            hits = self.hits[client]
            while hits and now - hits[0] > self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True


limiter = RateLimiter(RATE_LIMIT)
cache: dict[tuple, Answer] = {}


def client_ip(request: Request) -> str:
    # Behind the hosting proxy the real client is the first forwarded address.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.post("/api/ask")
def ask(body: AskRequest, request: Request) -> Answer:
    key = body.cache_key()
    if key in cache:
        return cache[key]

    if not limiter.allow(client_ip(request)):
        raise HTTPException(429, f"Demo limit reached ({RATE_LIMIT} questions per hour).")

    try:
        answer = answer_question(body.question, body.years, body.top_k)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except (DataIntegrityError, RetrievalError, psycopg.Error) as error:
        log.exception("Data unavailable")
        raise HTTPException(503, "Financial data is temporarily unavailable.") from error

    if len(cache) >= CACHE_SIZE:
        cache.pop(next(iter(cache)))
    cache[key] = answer
    return answer


@app.get("/api/health")
def health() -> dict:
    try:
        with psycopg.connect(database_conninfo(), connect_timeout=5) as conn:
            conn.execute("SELECT 1")
    except psycopg.Error as error:
        raise HTTPException(503, "Database unreachable") from error
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")
