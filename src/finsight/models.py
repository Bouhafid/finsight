"""Data returned by the pipeline, shared by the CLI and the API."""

from typing import Literal

from pydantic import BaseModel


class Source(BaseModel):
    id: str
    kind: Literal["XBRL", "CALCULATION", "DOCUMENT"]
    label: str
    url: str | None = None
    dependencies: list[str] = []
    excerpt: str | None = None
    score: float | None = None


class Quote(BaseModel):
    source_id: str
    quote: str


class Claim(BaseModel):
    text: str
    source_ids: list[str]
    quotes: list[Quote] = []


class Stats(BaseModel):
    generated: int = 0
    rejected_numeric: int = 0
    rejected_unverified: int = 0


class Answer(BaseModel):
    question: str
    years: list[int]
    numeric_claims: list[Claim]
    document_claims: list[Claim]
    sources: dict[str, Source]
    advice_requested: bool = False
    warnings: list[str] = []
    stats: Stats = Stats()
