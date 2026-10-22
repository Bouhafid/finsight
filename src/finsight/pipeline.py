"""End-to-end answer: XBRL figures + retrieved excerpts + verified LLM claims."""

import logging

import psycopg

from finsight.config import AVAILABLE_YEARS, database_conninfo
from finsight.facts import fetch_financial_facts
from finsight.generation import filter_numeric_claims, generate_claims
from finsight.llm import LLM
from finsight.metrics import authoritative_values, build_numeric_claims, calculation_sources
from finsight.models import Answer, Stats
from finsight.retrieval import retrieve_documents
from finsight.verification import verify_document_claims

log = logging.getLogger(__name__)

MAX_TOP_K = 5


def answer_question(
    question: str,
    years=AVAILABLE_YEARS,
    top_k: int = 2,
    llm: LLM | None = None,
) -> Answer:
    question = question.strip()
    years = sorted(set(years))

    if not question:
        raise ValueError("The question is empty")
    if not years or not set(years) <= set(AVAILABLE_YEARS):
        raise ValueError(f"Years must be among {list(AVAILABLE_YEARS)}")
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")

    with psycopg.connect(database_conninfo()) as conn:
        log.info("1/4 Loading XBRL facts for FY%s", years)
        facts, sources = fetch_financial_facts(conn, years)

        log.info("2/4 Searching 10-K excerpts")
        documents = retrieve_documents(conn, question, years, top_k)

    sources |= calculation_sources(authoritative_values(facts, years))
    sources |= {doc.id: doc for doc in documents}
    numeric_claims = build_numeric_claims(facts, years)

    log.info("3/4 Generating qualitative claims")
    draft = generate_claims(question, documents, llm)
    claims, rejected_numeric = filter_numeric_claims(draft.claims)

    log.info("4/4 Verifying %d claim(s) against exact quotes", len(claims))
    verified = verify_document_claims(claims, sources, llm)

    warnings = []
    if draft.failed:
        warnings.append(
            "The language model did not return valid cited claims; "
            "only the verified figures are reported."
        )
    if draft.advice_requested:
        warnings.append(
            "FinSight reports what Apple discloses in its 10-K filings; "
            "it does not give investment or strategy advice."
        )

    return Answer(
        question=question,
        years=years,
        numeric_claims=numeric_claims,
        document_claims=verified,
        sources=sources,
        advice_requested=draft.advice_requested,
        warnings=warnings,
        stats=Stats(
            generated=len(draft.claims),
            rejected_numeric=rejected_numeric,
            rejected_unverified=len(claims) - len(verified),
        ),
    )
