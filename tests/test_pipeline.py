import json
from contextlib import nullcontext

import pytest
from conftest import FakeLLM

from finsight import pipeline
from finsight.facts import DataIntegrityError, fetch_financial_facts

REF = "D2025_apple-2025-00067"
QUOTE = "faces significant risks of supply shortages and price increases"


@pytest.fixture
def offline(monkeypatch, facts, documents):
    """Run the pipeline without Postgres or the embedding model."""
    monkeypatch.setattr(pipeline.psycopg, "connect", lambda _: nullcontext())
    monkeypatch.setattr(pipeline, "fetch_financial_facts", lambda *_: (facts, {}))
    monkeypatch.setattr(pipeline, "retrieve_documents", lambda *_: list(documents.values()))
    monkeypatch.setattr("finsight.verification.VERIFY_WORKERS", 1)


def test_answer_combines_figures_and_verified_disclosures(offline):
    claims = [
        {"text": "Apple faces supply shortage risks.", "source_ids": [REF]},
        {"text": "Apple's net margin was 26.92%.", "source_ids": [REF]},
    ]
    supported = {
        "atomic_claims": [
            {
                "statement": "Apple faces supply shortage risks.",
                "verdict": "SUPPORTED",
                "evidence": [{"source_id": REF, "quote": QUOTE}],
                "reason": "Risk factors.",
            }
        ]
    }
    llm = FakeLLM(
        json.dumps({"claims": claims, "advice_requested": False, "insufficient_context": False}),
        json.dumps(supported),
    )

    answer = pipeline.answer_question("Supply risks?", [2025], llm=llm)

    assert "26.92%" in answer.numeric_claims[0].text
    assert [claim.text for claim in answer.document_claims] == [
        "Apple faces supply shortage risks."
    ]
    assert answer.stats.model_dump() == {
        "generated": 2,
        "rejected_numeric": 1,
        "rejected_unverified": 0,
    }
    assert "C2025_NET_MARGIN" in answer.sources
    assert answer.warnings == []


def test_failed_generation_still_reports_figures(offline):
    answer = pipeline.answer_question("Supply risks?", [2025], llm=FakeLLM("oops", "oops"))

    assert answer.numeric_claims
    assert answer.document_claims == []
    assert "did not return valid cited claims" in answer.warnings[0]


@pytest.mark.parametrize(
    ("question", "years", "top_k"),
    [("  ", [2025], 2), ("Risks?", [2019], 2), ("Risks?", [2025], 9)],
)
def test_invalid_requests_are_refused(question, years, top_k):
    with pytest.raises(ValueError):
        pipeline.answer_question(question, years, top_k)


class FakeConnection:
    def __init__(self, rows):
        self.rows = rows

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, query, params):
        pass

    def fetchall(self):
        return self.rows


def test_facts_keep_their_sec_provenance(facts):
    rows = [
        (2025, metric, value, "USD", "https://sec.gov/x", "0000320193-25-000079", "us-gaap:X")
        for (_, metric), value in facts.items()
    ]

    values, sources = fetch_financial_facts(FakeConnection(rows), [2025])

    assert values == facts
    assert sources["F2025_REVENUE"].url == "https://sec.gov/x"


def test_facts_without_provenance_stop_the_report(facts):
    rows = [(2025, metric, value, "USD", None, None, None) for (_, metric), value in facts.items()]

    with pytest.raises(DataIntegrityError, match="provenance"):
        fetch_financial_facts(FakeConnection(rows), [2025])
