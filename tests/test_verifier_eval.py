import json
from collections import Counter

import pytest
from conftest import SUPPLY_TEXT, FakeLLM

from finsight.config import ROOT
from finsight.evaluation import load_chunks
from finsight.generation import FISCAL_YEAR, NUMERIC_PATTERN
from finsight.models import Source
from finsight.verifier_eval import EvalClaim, check_claim, find_source, load_claims, score

CLAIMS = load_claims()
HAS_CHUNKS = (ROOT / "data" / "processed" / "apple").exists()
QUOTE = "faces significant risks of supply shortages and price increases"


def _claim(expected="accept", trap="true"):
    return EvalClaim(
        id="c", year=2025, lang="en", source="supply", expected=expected, trap=trap,
        claim="Apple faces supply risks.",
    )  # fmt: skip


def _source():
    return Source(id="D2025_x", kind="DOCUMENT", label="test", excerpt=SUPPLY_TEXT)


def _verdict(verdict, quote=QUOTE):
    evidence = [{"source_id": "D2025_x", "quote": quote}] if quote else []
    return json.dumps(
        {
            "atomic_claims": [
                {"statement": "s", "verdict": verdict, "evidence": evidence, "reason": "r"}
            ]
        }
    )


def test_claim_ids_are_unique_and_both_decisions_are_tested():
    assert all(n == 1 for n in Counter(c.id for c in CLAIMS).values())
    assert {c.expected for c in CLAIMS} == {"accept", "reject"}


def test_claims_contain_no_figures():
    """The pipeline drops figures before verification, so test claims have none."""
    for claim in CLAIMS:
        assert not NUMERIC_PATTERN.search(FISCAL_YEAR.sub("", claim.claim)), claim.id


@pytest.mark.skipif(not HAS_CHUNKS, reason="data/processed is not versioned")
def test_every_claim_source_is_found():
    chunks = {year: load_chunks(year) for year in {c.year for c in CLAIMS}}
    for claim in CLAIMS:
        assert find_source(claim, chunks[claim.year]).excerpt


def test_supported_claim_is_accepted():
    row = check_claim(_claim(), _source(), FakeLLM(_verdict("SUPPORTED")))
    assert row["accepted"] and row["correct"]
    assert row["verdicts"] == ["SUPPORTED"]


def test_insufficient_claim_is_rejected():
    row = check_claim(_claim("reject"), _source(), FakeLLM(_verdict("INSUFFICIENT", quote=None)))
    assert not row["accepted"] and row["correct"]


def test_invented_quote_is_rejected_even_if_the_model_says_supported():
    fake = _verdict("SUPPORTED", quote="Apple has no supply risk whatsoever in any market.")
    row = check_claim(_claim("reject"), _source(), FakeLLM(fake, fake, fake))
    assert not row["accepted"]
    assert row["error"]


def test_score_counts_false_accepts_and_false_rejects():
    rows = [
        {"expected": "accept", "accepted": True, "correct": True},
        {"expected": "accept", "accepted": False, "correct": False},
        {"expected": "reject", "accepted": True, "correct": False},
        {"expected": "reject", "accepted": False, "correct": True},
        {"expected": "reject", "accepted": False, "correct": True},
    ]
    for row in rows:
        row.update(trap="t", error=None, seconds=1.0)

    summary = score(rows)
    assert summary["accuracy"] == 0.6
    assert summary["false_accept_rate"] == 0.333
    assert summary["false_reject_rate"] == 0.5
    assert summary["precision"] == 0.5
