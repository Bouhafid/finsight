import json

import pytest
from conftest import FakeLLM

from finsight.models import Claim
from finsight.verification import (
    InvalidVerdict,
    VerificationError,
    quote_exists,
    validate_result,
    verify_claim,
    verify_document_claims,
)

REF = "D2025_apple-2025-00067"
QUOTE = "faces significant risks of supply shortages and price increases"


def verdict(verdict="SUPPORTED", quote=QUOTE, ref=REF):
    evidence = [{"source_id": ref, "quote": quote}] if quote else []
    return {
        "atomic_claims": [
            {
                "statement": "Apple faces supply risks.",
                "verdict": verdict,
                "evidence": evidence,
                "reason": "Stated in the risk factors.",
            }
        ]
    }


def test_quote_matching_ignores_typography_and_spacing():
    source = "The Company’s  products — and services"
    assert quote_exists("The Company's products - and services", source)


def test_short_or_absent_quotes_do_not_count():
    assert not quote_exists("risks", "risks of supply")
    assert not quote_exists("risks of supply shortages and floods", "risks of supply shortages")


@pytest.mark.parametrize(
    "result",
    [
        verdict(quote="Apple faces no supply risk at all, ever."),  # paraphrased quote
        verdict(ref="D2025_apple-2025-00001"),  # invented source
        verdict(verdict="PROBABLY"),
        verdict(quote=None),  # SUPPORTED without a quote
        {"atomic_claims": []},
    ],
)
def test_untrustworthy_verdicts_are_rejected(documents, result):
    with pytest.raises(InvalidVerdict):
        validate_result(result, documents)


def test_insufficient_needs_no_quote(documents):
    assert validate_result(verdict("INSUFFICIENT", quote=None), documents)


def test_truncated_output_is_retried(documents):
    llm = FakeLLM(('{"atomic_claims": [', "length"), json.dumps(verdict()))

    propositions = verify_claim("Apple faces supply risks.", documents, llm)

    assert propositions[0]["verdict"] == "SUPPORTED"
    assert "cut off" in llm.calls[1][-1]["content"]


def test_three_invalid_outputs_raise(documents):
    with pytest.raises(VerificationError):
        verify_claim("Apple faces supply risks.", documents, FakeLLM("{}", "{}", "{}"))


def test_only_fully_supported_claims_are_kept(documents, monkeypatch):
    monkeypatch.setattr("finsight.verification.VERIFY_WORKERS", 1)  # FakeLLM replies in order
    claims = [
        Claim(text="Apple faces supply risks.", source_ids=[REF]),
        Claim(text="Apple lost money because of shortages.", source_ids=[REF]),
    ]
    llm = FakeLLM(json.dumps(verdict()), json.dumps(verdict("INSUFFICIENT", quote=None)))

    kept = verify_document_claims(claims, documents, llm)

    assert [claim.text for claim in kept] == ["Apple faces supply risks."]
    assert kept[0].quotes[0].quote == QUOTE
