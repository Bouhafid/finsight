import json

import pytest
from conftest import FakeLLM

from finsight.generation import filter_numeric_claims, generate_claims
from finsight.models import Claim

REF = "D2025_apple-2025-00067"


def answer(claims, advice=False, insufficient=False):
    return json.dumps(
        {"claims": claims, "advice_requested": advice, "insufficient_context": insufficient}
    )


def test_valid_claims_are_parsed(documents):
    llm = FakeLLM(answer([{"text": "Apple discloses supply risks.", "source_ids": [f"[{REF}]"]}]))

    draft = generate_claims("Supply risks?", list(documents.values()), llm)

    assert draft.claims == [Claim(text="Apple discloses supply risks.", source_ids=[REF])]
    assert not draft.failed


@pytest.mark.parametrize(
    "bad_output",
    [
        "not json",
        answer([{"text": "Invented source.", "source_ids": ["D2025_apple-2025-99999"]}]),
        answer([{"text": "No citation.", "source_ids": []}]),
        answer([{"text": "Contradiction.", "source_ids": [REF]}], insufficient=True),
        json.dumps({"claims": [], "insufficient_context": "no"}),
    ],
)
def test_invalid_output_is_retried_with_the_error(documents, bad_output):
    fixed = answer([{"text": "Apple discloses supply risks.", "source_ids": [REF]}])
    llm = FakeLLM(bad_output, fixed)

    draft = generate_claims("Supply risks?", list(documents.values()), llm)

    assert len(draft.claims) == 1
    assert "Fix this error" in llm.calls[1][-1]["content"]


def test_two_invalid_outputs_fail_without_claims(documents):
    llm = FakeLLM("not json", "still not json")

    draft = generate_claims("Supply risks?", list(documents.values()), llm)

    assert draft.failed
    assert draft.claims == []


def test_advice_request_is_flagged(documents):
    llm = FakeLLM(answer([], advice=True, insufficient=True))

    draft = generate_claims("How should Apple make more profit?", list(documents.values()), llm)

    assert draft.advice_requested
    assert draft.claims == []


@pytest.mark.parametrize(
    ("text", "kept"),
    [
        ("In FY2025 Apple disclosed supply chain risks.", True),
        ("Apple is one of the largest buyers of components.", True),
        ("Apple's net margin was 29.92%.", False),
        ("Revenue reached four hundred billion.", False),
        ("Services revenue grew by ten percent.", False),
    ],
)
def test_figures_are_filtered_out(text, kept):
    accepted, rejected = filter_numeric_claims([Claim(text=text, source_ids=[REF])])
    assert (len(accepted), rejected) == ((1, 0) if kept else (0, 1))


def test_claims_citing_figures_are_filtered_out():
    claim = Claim(text="Margins improved.", source_ids=["C2025_NET_MARGIN"])
    assert filter_numeric_claims([claim]) == ([], 1)
