from decimal import Decimal

import pytest

from finsight.models import Source


class FakeLLM:
    """Replays canned JSON answers and records the messages it was sent."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def complete_json(self, messages, max_tokens, schema=None):
        self.calls.append(list(messages))
        response = self.responses.pop(0)
        if isinstance(response, tuple):
            return response
        return response, "stop"


FACTS_2025 = {
    (2025, "Total net sales"): Decimal("416161000000"),
    (2025, "Net income"): Decimal("112010000000"),
    (2025, "Operating income"): Decimal("133050000000"),
    (2025, "Total assets"): Decimal("359241000000"),
    (2025, "Total liabilities"): Decimal("285508000000"),
}

FACTS_2024 = {
    (2024, "Total net sales"): Decimal("391035000000"),
    (2024, "Net income"): Decimal("93736000000"),
    (2024, "Operating income"): Decimal("123216000000"),
    (2024, "Total assets"): Decimal("364980000000"),
    (2024, "Total liabilities"): Decimal("308030000000"),
}

SUPPLY_TEXT = (
    "The Company depends on component and product manufacturing and logistical "
    "services provided by outsourcing partners, many of which are located outside "
    "of the U.S. The Company faces significant risks of supply shortages and "
    "price increases."
)


@pytest.fixture
def facts():
    return dict(FACTS_2025)


@pytest.fixture
def documents():
    return {
        "D2025_apple-2025-00067": Source(
            id="D2025_apple-2025-00067",
            kind="DOCUMENT",
            label="FY2025 10-K, Risk Factors (paragraph)",
            url="https://www.sec.gov/example",
            excerpt=SUPPLY_TEXT,
            score=0.71,
        )
    }
