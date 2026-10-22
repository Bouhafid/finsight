from decimal import Decimal

import pytest
from conftest import FACTS_2024

from finsight.metrics import (
    NumericVerificationError,
    authoritative_values,
    build_numeric_claims,
    calculation_sources,
    verify_assertion,
)


def test_correct_margin_is_supported(facts):
    result = verify_assertion("C2025_NET_MARGIN", "26.92", "percent", facts, [2025], 2025)
    assert result["status"] == "SUPPORTED"


@pytest.mark.parametrize(
    ("value", "unit", "year", "status"),
    [
        ("29.92", "percent", None, "CONTRADICTED"),  # forged value
        ("26.92", "USD", None, "CONTRADICTED"),  # wrong unit
        ("26.92", "percent", 2024, "CONTRADICTED"),  # wrong fiscal year
        ("abc", "percent", None, "INSUFFICIENT"),  # not a number
        ("NaN", "percent", None, "INSUFFICIENT"),
    ],
)
def test_wrong_assertions_are_rejected(facts, value, unit, year, status):
    result = verify_assertion("C2025_NET_MARGIN", value, unit, facts, [2025], year)
    assert result["status"] == status


def test_unknown_reference_is_never_supported(facts):
    result = verify_assertion("C2025_FAKE_METRIC", "26.92", "percent", facts, [2025])
    assert result["status"] == "INSUFFICIENT"


def test_missing_fact_stops_the_report(facts):
    del facts[(2025, "Net income")]
    with pytest.raises(NumericVerificationError):
        authoritative_values(facts, [2025])


def test_growth_needs_consecutive_years(facts):
    values = authoritative_values(facts | FACTS_2024, [2024, 2025])
    assert values["C2025_REVENUE_GROWTH"] == (Decimal("6.43"), "percent")

    gap = {(2023, metric): value for (_, metric), value in FACTS_2024.items()}
    assert "C2025_REVENUE_GROWTH" not in authoritative_values(facts | gap, [2023, 2025])


def test_numeric_report_is_deterministic(facts):
    claims = build_numeric_claims(facts | FACTS_2024, [2025, 2024])

    assert [claim.text for claim in claims] == [
        "FY2024: revenue 391.035 billion USD; net income 93.736 billion USD; "
        "net margin 23.97%; operating margin 31.51%.",
        "FY2025: revenue 416.161 billion USD; net income 112.010 billion USD; "
        "net margin 26.92%; operating margin 31.97%.",
        "Revenue growth FY2024 to FY2025: 6.43%.",
    ]


def test_calculations_cite_their_inputs(facts):
    sources = calculation_sources(authoritative_values(facts | FACTS_2024, [2024, 2025]))

    assert sources["C2025_NET_MARGIN"].dependencies == ["F2025_NET_INCOME", "F2025_REVENUE"]
    assert sources["C2025_REVENUE_GROWTH"].dependencies == ["F2024_REVENUE", "F2025_REVENUE"]
