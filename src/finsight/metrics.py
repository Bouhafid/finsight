"""Deterministic financial figures: every number in a report comes from here.

The LLM never writes figures. Ratios and growth are computed in Python from
XBRL facts, so each number can be traced back to SEC data and checked exactly.
"""

from decimal import Decimal, InvalidOperation
from itertools import pairwise

from finsight.config import METRICS
from finsight.models import Claim, Source

PRECISION = Decimal("0.01")
BILLION = Decimal("1000000000")

# Calculated indicator -> the fact codes it is derived from.
FORMULAS = {
    "NET_MARGIN": ("NET_INCOME", "REVENUE"),
    "OPERATING_MARGIN": ("OPERATING_INCOME", "REVENUE"),
}

Facts = dict[tuple[int, str], Decimal]


class NumericVerificationError(Exception):
    pass


def authoritative_values(facts: Facts, years) -> dict[str, tuple[Decimal, str]]:
    """Map every source ID (facts and calculations) to its (value, unit)."""
    values = {}
    ordered_years = sorted(set(years))

    for year in ordered_years:
        for metric, code in METRICS.items():
            if (year, metric) not in facts:
                raise NumericVerificationError(f"Missing XBRL fact: FY{year} {metric}")

            value = Decimal(str(facts[(year, metric)]))
            if not value.is_finite():
                raise NumericVerificationError(f"Non-finite value: FY{year} {metric}")

            values[f"F{year}_{code}"] = (value, "USD")

        revenue = values[f"F{year}_REVENUE"][0]
        if revenue <= 0:
            raise NumericVerificationError(f"Non-positive revenue: FY{year}")

        for name, (numerator, denominator) in FORMULAS.items():
            ratio = values[f"F{year}_{numerator}"][0] / values[f"F{year}_{denominator}"][0]
            values[f"C{year}_{name}"] = ((ratio * 100).quantize(PRECISION), "percent")

    for previous, current in _consecutive(ordered_years):
        old = values[f"F{previous}_REVENUE"][0]
        new = values[f"F{current}_REVENUE"][0]
        growth = ((new - old) / old * 100).quantize(PRECISION)
        values[f"C{current}_REVENUE_GROWTH"] = (growth, "percent")

    return values


def dependencies(ref: str) -> list[str]:
    """Fact IDs a calculation ID is derived from, e.g. C2025_NET_MARGIN."""
    year = int(ref[1:5])
    name = ref[6:]

    if name == "REVENUE_GROWTH":
        return [f"F{year - 1}_REVENUE", f"F{year}_REVENUE"]

    return [f"F{year}_{code}" for code in FORMULAS[name]]


def calculation_sources(values: dict[str, tuple[Decimal, str]]) -> dict[str, Source]:
    sources = {}

    for ref, (value, _) in values.items():
        if not ref.startswith("C"):
            continue

        name = ref[6:].replace("_", " ").lower()
        sources[ref] = Source(
            id=ref,
            kind="CALCULATION",
            label=f"FY{ref[1:5]} {name}: {value}%",
            dependencies=dependencies(ref),
        )

    return sources


def verify_assertion(
    reference, claimed_value, claimed_unit, facts, years, claimed_year=None
) -> dict:
    """Check a structured numeric assertion against the authoritative values.

    An unknown reference is never valid; any mismatch is rejected.
    """
    truth = authoritative_values(facts, years)

    if reference not in truth:
        return {
            "status": "INSUFFICIENT",
            "reason": "Unknown or unavailable source",
            "reference": reference,
        }

    expected, expected_unit = truth[reference]

    if claimed_year is not None and int(reference[1:5]) != claimed_year:
        return {
            "status": "CONTRADICTED",
            "reason": "Fiscal year mismatch",
            "reference": reference,
        }

    if claimed_unit != expected_unit:
        return {
            "status": "CONTRADICTED",
            "reason": "Unit mismatch",
            "expected_unit": expected_unit,
            "received_unit": claimed_unit,
            "reference": reference,
        }

    try:
        received = Decimal(str(claimed_value))
    except (InvalidOperation, ValueError):
        return {
            "status": "INSUFFICIENT",
            "reason": "Invalid numeric value",
            "reference": reference,
        }

    if not received.is_finite():
        return {
            "status": "INSUFFICIENT",
            "reason": "Non-finite numeric value",
            "reference": reference,
        }

    if received != expected:
        return {
            "status": "CONTRADICTED",
            "reason": "Value mismatch",
            "expected": str(expected),
            "received": str(received),
            "reference": reference,
        }

    return {
        "status": "SUPPORTED",
        "reason": "Exact deterministic match",
        "reference": reference,
        "value": str(expected),
        "unit": expected_unit,
    }


def build_numeric_claims(facts: Facts, years) -> list[Claim]:
    """Write the figures of the report without any LLM."""
    truth = authoritative_values(facts, years)
    ordered_years = sorted(set(years))
    claims = []

    for year in ordered_years:
        revenue = truth[f"F{year}_REVENUE"][0]
        income = truth[f"F{year}_NET_INCOME"][0]
        net_margin = truth[f"C{year}_NET_MARGIN"][0]
        operating_margin = truth[f"C{year}_OPERATING_MARGIN"][0]

        claims.append(
            Claim(
                text=(
                    f"FY{year}: revenue {revenue / BILLION:.3f} billion USD; "
                    f"net income {income / BILLION:.3f} billion USD; "
                    f"net margin {net_margin:.2f}%; "
                    f"operating margin {operating_margin:.2f}%."
                ),
                source_ids=[
                    f"F{year}_REVENUE",
                    f"F{year}_NET_INCOME",
                    f"C{year}_NET_MARGIN",
                    f"C{year}_OPERATING_MARGIN",
                ],
            )
        )

    for previous, current in _consecutive(ordered_years):
        ref = f"C{current}_REVENUE_GROWTH"
        claims.append(
            Claim(
                text=f"Revenue growth FY{previous} to FY{current}: {truth[ref][0]:.2f}%.",
                source_ids=[ref],
            )
        )

    return claims


def _consecutive(ordered_years):
    """Pairs of back-to-back fiscal years; growth across a gap is not reported."""
    return [
        (previous, current)
        for previous, current in pairwise(ordered_years)
        if current == previous + 1
    ]
