import re
from decimal import Decimal, InvalidOperation
from typing import Dict, Tuple

PRECISION = Decimal("0.01")
BILLION = Decimal("1000000000")

METRICS = {
    "Total net sales": "REVENUE",
    "Net income": "NET_INCOME",
    "Operating income": "OPERATING_INCOME",
    "Total assets": "ASSETS",
    "Total liabilities": "LIABILITIES",
}


class NumericVerificationError(Exception):
    pass


def authoritative_values(facts, years):
    """
    Construire la verite numerique a partir des
    donnees SQL/XBRL et des formules Python.

    Retourne:
        reference -> (Decimal, unit)
    """

    values = {}
    ordered_years = sorted(set(years))

    for year in ordered_years:
        for metric, code in METRICS.items():
            key = (year, metric)

            if key not in facts:
                raise NumericVerificationError(
                    f"Donnee XBRL manquante : {key}"
                )

            value = Decimal(str(facts[key]))

            if not value.is_finite():
                raise NumericVerificationError(
                    f"Valeur non finie : {key}"
                )

            values[f"F{year}_{code}"] = (
                value,
                "USD"
            )

        revenue = values[f"F{year}_REVENUE"][0]
        income = values[f"F{year}_NET_INCOME"][0]
        operating = values[f"F{year}_OPERATING_INCOME"][0]

        if revenue <= 0:
            raise NumericVerificationError(
                f"Revenue non positif FY{year}"
            )

        values[f"C{year}_NET_MARGIN"] = (
            (income / revenue * 100).quantize(PRECISION),
            "percent"
        )

        values[f"C{year}_OPERATING_MARGIN"] = (
            (operating / revenue * 100).quantize(PRECISION),
            "percent"
        )

    for previous, current in zip(
        ordered_years,
        ordered_years[1:]
    ):
        if current != previous + 1:
            continue

        old_revenue = values[f"F{previous}_REVENUE"][0]
        new_revenue = values[f"F{current}_REVENUE"][0]

        growth = (
            (new_revenue - old_revenue)
            / old_revenue * 100
        ).quantize(PRECISION)

        values[f"C{current}_REVENUE_GROWTH"] = (
            growth,
            "percent"
        )

    return values


def verify_assertion(
    reference,
    claimed_value,
    claimed_unit,
    facts,
    years,
    claimed_year=None
):
    """
    Verifier une assertion numerique structuree.

    Une reference inconnue ne devient jamais valide.
    Une valeur incorrecte est rejetee.
    """

    truth = authoritative_values(facts, years)

    if reference not in truth:
        return {
            "status": "INSUFFICIENT",
            "reason": "Unknown or unavailable source",
            "reference": reference
        }

    expected, expected_unit = truth[reference]

    if (
        claimed_year is not None
        and reference[1:5].isdigit()
        and int(reference[1:5]) != claimed_year
    ):
        return {
            "status": "CONTRADICTED",
            "reason": "Fiscal year mismatch",
            "reference": reference
        }

    if claimed_unit != expected_unit:
        return {
            "status": "CONTRADICTED",
            "reason": "Unit mismatch",
            "expected_unit": expected_unit,
            "received_unit": claimed_unit,
            "reference": reference
        }

    try:
        received = Decimal(str(claimed_value))
    except (InvalidOperation, ValueError):
        return {
            "status": "INSUFFICIENT",
            "reason": "Invalid numeric value",
            "reference": reference
        }

    if not received.is_finite():
        return {
            "status": "INSUFFICIENT",
            "reason": "Non-finite numeric value",
            "reference": reference
        }

    if received != expected:
        return {
            "status": "CONTRADICTED",
            "reason": "Value mismatch",
            "expected": str(expected),
            "received": str(received),
            "reference": reference
        }

    return {
        "status": "SUPPORTED",
        "reason": "Exact deterministic match",
        "reference": reference,
        "value": str(expected),
        "unit": expected_unit
    }


def build_numeric_claims(facts, years):
    """
    Produire les affirmations numeriques sans LLM.
    Le contenu est derive exclusivement de XBRL.
    """

    truth = authoritative_values(facts, years)
    ordered_years = sorted(set(years))
    claims = []

    for year in ordered_years:
        revenue = truth[f"F{year}_REVENUE"][0]
        income = truth[f"F{year}_NET_INCOME"][0]
        net_margin = truth[f"C{year}_NET_MARGIN"][0]
        operating_margin = truth[
            f"C{year}_OPERATING_MARGIN"
        ][0]

        text = (
            f"FY{year}: revenue "
            f"{revenue / BILLION:.3f} billion USD; "
            f"net income {income / BILLION:.3f} "
            f"billion USD; net margin {net_margin:.2f}%; "
            f"operating margin "
            f"{operating_margin:.2f}%."
        )

        claims.append({
            "text": text,
            "source_ids": [
                f"F{year}_REVENUE",
                f"F{year}_NET_INCOME",
                f"C{year}_NET_MARGIN",
                f"C{year}_OPERATING_MARGIN"
            ]
        })

    for previous, current in zip(
        ordered_years,
        ordered_years[1:]
    ):
        ref = f"C{current}_REVENUE_GROWTH"

        if ref not in truth:
            continue

        value = truth[ref][0]

        claims.append({
            "text": (
                f"Revenue growth FY{previous} "
                f"to FY{current}: {value:.2f}%."
            ),
            "source_ids": [ref]
        })

    return claims


def filter_document_claims(claims):
    """
    N'autoriser les sorties libres du LLM que pour
    des affirmations documentaires qualitatives.

    Ce filtre est conservateur, pas un verificateur
    semantique des passages SEC.
    """

    accepted = []
    rejected = 0

    numeric_pattern = re.compile(
        r"\d|[%$€]|\b("
        r"USD|EUR|percent|percentage|pour cent|"
        r"billion|billionaire|million|milliard|"
        r"zero|one|two|three|four|five|"
        r"six|seven|eight|nine|ten|hundred"
        r")\b",
        flags=re.IGNORECASE
    )

    for claim in claims:
        ids = claim.get("source_ids", [])
        text = claim.get("text", "")

        if (
            not isinstance(text, str)
            or not isinstance(ids, list)
            or not ids
        ):
            rejected += 1
            continue

        # Le LLM n'a pas le droit de generer
        # le bilan chiffre lui-meme.
        if not all(
            isinstance(ref, str) and ref.startswith("D")
            for ref in ids
        ):
            rejected += 1
            continue

        # Autoriser les annees fiscales comme contexte.
        without_years = re.sub(
            r"\b(?:FY\s*)?20(?:23|24|25)\b",
            "",
            text,
            flags=re.IGNORECASE
        )

        if numeric_pattern.search(without_years):
            rejected += 1
            continue

        accepted.append(claim)

    return accepted, rejected
