import unittest
from decimal import Decimal

from numeric_verifier import (
    verify_assertion,
    build_numeric_claims,
    filter_document_claims,
)

FACTS = {
    (2025, "Total net sales"): Decimal("416161000000"),
    (2025, "Net income"): Decimal("112010000000"),
    (2025, "Operating income"): Decimal("133050000000"),
    (2025, "Total assets"): Decimal("359241000000"),
    (2025, "Total liabilities"): Decimal("285508000000"),
}


class NumericVerifierTests(unittest.TestCase):

    def test_correct_margin(self):
        result = verify_assertion(
            "C2025_NET_MARGIN",
            "26.92",
            "percent",
            FACTS,
            [2025],
            claimed_year=2025
        )
        self.assertEqual(result["status"], "SUPPORTED")

    def test_forged_margin(self):
        result = verify_assertion(
            "C2025_NET_MARGIN",
            "29.92",
            "percent",
            FACTS,
            [2025]
        )
        self.assertEqual(
            result["status"],
            "CONTRADICTED"
        )

    def test_wrong_unit(self):
        result = verify_assertion(
            "C2025_NET_MARGIN",
            "26.92",
            "USD",
            FACTS,
            [2025]
        )
        self.assertEqual(
            result["status"],
            "CONTRADICTED"
        )

    def test_wrong_year(self):
        result = verify_assertion(
            "C2025_NET_MARGIN",
            "26.92",
            "percent",
            FACTS,
            [2025],
            claimed_year=2024
        )
        self.assertEqual(
            result["status"],
            "CONTRADICTED"
        )

    def test_unknown_reference(self):
        result = verify_assertion(
            "C2025_FAKE_METRIC",
            "26.92",
            "percent",
            FACTS,
            [2025]
        )
        self.assertEqual(
            result["status"],
            "INSUFFICIENT"
        )

    def test_deterministic_report(self):
        claims = build_numeric_claims(
            FACTS,
            [2025]
        )
        self.assertEqual(len(claims), 1)
        self.assertIn(
            "26.92%",
            claims[0]["text"]
        )

    def test_reject_llm_numeric_claim(self):
        claims = [{
            "text": "Apple's net margin was 29.92%.",
            "source_ids": ["C2025_NET_MARGIN"]
        }]

        accepted, rejected = filter_document_claims(
            claims
        )

        self.assertEqual(len(accepted), 0)
        self.assertEqual(rejected, 1)

    def test_allow_qualitative_claim(self):
        claims = [{
            "text": (
                "In FY2025 Apple disclosed "
                "supply chain risks."
            ),
            "source_ids": ["D2025_apple-2025-00067"]
        }]

        accepted, rejected = filter_document_claims(
            claims
        )

        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
