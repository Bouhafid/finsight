"""Command line: finsight "What are Apple's main supply chain risks?" """

import argparse
import logging

from finsight.config import AVAILABLE_YEARS
from finsight.models import Answer
from finsight.pipeline import MAX_TOP_K, answer_question


def print_answer(answer: Answer) -> None:
    print(f"\nQuestion: {answer.question}")
    print(f"Fiscal years: {', '.join(f'FY{year}' for year in answer.years)}\n")

    for warning in answer.warnings:
        print(f"! {warning}")

    print("\n== Figures (computed from XBRL, no LLM) ==")
    for claim in answer.numeric_claims:
        print(f"- {claim.text} {_citations(claim.source_ids)}")

    stats = answer.stats
    print(
        f"\n== Disclosures ({len(answer.document_claims)}/{stats.generated} "
        "LLM claims verified by exact quote) =="
    )
    for claim in answer.document_claims:
        print(f"- {claim.text} {_citations(claim.source_ids)}")
        for quote in claim.quotes:
            print(f'    > "{quote.quote}" [{quote.source_id}]')

    if not answer.document_claims:
        print("- No disclosure could be verified for this question.")

    cited = {
        ref for claim in answer.numeric_claims + answer.document_claims for ref in claim.source_ids
    }
    print("\n== Sources ==")
    for ref in sorted(cited):
        source = answer.sources[ref]
        print(f"[{ref}] {source.label}")
        if source.dependencies:
            print(f"    from {', '.join(source.dependencies)}")
        if source.url:
            print(f"    {source.url}")


def _citations(refs: list[str]) -> str:
    return " ".join(f"[{ref}]" for ref in refs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evidence-grounded answers on Apple's 10-K filings."
    )
    parser.add_argument("question")
    parser.add_argument(
        "--years", nargs="+", type=int, choices=AVAILABLE_YEARS, default=AVAILABLE_YEARS
    )
    parser.add_argument("--top-k", type=int, default=2, choices=range(1, MAX_TOP_K + 1))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    # Progress steps are worth seeing even without --verbose.
    logging.getLogger("finsight.pipeline").setLevel(logging.INFO)

    print_answer(answer_question(args.question, args.years, args.top_k))


if __name__ == "__main__":
    main()
