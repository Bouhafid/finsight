"""Verifier evaluation: are hallucinated claims rejected and true ones kept?

Each test claim is checked against one real 10-K chunk, exactly as the
pipeline does. The decision is binary, as in production: a claim is kept
only when every proposition is SUPPORTED by a verified quote.
"""

import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from finsight.config import ROOT
from finsight.evaluation import RESULTS_DIR, load_chunks, normalize
from finsight.llm import LLM, get_llm
from finsight.models import Source
from finsight.verification import VerificationError, is_supported, verify_claim

CLAIMS_PATH = ROOT / "eval" / "claims.jsonl"


class EvalClaim(BaseModel):
    id: str
    year: int
    lang: Literal["en", "fr"]
    # Exact text that identifies the 10-K chunk given to the verifier.
    source: str
    expected: Literal["accept", "reject"]
    trap: str
    claim: str


def load_claims(path: Path = CLAIMS_PATH) -> list[EvalClaim]:
    with path.open(encoding="utf-8") as file:
        return [EvalClaim.model_validate_json(line) for line in file if line.strip()]


def find_source(claim: EvalClaim, chunks: list[dict]) -> Source:
    """The first chunk of the claim's year containing its source text."""
    needle = normalize(claim.source)
    for chunk in chunks:
        if needle in normalize(chunk["texte"]):
            return Source(
                id=f"D{claim.year}_{chunk['chunk_id']}",
                kind="DOCUMENT",
                label=f"FY{claim.year} 10-K, {chunk['section']} ({chunk['kind']})",
                url=chunk["source_url"],
                excerpt=chunk["texte"],
            )
    raise ValueError(f"{claim.id}: source text not found in FY{claim.year} chunks")


def check_claim(claim: EvalClaim, source: Source, llm: LLM) -> dict:
    start = time.perf_counter()
    try:
        propositions = verify_claim(claim.claim, {source.id: source}, llm)
        accepted = is_supported(propositions)
        error = None
    except VerificationError as exc:
        # The pipeline drops claims it cannot verify, so this counts as a rejection.
        propositions, accepted, error = [], False, str(exc)

    return {
        "id": claim.id,
        "trap": claim.trap,
        "lang": claim.lang,
        "expected": claim.expected,
        "accepted": accepted,
        "correct": accepted == (claim.expected == "accept"),
        "error": error,
        "seconds": round(time.perf_counter() - start, 2),
        "verdicts": [item["verdict"] for item in propositions],
        "propositions": propositions,
    }


def score(rows: list[dict]) -> dict:
    should_accept = [r for r in rows if r["expected"] == "accept"]
    should_reject = [r for r in rows if r["expected"] == "reject"]
    accepted = [r for r in rows if r["accepted"]]

    def share(part, whole):
        return round(len(part) / len(whole), 3) if whole else None

    by_trap = defaultdict(list)
    for row in rows:
        by_trap[row["trap"]].append(row)

    return {
        "n": len(rows),
        "accuracy": share([r for r in rows if r["correct"]], rows),
        # Hallucinations that would reach the user: the key safety number.
        "false_accept_rate": share([r for r in should_reject if r["accepted"]], should_reject),
        # True claims lost: the price paid for caution.
        "false_reject_rate": share([r for r in should_accept if not r["accepted"]], should_accept),
        "precision": share([r for r in accepted if r["expected"] == "accept"], accepted),
        "errors": sum(r["error"] is not None for r in rows),
        "mean_seconds": round(sum(r["seconds"] for r in rows) / len(rows), 2) if rows else None,
        "by_trap": {
            trap: {"n": len(group), "accuracy": share([r for r in group if r["correct"]], group)}
            for trap, group in sorted(by_trap.items())
        },
    }


def evaluate_verifier(claims: list[EvalClaim], llm: LLM, progress=print) -> dict:
    # Sequential on purpose: latency stays comparable between runs and models.
    chunks = {year: load_chunks(year) for year in {c.year for c in claims}}
    rows = []
    for i, claim in enumerate(claims, start=1):
        row = check_claim(claim, find_source(claim, chunks[claim.year]), llm)
        rows.append(row)
        status = "ok " if row["correct"] else "BAD"
        progress(f"[{i}/{len(claims)}] {status} {claim.id} {claim.trap:<18} {row['verdicts']}")
    return {"model": llm.model, "summary": score(rows), "claims": rows}


def print_report(results: dict) -> None:
    s = results["summary"]
    print(f"\nVerifier model: {results['model']} ({s['n']} claims)")
    print(f"accuracy           {s['accuracy']}")
    print(f"false accept rate  {s['false_accept_rate']}  (hallucinations kept)")
    print(f"false reject rate  {s['false_reject_rate']}  (true claims lost)")
    print(f"precision          {s['precision']}")
    print(f"errors             {s['errors']}")
    print(f"mean seconds       {s['mean_seconds']}")
    print("\nBy trap:")
    for trap, group in s["by_trap"].items():
        print(f"  {trap:<20}{group['n']:>4}{group['accuracy']:>8.2f}")
    wrong = [r["id"] for r in results["claims"] if not r["correct"]]
    print(f"\nWrong decisions: {', '.join(wrong) or 'none'}")


def run(name: str | None, limit: int | None) -> None:
    llm = get_llm()
    claims = load_claims()[:limit]
    results = evaluate_verifier(claims, llm)
    print_report(results)

    name = name or re.sub(r"[^\w.-]", "-", llm.model)
    RESULTS_DIR.mkdir(exist_ok=True)
    output = RESULTS_DIR / f"verifier_{name}.json"
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {output.relative_to(ROOT)}")
