"""Evaluation set and retrieval metrics: finsight-eval

Relevance is defined by evidence text, not by chunk IDs, so the same set
still applies when the chunking or the embedding model changes.
"""

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

from finsight.config import EMBEDDING_MODEL, ROOT

QUESTIONS_PATH = ROOT / "eval" / "questions.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"
CHUNKS_DIR = ROOT / "data" / "processed" / "apple"

KS = (1, 2, 3, 5, 10)

# (question, year, depth) -> chunk texts, best first.
Retriever = Callable[[str, int, int], list[str]]


class EvalQuestion(BaseModel):
    id: str
    year: int
    lang: Literal["en", "fr"]
    category: str
    question: str
    # Empty for questions the 10-K cannot answer (abstention cases).
    evidence: list[str]

    @property
    def answerable(self) -> bool:
        return bool(self.evidence)


def load_questions(path: Path = QUESTIONS_PATH) -> list[EvalQuestion]:
    with path.open(encoding="utf-8") as file:
        return [EvalQuestion.model_validate_json(line) for line in file if line.strip()]


def normalize(text: str) -> str:
    """Lowercase, straighten quotes and collapse whitespace before matching."""
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip().lower()


def is_relevant(chunk_text: str, question: EvalQuestion) -> bool:
    chunk = normalize(chunk_text)
    return any(normalize(snippet) in chunk for snippet in question.evidence)


def load_chunks(year: int, data_dir: Path = CHUNKS_DIR) -> list[dict]:
    with (data_dir / f"chunks_structured_{year}.jsonl").open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def first_relevant_rank(texts: list[str], question: EvalQuestion) -> int | None:
    """1-based rank of the first relevant chunk, or None if none was retrieved."""
    for rank, text in enumerate(texts, start=1):
        if is_relevant(text, question):
            return rank
    return None


def summarize(ranks: list[int | None], ks=KS) -> dict:
    """Hit@k (share of questions with a relevant chunk in the top k) and MRR."""
    n = len(ranks)
    if not n:
        return {"n": 0}
    summary = {"n": n}
    for k in ks:
        summary[f"hit@{k}"] = sum(r is not None and r <= k for r in ranks) / n
    summary["mrr"] = sum(1 / r for r in ranks if r is not None) / n
    return summary


def evaluate(retriever: Retriever, questions: list[EvalQuestion], ks=KS) -> dict:
    """Score a retriever on the answerable questions, overall and by group."""
    depth = max(ks)
    per_question = []
    groups: dict[str, list] = defaultdict(list)

    for q in questions:
        if not q.answerable:
            continue
        rank = first_relevant_rank(retriever(q.question, q.year, depth), q)
        per_question.append({"id": q.id, "rank": rank})
        for group in ("lang", "category", "year"):
            groups[f"{group}={getattr(q, group)}"].append(rank)

    ranks = [row["rank"] for row in per_question]
    return {
        "overall": summarize(ranks, ks),
        "groups": {name: summarize(r, ks) for name, r in sorted(groups.items())},
        "questions": per_question,
    }


class DenseRetriever:
    """Exact cosine search over the chunk embeddings stored in data/processed.

    These are the vectors inserted in Postgres, so this mirrors the production
    search (which uses an approximate HNSW index) without needing the database.
    """

    def __init__(self, data_dir: Path = CHUNKS_DIR):
        self.data_dir = data_dir
        self.cache: dict[int, tuple[list[str], np.ndarray]] = {}

    def _year(self, year: int) -> tuple[list[str], np.ndarray]:
        if year not in self.cache:
            manifest = json.loads(
                (self.data_dir / f"embeddings_manifest_{year}.json").read_text(encoding="utf-8")
            )
            if manifest["model"] != EMBEDDING_MODEL:
                raise ValueError(f"FY{year} embeddings were made with {manifest['model']}")
            texts = [chunk["texte"] for chunk in load_chunks(year, self.data_dir)]
            vectors = np.load(self.data_dir / f"embeddings_structured_{year}.npy")
            if len(texts) != len(vectors):
                raise ValueError(f"FY{year}: {len(texts)} chunks but {len(vectors)} embeddings")
            self.cache[year] = (texts, vectors)
        return self.cache[year]

    def __call__(self, question: str, year: int, depth: int) -> list[str]:
        from finsight.retrieval import encode_query

        texts, vectors = self._year(year)
        scores = vectors @ encode_query(question)
        return [texts[i] for i in np.argsort(-scores)[:depth]]


def print_report(name: str, results: dict) -> None:
    columns = [key for key in results["overall"] if key != "n"]
    print(f"\nRetriever: {name}")
    print(f"{'group':<32}{'n':>4}" + "".join(f"{c:>8}" for c in columns))
    rows = [("overall", results["overall"]), *results["groups"].items()]
    for group, summary in rows:
        print(f"{group:<32}{summary['n']:>4}" + "".join(f"{summary[c]:>8.2f}" for c in columns))

    missed = [row["id"] for row in results["questions"] if row["rank"] is None]
    print(f"\nNot found in top {max(KS)}: {', '.join(missed) or 'none'}")


def run_retrieval(name: str) -> None:
    results = evaluate(DenseRetriever(), load_questions())
    print_report(name, results)

    RESULTS_DIR.mkdir(exist_ok=True)
    output = RESULTS_DIR / f"retrieval_{name}.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved to {output.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FinSight evaluation benchmarks.")
    commands = parser.add_subparsers(dest="command", required=True)

    retrieval = commands.add_parser("retrieval", help="Hit@k and MRR on eval/questions.jsonl")
    retrieval.add_argument("--name", default="dense-minilm", help="Label of the saved results.")

    verifier = commands.add_parser("verifier", help="Claim verifier on eval/claims.jsonl (LLM)")
    verifier.add_argument("--name", help="Label of the saved results (default: model name).")
    verifier.add_argument("--limit", type=int, help="Only check the first N claims.")

    args = parser.parse_args()
    if args.command == "retrieval":
        run_retrieval(args.name)
    else:
        from finsight.verifier_eval import run

        run(args.name, args.limit)


if __name__ == "__main__":
    main()
