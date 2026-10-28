"""Evaluation set and retrieval metrics: finsight-eval

Relevance is defined by evidence text, not by chunk IDs, so the same set
still applies when the chunking or the embedding model changes.
"""

import argparse
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
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
        start = time.perf_counter()
        texts = retriever(q.question, q.year, depth)
        ms = (time.perf_counter() - start) * 1000
        rank = first_relevant_rank(texts, q)
        per_question.append({"id": q.id, "rank": rank, "ms": round(ms, 1)})
        for group in ("lang", "category", "year"):
            groups[f"{group}={getattr(q, group)}"].append(rank)

    ranks = [row["rank"] for row in per_question]
    return {
        "overall": summarize(ranks, ks),
        # Median search time per question, models and indexes already loaded.
        "latency_ms": statistics.median(row["ms"] for row in per_question)
        if per_question
        else None,
        "groups": {name: summarize(r, ks) for name, r in sorted(groups.items())},
        "questions": per_question,
    }


E5_SMALL = "intfloat/multilingual-e5-small"
# Multilingual cross-encoders: a large accurate one and a small one for CPU serving.
RERANKER = "BAAI/bge-reranker-v2-m3"
RERANKER_MINI = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# E5 models were trained with these prefixes and score worse without them.
PREFIXES = {E5_SMALL: ("query: ", "passage: ")}


def chunk_texts(year: int, data_dir: Path = CHUNKS_DIR) -> list[str]:
    return [chunk["texte"] for chunk in load_chunks(year, data_dir)]


def top(texts: list[str], scores: np.ndarray, depth: int) -> list[str]:
    # Stable sort: ties keep chunk order, so results are reproducible.
    return [texts[i] for i in np.argsort(-scores, kind="stable")[:depth]]


class DenseRetriever:
    """Exact cosine search over chunk embeddings.

    With the production model, the vectors are the ones stored in data/processed
    and inserted in Postgres, so this mirrors the production search (an
    approximate HNSW index) without the database. Other models embed the chunks
    on the fly.
    """

    def __init__(self, model: str = EMBEDDING_MODEL, data_dir: Path = CHUNKS_DIR):
        self.model = model
        self.data_dir = data_dir
        self.query_prefix, self.passage_prefix = PREFIXES.get(model, ("", ""))
        self.cache: dict[int, tuple[list[str], np.ndarray]] = {}

    def _encoder(self):
        from finsight.retrieval import _model

        if self.model == EMBEDDING_MODEL:
            return _model()

        from sentence_transformers import SentenceTransformer

        if not hasattr(self, "_other"):
            self._other = SentenceTransformer(self.model, device="cpu")
        return self._other

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        return self._encoder().encode(
            [prefix + text for text in texts], normalize_embeddings=True, batch_size=16
        )

    def _year(self, year: int) -> tuple[list[str], np.ndarray]:
        if year not in self.cache:
            texts = chunk_texts(year, self.data_dir)
            if self.model == EMBEDDING_MODEL:
                vectors = self._stored_vectors(year, len(texts))
            else:
                vectors = self._encode(texts, self.passage_prefix)
            self.cache[year] = (texts, vectors)
        return self.cache[year]

    def _stored_vectors(self, year: int, count: int) -> np.ndarray:
        manifest = json.loads(
            (self.data_dir / f"embeddings_manifest_{year}.json").read_text(encoding="utf-8")
        )
        if manifest["model"] != EMBEDDING_MODEL:
            raise ValueError(f"FY{year} embeddings were made with {manifest['model']}")
        vectors = np.load(self.data_dir / f"embeddings_structured_{year}.npy")
        if len(vectors) != count:
            raise ValueError(f"FY{year}: {count} chunks but {len(vectors)} embeddings")
        return vectors

    def __call__(self, question: str, year: int, depth: int) -> list[str]:
        texts, vectors = self._year(year)
        query = self._encode([question], self.query_prefix)[0]
        return top(texts, vectors @ query, depth)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """Okapi BM25 keyword search: catches exact terms (names, dates) that embeddings blur."""

    def __init__(self, data_dir: Path = CHUNKS_DIR, k1: float = 1.5, b: float = 0.75):
        self.data_dir = data_dir
        self.k1, self.b = k1, b
        self.index: dict[int, tuple] = {}

    def _year(self, year: int) -> tuple:
        if year not in self.index:
            texts = chunk_texts(year, self.data_dir)
            counts = [Counter(tokenize(text)) for text in texts]
            lengths = np.array([sum(c.values()) for c in counts], dtype=float)
            n = len(texts)
            df = Counter(term for c in counts for term in c)
            idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}
            self.index[year] = (texts, counts, lengths / lengths.mean(), idf)
        return self.index[year]

    def __call__(self, question: str, year: int, depth: int) -> list[str]:
        texts, counts, relative_length, idf = self._year(year)
        terms = [term for term in set(tokenize(question)) if term in idf]
        scores = np.zeros(len(texts))
        for i, count in enumerate(counts):
            norm = self.k1 * (1 - self.b + self.b * relative_length[i])
            for term in terms:
                tf = count.get(term, 0)
                if tf:
                    scores[i] += idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return top(texts, scores, depth)


class HybridRetriever:
    """Reciprocal rank fusion: merges rankings without calibrating their scores."""

    def __init__(self, *retrievers: Retriever, pool: int = 50, k: int = 60):
        self.retrievers = retrievers
        self.pool = pool
        self.k = k

    def __call__(self, question: str, year: int, depth: int) -> list[str]:
        scores: dict[str, float] = defaultdict(float)
        for retriever in self.retrievers:
            for rank, text in enumerate(retriever(question, year, self.pool), start=1):
                scores[text] += 1 / (self.k + rank)
        return sorted(scores, key=scores.__getitem__, reverse=True)[:depth]


class RerankRetriever:
    """Re-scores the first stage's candidates with a cross-encoder.

    A cross-encoder reads the question and the passage together: slower than
    comparing two vectors, but much finer, so it only sees a few candidates.
    """

    def __init__(self, first_stage: Retriever, model: str = RERANKER, candidates: int = 20):
        self.first_stage = first_stage
        self.model = model
        self.candidates = candidates

    def _cross_encoder(self):
        from sentence_transformers import CrossEncoder

        if not hasattr(self, "_loaded"):
            self._loaded = CrossEncoder(self.model, device="cpu")
        return self._loaded

    def __call__(self, question: str, year: int, depth: int) -> list[str]:
        texts = self.first_stage(question, year, max(depth, self.candidates))
        scores = self._cross_encoder().predict([(question, text) for text in texts])
        return top(texts, np.asarray(scores), depth)


# Every saved result comes from one of these named, reproducible configurations.
VARIANTS: dict[str, Callable[[], Retriever]] = {
    "dense-minilm": lambda: DenseRetriever(),
    "bm25": lambda: BM25Retriever(),
    "hybrid-minilm": lambda: HybridRetriever(DenseRetriever(), BM25Retriever()),
    "dense-e5-small": lambda: DenseRetriever(E5_SMALL),
    "hybrid-e5-small": lambda: HybridRetriever(DenseRetriever(E5_SMALL), BM25Retriever()),
    "dense-e5-small+rerank": lambda: RerankRetriever(DenseRetriever(E5_SMALL)),
    "hybrid-e5-small+rerank": lambda: RerankRetriever(
        HybridRetriever(DenseRetriever(E5_SMALL), BM25Retriever())
    ),
    "dense-e5-small+rerank-mini": lambda: RerankRetriever(DenseRetriever(E5_SMALL), RERANKER_MINI),
}


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
    retriever = VARIANTS[name]()
    questions = load_questions()
    # Load models and build each year's index before timing the questions.
    for year in sorted({q.year for q in questions}):
        retriever("warm-up", year, 1)

    results = evaluate(retriever, questions)
    print_report(name, results)
    print(f"Median latency: {results['latency_ms']:.0f} ms per question")

    RESULTS_DIR.mkdir(exist_ok=True)
    output = RESULTS_DIR / f"retrieval_{name}.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved to {output.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FinSight evaluation benchmarks.")
    commands = parser.add_subparsers(dest="command", required=True)

    retrieval = commands.add_parser("retrieval", help="Hit@k and MRR on eval/questions.jsonl")
    retrieval.add_argument(
        "--name", default="dense-minilm", choices=VARIANTS, help="Retriever variant to score."
    )

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
