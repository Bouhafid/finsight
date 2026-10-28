import json
from collections import Counter
from types import SimpleNamespace

import pytest

from finsight.config import AVAILABLE_YEARS, ROOT
from finsight.evaluation import (
    BM25Retriever,
    EvalQuestion,
    HybridRetriever,
    RerankRetriever,
    evaluate,
    first_relevant_rank,
    is_relevant,
    load_chunks,
    load_questions,
    summarize,
)

QUESTIONS = load_questions()
HAS_CHUNKS = (ROOT / "data" / "processed" / "apple").exists()


def test_question_ids_are_unique():
    duplicates = [qid for qid, n in Counter(q.id for q in QUESTIONS).items() if n > 1]
    assert not duplicates


def test_questions_use_available_years():
    assert {q.year for q in QUESTIONS} <= set(AVAILABLE_YEARS)


def test_set_covers_french_and_unanswerable_questions():
    assert any(q.lang == "fr" for q in QUESTIONS)
    assert any(not q.answerable for q in QUESTIONS)


def test_matching_ignores_case_quotes_and_whitespace():
    question = EvalQuestion(
        id="t",
        year=2025,
        lang="en",
        category="t",
        question="?",
        evidence=["the Company's  Head of\nSecurity"],
    )
    assert is_relevant("THE COMPANY’S HEAD OF SECURITY leads", question)
    assert not is_relevant("an unrelated passage", question)


def _question(qid, evidence, lang="en"):
    return EvalQuestion(id=qid, year=2025, lang=lang, category="c", question=qid, evidence=evidence)


def test_first_relevant_rank():
    question = _question("q", ["supply shortages"])
    assert first_relevant_rank(["a", "risk of SUPPLY shortages", "supply shortages"], question) == 2
    assert first_relevant_rank(["a", "b"], question) is None


def test_summarize_hit_rate_and_mrr():
    summary = summarize([1, 3, None, 2], ks=(1, 3))
    assert summary["n"] == 4
    assert summary["hit@1"] == 0.25
    assert summary["hit@3"] == 0.75
    assert summary["mrr"] == pytest.approx((1 + 1 / 3 + 1 / 2) / 4)


def test_evaluate_skips_unanswerable_and_groups_results():
    questions = [
        _question("hit", ["tariffs"]),
        _question("miss", ["rare earths"], lang="fr"),
        _question("none", []),
    ]

    def retriever(question, year, depth):
        return ["new tariffs were announced", "other text"][:depth]

    results = evaluate(retriever, questions, ks=(1, 2))
    assert results["overall"]["n"] == 2
    assert results["overall"]["hit@1"] == 0.5
    assert results["groups"]["lang=fr"]["hit@2"] == 0
    assert [row["id"] for row in results["questions"]] == ["hit", "miss"]


def test_evaluate_reports_median_latency():
    results = evaluate(lambda question, year, depth: ["tariffs"], [_question("q", ["tariffs"])])
    assert results["latency_ms"] >= 0


def test_bm25_ranks_exact_terms_first(tmp_path):
    texts = [
        "The Company sells iPhone and Mac products.",
        "Cyberattacks and ransomware target the Company's systems.",
        "Net sales vary with the holiday season.",
    ]
    lines = [json.dumps({"texte": text}) for text in texts]
    (tmp_path / "chunks_structured_2025.jsonl").write_text("\n".join(lines), encoding="utf-8")

    retriever = BM25Retriever(data_dir=tmp_path)
    assert retriever("Why is Apple a target of cyberattacks?", 2025, 1) == [texts[1]]
    assert len(retriever("unknown words only", 2025, 3)) == 3


def test_hybrid_fusion_favours_passages_found_by_both():
    def dense(question, year, depth):
        return ["a", "shared", "b"][:depth]

    def keywords(question, year, depth):
        return ["c", "shared", "d"][:depth]

    assert HybridRetriever(dense, keywords)("q", 2025, 1) == ["shared"]


def test_rerank_reorders_the_first_stage_candidates():
    def first_stage(question, year, depth):
        return ["weak match", "best match", "no match"][:depth]

    reranker = RerankRetriever(first_stage, candidates=3)
    # Fake cross-encoder: the score is the position of "best" in each passage.
    reranker._loaded = SimpleNamespace(predict=lambda pairs: [p[1].find("best") for p in pairs])
    assert reranker("q", 2025, 2) == ["best match", "weak match"]


@pytest.mark.skipif(not HAS_CHUNKS, reason="data/processed is not versioned")
@pytest.mark.parametrize("question", [q for q in QUESTIONS if q.answerable], ids=lambda q: q.id)
def test_evidence_is_found_in_the_filing(question):
    """Every evidence sentence must appear in at least one chunk of its year."""
    chunks = load_chunks(question.year)
    for snippet in question.evidence:
        single = question.model_copy(update={"evidence": [snippet]})
        assert any(is_relevant(c["texte"], single) for c in chunks), snippet
