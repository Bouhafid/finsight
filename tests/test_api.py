import pytest
from fastapi.testclient import TestClient

from finsight import api
from finsight.models import Answer
from finsight.retrieval import RetrievalError


@pytest.fixture
def client(monkeypatch):
    calls = []

    def fake_answer(question, years, top_k):
        calls.append(question)
        if question == "broken":
            raise RetrievalError("no chunks")
        return Answer(
            question=question, years=years, numeric_claims=[], document_claims=[], sources={}
        )

    monkeypatch.setattr(api, "answer_question", fake_answer)
    monkeypatch.setattr(api, "limiter", api.RateLimiter(limit=2))
    monkeypatch.setattr(api, "cache", {})
    client = TestClient(api.app)
    client.calls = calls
    return client


def test_ask_returns_the_answer(client):
    response = client.post("/api/ask", json={"question": "Supply risks?", "years": [2025]})

    assert response.status_code == 200
    assert response.json()["years"] == [2025]


def test_repeated_questions_are_served_from_cache(client):
    for question in ("Supply risks?", "  supply   RISKS? "):
        assert client.post("/api/ask", json={"question": question}).status_code == 200

    assert len(client.calls) == 1


def test_rate_limit(client):
    statuses = [client.post("/api/ask", json={"question": f"Q{i}??"}).status_code for i in range(3)]
    assert statuses == [200, 200, 429]


@pytest.mark.parametrize(
    "body",
    [{"question": "?"}, {"question": "Risks?", "top_k": 9}, {"question": "Risks?", "years": []}],
)
def test_invalid_requests(client, body):
    assert client.post("/api/ask", json=body).status_code == 422


def test_missing_data_is_a_503(client):
    assert client.post("/api/ask", json={"question": "broken"}).status_code == 503


def test_demo_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "FinSight" in response.text
