"""Integration tests against a running RAG server — skipped when the server is down."""
import os

import httpx
import pytest
from conftest import load_script

test_rag = load_script("02_rag/test_rag.py")

SERVER = os.environ.get("RAG_TEST_SERVER", "http://localhost:8080")


def get_client() -> httpx.Client:
    client = httpx.Client(base_url=SERVER, timeout=10)
    try:
        resp = client.get("/health")
        resp.raise_for_status()
    except httpx.HTTPError as e:
        pytest.skip(f"RAG server not reachable at {SERVER}: {e}")
    health = resp.json()
    if health.get("status") != "ok" or "collection_count" not in health:
        pytest.skip(f"server at {SERVER} is reachable but is not this project's RAG service")
    return client


@pytest.mark.integration
def test_health_endpoint():
    client = get_client()
    health = client.get("/health").json()
    assert isinstance(health["collection_count"], int)
    client.close()


@pytest.mark.integration
def test_search_returns_ranked_results():
    get_client().close()  # skip if server down / not our service
    results = test_rag.search_only(SERVER, "虎蹲炮的射程是多少？", top_k=3)
    assert results
    for r in results:
        assert r["text"]
        assert "rerank_score" in r
    scores = [r["rerank_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.integration
def test_chat_completion_has_sources():
    get_client().close()  # skip if server down / not our service
    result = test_rag.ask(SERVER, "明代神机营的编制与装备是什么？")
    assert result["choices"][0]["message"]["content"]
    assert result.get("retrieved_contexts")
