import pytest
from unittest.mock import AsyncMock
from httpx import AsyncClient, Response as HttpxResponse, Request as HttpxRequest, ASGITransport
from sentence_transformers import SentenceTransformer

from semantic_router.app.config import Settings
from semantic_router.app.router.engine import SemanticEngine
from semantic_router.app.router.models import ChatCompletionRequest, ChatMessage
from semantic_router.app.telemetry.tracker import TelemetryTracker
from semantic_router.app.upstream.client import UpstreamClient
from semantic_router.app.main import app, get_engine, get_client, get_tracker, get_settings


@pytest.fixture(scope="module")
def settings():
    return Settings(
        SIMILARITY_THRESHOLD=0.45,
        FAST_MODEL_NAME="gemma2:2b",
        FRONTIER_MODEL_NAME="llama-3.3-70b-versatile",
        ROUTES_FILE="routes.yaml"
    )


@pytest.fixture(scope="module")
def router_engine(settings):
    model_name = settings.EMBEDDING_MODEL_NAME
    if not model_name.startswith("sentence-transformers/") and "/" not in model_name:
        model_name = f"sentence-transformers/{model_name}"

    st_model = SentenceTransformer(model_name)
    engine = SemanticEngine(settings=settings, model=st_model)
    engine.load_routes(settings.ROUTES_FILE)
    engine.classify("warmup init prompt 1")
    engine.classify("warmup init prompt 2")
    return engine


@pytest.fixture(scope="module")
def tracker(settings):
    return TelemetryTracker(settings=settings)


def test_temperature_conversion_routes_to_fast_lane(router_engine):
    """Test that 'what is 25 degree in fahrenheit' strictly routes to fast_lane."""
    prompt = "what is 25 degree in fahrenheit"
    decision = router_engine.classify(prompt)

    assert decision.target_route == "fast_lane"
    assert decision.selected_model == "gemma2:2b"
    assert decision.classification_time_ms < 35.0


def test_unit_math_factual_queries_route_to_fast_lane(router_engine):
    """Test everyday math, unit conversions, and factual queries route to fast_lane."""
    queries = [
        "convert 100 celsius to fahrenheit",
        "how many kilometers in 5 miles",
        "calculate 15 percent of 80",
        "what is the capital of Japan?",
        "What is the capital of France?"
    ]
    for q in queries:
        decision = router_engine.classify(q)
        assert decision.target_route == "fast_lane", f"Query '{q}' should route to fast_lane but got '{decision.target_route}'"


def test_byzantine_consensus_routes_to_deep_lane(router_engine):
    """Test that complex system design queries route to deep_lane."""
    prompt = "Design a distributed consensus algorithm with Byzantine fault tolerance in C++"
    decision = router_engine.classify(prompt)

    assert decision.target_route == "deep_lane"
    assert decision.selected_model == "llama-3.3-70b-versatile"
    assert decision.classification_time_ms < 35.0


def test_high_complexity_queries_route_to_deep_lane(router_engine):
    """Test high-complexity tasks (C++ memory allocators, distributed systems, quantum algorithms, smart contracts) resolve to deep_lane."""
    queries = [
        "Design a custom C++ memory allocator with arena allocation",
        "Architect a distributed consensus protocol similar to Raft with leader election",
        "Explain Quantum Field Theory and derive the Dirac equation step by step",
        "Audit this Ethereum smart contract for reentrancy vulnerabilities and flash loan exploits"
    ]
    for q in queries:
        decision = router_engine.classify(q)
        assert decision.target_route == "deep_lane", f"Query '{q}' should route to deep_lane but got '{decision.target_route}'"


def test_telemetry_tracker(settings):
    tr = TelemetryTracker(settings=settings)
    tr.record_routing("fast_lane", 4.2)
    tr.record_routing("deep_lane", 7.8)
    tr.record_usage("fast_lane", 1000, 500)

    summary = tr.get_summary()
    assert summary.total_requests == 2
    assert summary.fast_lane_count == 1
    assert summary.deep_lane_count == 1
    assert summary.fast_lane_percentage == 50.0
    assert summary.deep_lane_percentage == 50.0


@pytest.mark.asyncio
async def test_health_endpoint(router_engine, tracker, settings):
    app.dependency_overrides[get_engine] = lambda: router_engine
    app.dependency_overrides[get_tracker] = lambda: tracker
    app.dependency_overrides[get_settings] = lambda: settings

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["indexed_anchors"] > 0

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_chat_completions_fast_lane_proxy(router_engine, tracker, settings):
    mock_request = HttpxRequest("POST", "http://localhost:11434/v1/chat/completions")
    mock_response = HttpxResponse(
        status_code=200,
        json={
            "id": "chatcmpl-mock-123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gemma2:2b",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "77°F"},
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
        },
        request=mock_request
    )

    upstream_client_instance = UpstreamClient(tracker=tracker)
    upstream_client_instance.client.post = AsyncMock(return_value=mock_response)

    app.dependency_overrides[get_engine] = lambda: router_engine
    app.dependency_overrides[get_tracker] = lambda: tracker
    app.dependency_overrides[get_client] = lambda: upstream_client_instance
    app.dependency_overrides[get_settings] = lambda: settings

    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": "what is 25 degree in fahrenheit"}]
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.post("/v1/chat/completions", json=payload)
        assert res.status_code == 200
        assert res.headers.get("x-semantic-route") == "fast_lane"
        assert res.headers.get("x-semantic-model") == "gemma2:2b"
        data = res.json()
        assert data["choices"][0]["message"]["content"] == "77°F"

    app.dependency_overrides.clear()
