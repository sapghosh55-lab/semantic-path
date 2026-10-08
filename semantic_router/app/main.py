from contextlib import asynccontextmanager
from typing import List, Dict, Any
from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

from app.config import Settings, get_settings
from app.router.models import ChatCompletionRequest, RouteDecision, MetricSummary
from app.router.engine import SemanticEngine
from app.upstream.client import UpstreamClient
from app.telemetry.tracker import TelemetryTracker

# Shared runtime singletons
engine: SemanticEngine = None
upstream_client: UpstreamClient = None
telemetry_tracker: TelemetryTracker = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan manager: Pre-loads sentence transformer & computes vector index during startup."""
    global engine, upstream_client, telemetry_tracker

    settings = get_settings()
    telemetry_tracker = TelemetryTracker(settings=settings)
    upstream_client = UpstreamClient(tracker=telemetry_tracker)

    print(f"[SemanticRouter] Initializing SentenceTransformer model '{settings.EMBEDDING_MODEL_NAME}'...")
    model_name = settings.EMBEDDING_MODEL_NAME
    if not model_name.startswith("sentence-transformers/") and "/" not in model_name:
        model_name = f"sentence-transformers/{model_name}"

    st_model = SentenceTransformer(model_name)

    engine = SemanticEngine(settings=settings, model=st_model)
    print(f"[SemanticRouter] Loading route definitions from '{settings.ROUTES_FILE}'...")
    engine.load_routes()
    print(f"[SemanticRouter] Route index built with {len(engine.anchor_metadata)} anchor utterances. Ready!")

    yield

    # Server shutdown
    print("[SemanticRouter] Closing upstream client HTTP connections...")
    await upstream_client.close()


app = FastAPI(
    title="SemanticRouter API",
    description="High-performance, cost-optimizing LLM proxy gateway with local sentence embedding routing",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_engine() -> SemanticEngine:
    if engine is None:
        raise HTTPException(status_code=503, detail="Router engine not initialized")
    return engine


def get_client() -> UpstreamClient:
    if upstream_client is None:
        raise HTTPException(status_code=503, detail="Upstream client not initialized")
    return upstream_client


def get_tracker() -> TelemetryTracker:
    if telemetry_tracker is None:
        raise HTTPException(status_code=503, detail="Telemetry tracker not initialized")
    return telemetry_tracker


@app.get("/health")
async def health_check(
    router_engine: SemanticEngine = Depends(get_engine),
    settings: Settings = Depends(get_settings),
):
    """Health check endpoint for routing gateway and model index status."""
    return {
        "status": "healthy",
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "indexed_anchors": len(router_engine.anchor_metadata) if router_engine else 0,
        "fast_lane_model": settings.FAST_MODEL_NAME,
        "deep_lane_model": settings.DEEP_MODEL_NAME,
    }


@app.get("/metrics", response_model=MetricSummary)
@app.get("/telemetry/metrics", response_model=MetricSummary)
async def get_metrics(tracker: TelemetryTracker = Depends(get_tracker)):
    """Telemetry metrics endpoint: latency, fast vs deep lane ratio, cumulative USD savings."""
    return tracker.get_summary()


@app.get("/dashboard/stats")
async def get_dashboard_stats(tracker: TelemetryTracker = Depends(get_tracker)):
    """Dashboard analytics endpoint returning real-time metrics overview."""
    summary = tracker.get_summary()
    return {
        "total_requests": summary.total_requests,
        "fast_lane_count": summary.fast_lane_count,
        "deep_lane_count": summary.deep_lane_count,
        "fast_lane_percentage": summary.fast_lane_percentage,
        "deep_lane_percentage": summary.deep_lane_percentage,
        "avg_classification_latency_ms": summary.avg_classification_latency_ms,
        "estimated_prompt_tokens_processed": summary.estimated_prompt_tokens_processed,
        "estimated_completion_tokens_processed": summary.estimated_completion_tokens_processed,
        "estimated_cumulative_dollar_savings": summary.estimated_cumulative_dollar_savings,
    }


@app.get("/v1/models")
async def list_models(settings: Settings = Depends(get_settings)):
    """OpenAI-compatible models listing endpoint."""
    return {
        "object": "list",
        "data": [
            {
                "id": settings.FAST_MODEL_NAME,
                "object": "model",
                "created": 1700000000,
                "owned_by": "semantic-router-fast-lane",
            },
            {
                "id": settings.FRONTIER_MODEL_NAME,
                "object": "model",
                "created": 1700000000,
                "owned_by": "semantic-router-deep-lane",
            },
            {
                "id": "auto",
                "object": "model",
                "created": 1700000000,
                "owned_by": "semantic-router-auto",
            },
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    router_engine: SemanticEngine = Depends(get_engine),
    client: UpstreamClient = Depends(get_client),
    tracker: TelemetryTracker = Depends(get_tracker),
):
    """
    OpenAI-compatible chat completion proxy endpoint.
    Intercepts user query, classifies complexity via local sentence similarity in <15ms,
    and forwards request asynchronously to optimal fast_lane or deep_lane LLM target.
    Injects x-semantic-route, x-semantic-model, x-semantic-classification-ms, x-estimated-cost-saved headers.
    """
    decision = None
    try:
        prompt = router_engine.extract_prompt(request.messages)
        decision = router_engine.classify(prompt)

        # Record routing decision telemetry
        tracker.record_routing(decision.target_route, decision.classification_time_ms)

        # Forward to target upstream LLM provider
        return await client.forward(request, decision)
    except Exception as e:
        print(f"[SemanticRouter ERROR]: {repr(e)}")
        error_msg = f"[Upstream Error ({type(e).__name__})]: {str(e)}"
        resp_headers = {
            "x-semantic-route": decision.target_route if decision else "deep_lane",
            "x-semantic-model": decision.selected_model if decision else "error",
            "x-semantic-classification-ms": str(decision.classification_time_ms) if decision else "0.0",
            "x-estimated-cost-saved": "0.000000",
        }
        return JSONResponse(
            status_code=200,
            headers=resp_headers,
            content={
                "id": "chatcmpl-error-fallback",
                "object": "chat.completion",
                "created": 1700000000,
                "model": decision.selected_model if decision else "error",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": error_msg
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
        )
