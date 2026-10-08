from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible /v1/chat/completions request schema."""
    model_config = ConfigDict(extra="allow")

    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None


class RouteDecision(BaseModel):
    """Routing decision metadata computed by SemanticEngine."""
    target_route: str          # "fast_lane" or "deep_lane"
    selected_model: str
    similarity_score: float
    classification_time_ms: float
    base_url: str = ""
    api_key: str = ""
    matched_utterance: Optional[str] = None

    @property
    def target_tier(self) -> str:
        return "fast" if self.target_route == "fast_lane" else "deep"

    @property
    def route_name(self) -> str:
        return self.target_route

    @property
    def classification_latency_ms(self) -> float:
        return self.classification_time_ms


class RouteDefinition(BaseModel):
    """Configuration format for anchor routes in routes.yaml."""
    name: str
    description: str
    target_tier: str
    utterances: List[str]


class MetricSummary(BaseModel):
    """Gateway telemetry metrics summary for /metrics and /dashboard/stats."""
    total_requests: int
    fast_lane_count: int
    deep_lane_count: int
    fast_lane_percentage: float
    deep_lane_percentage: float
    avg_classification_latency_ms: float
    estimated_prompt_tokens_processed: int
    estimated_completion_tokens_processed: int
    estimated_cumulative_dollar_savings: float

    # Aliases for backward compatibility
    @property
    def fast_tier_count(self) -> int:
        return self.fast_lane_count

    @property
    def frontier_tier_count(self) -> int:
        return self.deep_lane_count

    @property
    def fast_tier_percentage(self) -> float:
        return self.fast_lane_percentage

    @property
    def estimated_cost_saved_usd(self) -> float:
        return self.estimated_cumulative_dollar_savings
