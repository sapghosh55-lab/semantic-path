import time
import threading
from collections import deque
from typing import Deque
from app.router.models import MetricSummary
from app.config import Settings


class TelemetryTracker:
    """Thread-safe telemetry & metric recorder for SemanticRouter gateway."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._lock = threading.Lock()
        self.total_requests: int = 0
        self.fast_lane_count: int = 0
        self.deep_lane_count: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.cumulative_savings_usd: float = 0.0
        self.total_classification_latency_ms: float = 0.0
        self._recent_latencies: Deque[float] = deque(maxlen=500)

    def record_routing(self, tier_or_route: str, latency_ms: float) -> None:
        with self._lock:
            self.total_requests += 1
            if tier_or_route in ("fast", "fast_lane"):
                self.fast_lane_count += 1
            else:
                self.deep_lane_count += 1
            self.total_classification_latency_ms += latency_ms
            self._recent_latencies.append(latency_ms)

    def calculate_cost_saved(self, route_name: str, prompt_tokens: int = 100, completion_tokens: int = 50) -> float:
        """Calculate estimated USD savings for routing a request to fast_lane vs deep_lane."""
        if route_name not in ("fast", "fast_lane"):
            return 0.0
        frontier_cost = (
            (prompt_tokens / 1000.0) * self._settings.FRONTIER_MODEL_COST_PER_1K_INPUT
            + (completion_tokens / 1000.0) * self._settings.FRONTIER_MODEL_COST_PER_1K_OUTPUT
        )
        fast_cost = (
            (prompt_tokens / 1000.0) * self._settings.FAST_MODEL_COST_PER_1K_INPUT
            + (completion_tokens / 1000.0) * self._settings.FAST_MODEL_COST_PER_1K_OUTPUT
        )
        return max(0.0, frontier_cost - fast_cost)

    def record_usage(self, route_name: str, prompt_tokens: int, completion_tokens: int) -> float:
        with self._lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            saved = self.calculate_cost_saved(route_name, prompt_tokens, completion_tokens)
            self.cumulative_savings_usd += saved
            return saved

    def get_summary(self) -> MetricSummary:
        with self._lock:
            total = self.total_requests
            fast_pct = (self.fast_lane_count / total * 100.0) if total > 0 else 0.0
            deep_pct = (self.deep_lane_count / total * 100.0) if total > 0 else 0.0
            avg_lat = (
                (sum(self._recent_latencies) / len(self._recent_latencies))
                if self._recent_latencies
                else 0.0
            )

            # If exact token usage hasn't updated cumulative_savings_usd yet, estimate based on counts
            savings = self.cumulative_savings_usd
            if savings == 0.0 and self.fast_lane_count > 0:
                fast_ratio = (self.fast_lane_count / total) if total > 0 else 0.0
                fast_prompt_tokens = self.total_prompt_tokens * fast_ratio
                fast_completion_tokens = self.total_completion_tokens * fast_ratio
                frontier_cost = (
                    (fast_prompt_tokens / 1000.0) * self._settings.FRONTIER_MODEL_COST_PER_1K_INPUT
                    + (fast_completion_tokens / 1000.0) * self._settings.FRONTIER_MODEL_COST_PER_1K_OUTPUT
                )
                fast_cost = (
                    (fast_prompt_tokens / 1000.0) * self._settings.FAST_MODEL_COST_PER_1K_INPUT
                    + (fast_completion_tokens / 1000.0) * self._settings.FAST_MODEL_COST_PER_1K_OUTPUT
                )
                savings = max(0.0, frontier_cost - fast_cost)

            return MetricSummary(
                total_requests=total,
                fast_lane_count=self.fast_lane_count,
                deep_lane_count=self.deep_lane_count,
                fast_lane_percentage=round(fast_pct, 2),
                deep_lane_percentage=round(deep_pct, 2),
                avg_classification_latency_ms=round(avg_lat, 3),
                estimated_prompt_tokens_processed=self.total_prompt_tokens,
                estimated_completion_tokens_processed=self.total_completion_tokens,
                estimated_cumulative_dollar_savings=round(savings, 6),
            )
