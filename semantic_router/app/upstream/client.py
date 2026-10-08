import json
from typing import AsyncGenerator, Dict, Any, Tuple
import httpx
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import HTTPException
from app.router.models import ChatCompletionRequest, RouteDecision
from app.telemetry.tracker import TelemetryTracker
from app.config import get_settings


def is_valid_api_key(key: str) -> bool:
    """Check if an API key is present and not a dummy placeholder string."""
    if not key:
        return False
    k = key.strip().lower()
    if k in ("", "none", "null", "ollama", "your_openai_api_key_here", "your_gemini_api_key_here", "your_groq_api_key_here", "your_groq_or_openai_api_key_here"):
        return False
    if k.startswith("your_"):
        return False
    return True


class UpstreamClient:
    """Async client for forwarding chat completion requests to upstream LLM providers."""

    def __init__(self, tracker: TelemetryTracker):
        self.tracker = tracker
        self.settings = get_settings()
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))

    async def close(self) -> None:
        await self.client.aclose()

    def resolve_target(self, decision: RouteDecision) -> Tuple[str, str, str, str, str]:
        """
        Resolves (base_url, selected_model, api_key, target_route, fallback_reason).
        """
        if decision.target_route == "fast_lane":
            return (
                self.settings.FAST_MODEL_BASE_URL.strip(),
                self.settings.FAST_MODEL_NAME.strip(),
                self.settings.FAST_MODEL_API_KEY.strip(),
                "fast_lane",
                ""
            )

        # DEEP_LANE configuration
        deep_base_url = getattr(self.settings, "DEEP_MODEL_BASE_URL", "https://api.groq.com/openai/v1").strip()
        deep_model_name = getattr(self.settings, "DEEP_MODEL_NAME", "llama-3.3-70b-versatile").strip()
        deep_api_key = getattr(self.settings, "DEEP_MODEL_API_KEY", "").strip()

        return (
            deep_base_url,
            deep_model_name,
            deep_api_key,
            "deep_lane",
            ""
        )

    async def forward(
        self,
        request: ChatCompletionRequest,
        decision: RouteDecision
    ):
        base_url, model_name, api_key, route_name, fallback_reason = self.resolve_target(decision)

        if decision.target_route == "deep_lane":
            target_url = "https://api.groq.com/openai/v1/chat/completions"
            target_model = self.settings.DEEP_MODEL_NAME.strip()
            api_key = self.settings.DEEP_MODEL_API_KEY.strip()
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        else:
            target_url = f"{base_url.rstrip('/')}/chat/completions"
            target_model = model_name
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

        print(f"[SemanticRouter] Routing upstream to {route_name.upper()} ({target_model}) at {target_url}")

        payload = request.model_dump(exclude_none=True)

        if not request.stream:
            return await self._forward_sync(
                target_url, headers, payload, decision, target_model, route_name
            )
        else:
            return await self._forward_stream(
                target_url, headers, payload, decision, target_model, route_name
            )

    async def _forward_sync(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        decision: RouteDecision,
        model_name: str,
        route_name: str
    ) -> JSONResponse:
        resp_headers = {
            "x-semantic-route": route_name,
            "x-semantic-model": model_name,
            "x-semantic-classification-ms": str(decision.classification_time_ms),
            "x-estimated-cost-saved": "0.000000",
            "X-Semantic-Router-Tier": "fast" if route_name == "fast_lane" else "deep",
            "X-Semantic-Router-Model": model_name,
            "X-Semantic-Router-Score": str(decision.similarity_score),
            "X-Semantic-Router-Latency-MS": str(decision.classification_time_ms)
        }

        # Filter payload: Groq requires strict OpenAI payload formatting
        clean_payload = {
            "model": model_name.strip(),
            "messages": payload.get("messages", []),
            "temperature": payload.get("temperature", 0.7)
        }
        if "max_tokens" in payload and payload["max_tokens"] is not None:
            clean_payload["max_tokens"] = payload["max_tokens"]

        try:
            print(f"[DEBUG GROQ] URL: {url} | Model: {clean_payload['model']}")
            response = await self.client.post(url, json=clean_payload, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 100)
                completion_tokens = usage.get("completion_tokens", 50)
                cost_saved = self.tracker.record_usage(route_name, prompt_tokens, completion_tokens)
                resp_headers["x-estimated-cost-saved"] = f"{cost_saved:.6f}"
                return JSONResponse(content=data, status_code=200, headers=resp_headers)

            print(f"[DEBUG GROQ ERROR]: {response.status_code} - {response.text}")

            # If 404 model_not_found or non-200 error, fallback to candidate Groq models
            if route_name == "deep_lane":
                fallback_models = ["llama-3.1-8b-instant", "llama3-70b-8192", "openai/gpt-oss-120b"]
                for alt_model in fallback_models:
                    if alt_model == clean_payload['model']:
                        continue
                    fb_payload = clean_payload.copy()
                    fb_payload["model"] = alt_model
                    print(f"[DEBUG GROQ] URL: {url} | Model: {alt_model}")
                    fb_res = await self.client.post(url, json=fb_payload, headers=headers)
                    if fb_res.status_code == 200:
                        data = fb_res.json()
                        usage = data.get("usage", {})
                        prompt_tokens = usage.get("prompt_tokens", 100)
                        completion_tokens = usage.get("completion_tokens", 50)
                        cost_saved = self.tracker.record_usage(route_name, prompt_tokens, completion_tokens)
                        resp_headers["x-estimated-cost-saved"] = f"{cost_saved:.6f}"
                        resp_headers["x-semantic-model"] = alt_model
                        resp_headers["X-Semantic-Router-Model"] = alt_model
                        return JSONResponse(content=data, status_code=200, headers=resp_headers)
                    else:
                        print(f"[DEBUG GROQ ERROR]: {fb_res.status_code} - {fb_res.text}")

            err_msg = f"[Upstream Error (HTTP {response.status_code})]: {response.text}"

        except httpx.HTTPStatusError as e:
            print(f"[Groq Error {e.response.status_code}]: {e.response.text}")
            err_msg = f"[Upstream Error ({type(e).__name__})]: {str(e)}"
        except httpx.RequestError as e:
            print(f"[Groq Network Error]: {str(e)}")
            err_msg = f"[Upstream Error ({type(e).__name__})]: {str(e)}"
        except Exception as e:
            print(f"[Upstream General Error]: {repr(e)}")
            err_msg = f"[Upstream Error ({type(e).__name__})]: {str(e)}"

        return JSONResponse(
            status_code=200,
            headers=resp_headers,
            content={
                "id": "chatcmpl-upstream-error",
                "object": "chat.completion",
                "created": 1700000000,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": err_msg
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
        )

    async def _forward_stream(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        decision: RouteDecision,
        model_name: str,
        route_name: str
    ) -> StreamingResponse:
        estimated_saved = self.tracker.calculate_cost_saved(route_name, prompt_tokens=100, completion_tokens=50)

        clean_payload = {
            "model": model_name.strip(),
            "messages": payload.get("messages", []),
            "temperature": payload.get("temperature", 0.7),
            "stream": True
        }
        if "max_tokens" in payload and payload["max_tokens"] is not None:
            clean_payload["max_tokens"] = payload["max_tokens"]

        async def event_generator() -> AsyncGenerator[bytes, None]:
            try:
                print(f"[DEBUG GROQ] URL: {url} | Model: {clean_payload['model']}")
                async with self.client.stream("POST", url, json=clean_payload, headers=headers) as response:
                    if response.status_code >= 400:
                        err_bytes = await response.aread()
                        err_text = err_bytes.decode("utf-8", errors="replace")
                        print(f"[DEBUG GROQ ERROR]: {response.status_code} - {err_text}")

                        # Fallback for streaming if primary model 404s
                        if route_name == "deep_lane":
                            for alt_model in ["llama-3.1-8b-instant", "llama3-70b-8192", "openai/gpt-oss-120b"]:
                                if alt_model == clean_payload['model']:
                                    continue
                                fb_payload = clean_payload.copy()
                                fb_payload["model"] = alt_model
                                print(f"[DEBUG GROQ] URL: {url} | Model: {alt_model}")
                                async with self.client.stream("POST", url, json=fb_payload, headers=headers) as fb_res:
                                    if fb_res.status_code == 200:
                                        async for chunk in fb_res.aiter_bytes():
                                            yield chunk
                                        return
                                    else:
                                        fb_err = await fb_res.aread()
                                        print(f"[DEBUG GROQ ERROR]: {fb_res.status_code} - {fb_err.decode('utf-8', errors='replace')}")

                        err_json = {
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "content": f"[Upstream Error (HTTP {response.status_code})]: {err_text}"
                                }
                            }]
                        }
                        yield f"data: {json.dumps(err_json)}\n\n".encode("utf-8")
                        return

                    async for chunk in response.aiter_bytes():
                        yield chunk
            except Exception as e:
                print(f"[Upstream General Error]: {repr(e)}")
                err_json = {"choices": [{"message": {"role": "assistant", "content": f"[Upstream Error ({type(e).__name__})]: {str(e)}"}}]}
                yield f"data: {json.dumps(err_json)}\n\n".encode("utf-8")

        response_headers = {
            "x-semantic-route": route_name,
            "x-semantic-model": model_name,
            "x-semantic-classification-ms": str(decision.classification_time_ms),
            "x-estimated-cost-saved": f"{estimated_saved:.6f}",
            "X-Semantic-Router-Tier": "fast" if route_name == "fast_lane" else "deep",
            "X-Semantic-Router-Model": model_name,
            "X-Semantic-Router-Score": str(decision.similarity_score),
            "X-Semantic-Router-Latency-MS": str(decision.classification_time_ms)
        }

        return StreamingResponse(event_generator(), media_type="text/event-stream", headers=response_headers)
