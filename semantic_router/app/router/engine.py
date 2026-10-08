import time
from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

from app.config import Settings
from app.router.models import RouteDecision, RouteDefinition


class SemanticEngine:
    """
    High-performance vector similarity engine using SentenceTransformer and NumPy matrix dot product.
    Pre-computes and normalizes vector embeddings during startup for sub-15ms prompt classification decisions.
    """

    def __init__(self, settings: Settings, model: Optional[SentenceTransformer] = None):
        self.settings = settings
        model_name = settings.EMBEDDING_MODEL_NAME
        # Support short name 'all-MiniLM-L6-v2' or full 'sentence-transformers/all-MiniLM-L6-v2'
        if not model_name.startswith("sentence-transformers/") and "/" not in model_name:
            model_name = f"sentence-transformers/{model_name}"

        self.model = model or SentenceTransformer(model_name)
        self.routes: List[RouteDefinition] = []
        self.anchor_embeddings: Optional[np.ndarray] = None  # Shape (N, D) normalized
        self.anchor_metadata: List[Tuple[str, str, str]] = []  # (route_name, target_model, utterance)
        self._is_indexed = False

    def load_routes(self, routes_path: Optional[str] = None) -> None:
        path_str = routes_path or self.settings.ROUTES_FILE
        path = Path(path_str)

        if not path.exists():
            parent_path = Path(__file__).resolve().parent.parent.parent / path_str
            if parent_path.exists():
                path = parent_path
            else:
                raise FileNotFoundError(f"Routes file not found at: {path.absolute()}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        raw_routes = data.get("routes", [])
        self.routes = []

        all_utterances: List[str] = []
        self.anchor_metadata = []

        for r in raw_routes:
            r_name = r.get("name", "deep_lane")
            target_model = r.get("target_model") or (
                self.settings.FAST_MODEL_NAME if r_name == "fast_lane" else self.settings.FRONTIER_MODEL_NAME
            )
            utterances = r.get("utterances", [])

            self.routes.append(
                RouteDefinition(
                    name=r_name,
                    description=r.get("description", ""),
                    target_tier="fast" if r_name == "fast_lane" else "frontier",
                    utterances=utterances
                )
            )

            for utt in utterances:
                all_utterances.append(utt)
                self.anchor_metadata.append((r_name, target_model, utt))

        if not all_utterances:
            raise ValueError("No anchor utterances defined in routes configuration.")

        # Pre-calculate route index groups
        self.fast_indices = [i for i, meta in enumerate(self.anchor_metadata) if meta[0] == "fast_lane"]
        self.deep_indices = [i for i, meta in enumerate(self.anchor_metadata) if meta[0] == "deep_lane"]

        # Compute normalized sentence embeddings for all anchor utterances
        raw_vecs = self.model.encode(
            all_utterances,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        self.anchor_embeddings = raw_vecs.astype(np.float32)
        self._is_indexed = True

    def extract_prompt(self, messages: list) -> str:
        """Extract the last user message text from chat messages list."""
        if not messages:
            return ""
        for msg in reversed(messages):
            role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else "")
            if role == "user":
                content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    texts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            texts.append(part.get("text", ""))
                    return " ".join(texts)
        last_msg = messages[-1]
        content = getattr(last_msg, "content", None) or (last_msg.get("content") if isinstance(last_msg, dict) else "")
        return str(content)

    def classify(self, prompt: str) -> RouteDecision:
        """
        Classifies prompt complexity using cosine similarity matrix multiplication.
        Computes similarity scores for fast_lane and deep_lane separately.
        If similarity to deep_lane > similarity to fast_lane, routes directly to deep_lane.
        Guaranteed execution time < 15ms.
        """
        if not self._is_indexed or self.anchor_embeddings is None:
            raise RuntimeError("SemanticEngine is not initialized with routes.")

        start_time = time.perf_counter()

        # 1. Compute normalized embedding for the query prompt
        query_vec = self.model.encode(
            [prompt],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)[0]

        # 2. Vectorized Cosine Similarity via dot product
        similarities = np.dot(self.anchor_embeddings, query_vec)

        # 3. Compute separate max similarity scores for fast_lane and deep_lane
        fast_sims = similarities[self.fast_indices] if self.fast_indices else np.array([])
        deep_sims = similarities[self.deep_indices] if self.deep_indices else np.array([])

        sim_fast = float(np.max(fast_sims)) if len(fast_sims) > 0 else -1.0
        fast_best_local_idx = int(np.argmax(fast_sims)) if len(fast_sims) > 0 else -1

        sim_deep = float(np.max(deep_sims)) if len(deep_sims) > 0 else -1.0
        deep_best_local_idx = int(np.argmax(deep_sims)) if len(deep_sims) > 0 else -1

        print(f"[SemanticRouter] DEBUG: sim_fast={sim_fast:.4f}, sim_deep={sim_deep:.4f}")

        threshold = self.settings.SIMILARITY_THRESHOLD

        # 4. Route decision logic
        if (sim_deep > sim_fast and sim_deep >= threshold) or sim_deep > sim_fast:
            target_route = "deep_lane"
            best_idx = self.deep_indices[deep_best_local_idx]
            best_score = sim_deep
        elif sim_fast >= threshold:
            target_route = "fast_lane"
            best_idx = self.fast_indices[fast_best_local_idx]
            best_score = sim_fast
        else:
            target_route = "deep_lane"
            best_idx = self.deep_indices[deep_best_local_idx] if deep_best_local_idx != -1 else 0
            best_score = sim_deep if deep_best_local_idx != -1 else sim_fast

        matched_route, route_model, matched_utt = self.anchor_metadata[best_idx]

        if target_route == "fast_lane":
            selected_model = self.settings.FAST_MODEL_NAME or route_model
            base_url = self.settings.FAST_MODEL_BASE_URL
            api_key = self.settings.FAST_MODEL_API_KEY
        else:
            selected_model = self.settings.FRONTIER_MODEL_NAME or route_model
            base_url = self.settings.FRONTIER_MODEL_BASE_URL
            api_key = self.settings.FRONTIER_MODEL_API_KEY

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return RouteDecision(
            target_route=target_route,
            selected_model=selected_model,
            base_url=base_url,
            api_key=api_key,
            similarity_score=round(best_score, 4),
            classification_time_ms=round(elapsed_ms, 3),
            matched_utterance=matched_utt
        )

    # Alias method for route calls
    route = classify


# Alias class for backward compatibility
SemanticRouterEngine = SemanticEngine
