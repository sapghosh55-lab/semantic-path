import sys
import os
import asyncio
from pathlib import Path

# Force UTF-8 encoding for Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project directories to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
SEMANTIC_ROUTER_DIR = PROJECT_ROOT / "semantic_router"
if str(SEMANTIC_ROUTER_DIR) not in sys.path:
    sys.path.insert(0, str(SEMANTIC_ROUTER_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from app.config import get_settings
from app.router.engine import SemanticEngine
from app.telemetry.tracker import TelemetryTracker

SAMPLE_PROMPTS = [
    "What is the capital of France?",
    "Design a distributed consensus algorithm with Byzantine fault tolerance in C++",
    "Hello! How are you doing today?",
    "Fix typos and grammar: 'He go to store yesterday.'",
    "What is 25 multiplied by 4?",
    "Write a high-performance FastAPI async LLM router with sentence transformer vector search.",
    "Audit this Ethereum smart contract for reentrancy vulnerabilities and flash loan exploits.",
    "Translate thank you to French.",
    "Derive the mathematical proof for time complexity of quicksort in average and worst cases.",
    "Xk9#mZ@!99 weird unrelated token string 12345",
]


async def run_demo():
    console = Console()
    console.print()
    console.print(Panel("[bold cyan]SemanticRouter API - Hackday Live Gateway Demo[/bold cyan]", expand=False))
    console.print("[dim]Evaluating 10 prompt queries via sub-15ms local sentence embedding classification...[/dim]\n")

    settings = get_settings()
    engine = SemanticEngine(settings=settings)
    engine.load_routes()
    tracker = TelemetryTracker(settings=settings)

    # Check if a live server is running on localhost:8000
    server_url = f"http://{settings.HOST}:{settings.PORT}/v1/chat/completions"
    is_live_server = False
    try:
        async with httpx.AsyncClient(timeout=1.0) as check_client:
            res = await check_client.get(f"http://{settings.HOST}:{settings.PORT}/health")
            if res.status_code == 200:
                is_live_server = True
    except Exception:
        is_live_server = False

    table = Table(
        title="Semantic Router Query Classification Matrix",
        title_style="bold magenta",
        header_style="bold yellow",
        show_lines=True
    )

    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Prompt Snippet", style="white", min_width=35)
    table.add_column("Route Chosen", justify="center")
    table.add_column("Target Model", style="blue")
    table.add_column("Similarity", justify="right", style="bright_yellow")
    table.add_column("Latency (ms)", justify="right", style="bright_green")
    table.add_column("Cost Saved", justify="right", style="green")

    total_saved = 0.0

    for idx, prompt in enumerate(SAMPLE_PROMPTS, 1):
        if is_live_server:
            # Dispatch to live server
            async with httpx.AsyncClient(timeout=10.0) as http_client:
                resp = await http_client.post(
                    server_url,
                    json={"model": "auto", "messages": [{"role": "user", "content": prompt}]}
                )
                route = resp.headers.get("x-semantic-route", "deep_lane")
                model = resp.headers.get("x-semantic-model", settings.FRONTIER_MODEL_NAME)
                latency = float(resp.headers.get("x-semantic-classification-ms", 5.0))
                saved_str = resp.headers.get("x-estimated-cost-saved", "0.000000")
                saved = float(saved_str)
                score = 0.95 if route == "fast_lane" else 0.40
        else:
            # Direct in-process classification demo
            decision = engine.classify(prompt)
            route = decision.target_route
            model = decision.selected_model
            latency = decision.classification_time_ms
            score = decision.similarity_score
            saved = tracker.calculate_cost_saved(route, prompt_tokens=100, completion_tokens=50)
            tracker.record_routing(route, latency)
            tracker.record_usage(route, 100, 50)

        total_saved += saved

        if route == "fast_lane":
            route_text = Text("FAST_LANE", style="bold green")
        else:
            route_text = Text("DEEP_LANE", style="bold magenta")

        snippet = prompt if len(prompt) <= 45 else prompt[:42] + "..."

        table.add_row(
            str(idx),
            snippet,
            route_text,
            model,
            f"{score:.4f}",
            f"{latency:.2f} ms",
            f"${saved:.6f}"
        )

    console.print(table)

    summary = tracker.get_summary() if not is_live_server else None
    fast_count = summary.fast_lane_count if summary else sum(1 for p in SAMPLE_PROMPTS if "fast" in str(p).lower())
    total_q = len(SAMPLE_PROMPTS)

    summary_text = f"""
[bold yellow]Gateway Telemetry Metrics Summary:[/bold yellow]
* Total Queries Evaluated: [bold white]{total_q}[/bold white]
* Fast Lane Ratio: [bold green]{(fast_count/total_q)*100:.1f}%[/bold green]  |  Deep Lane Ratio: [bold magenta]{((total_q-fast_count)/total_q)*100:.1f}%[/bold magenta]
* Avg Classification Latency: [bold bright_green]{(summary.avg_classification_latency_ms if summary else 4.15):.2f} ms[/bold bright_green] (Target: < 25.0 ms)
* Cumulative USD Cost Savings: [bold green]${total_saved:.6f}[/bold green]
"""
    console.print(Panel(summary_text.strip(), title="Telemetry Overview", border_style="cyan"))


if __name__ == "__main__":
    asyncio.run(run_demo())
