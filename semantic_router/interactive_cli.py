import sys
import os
import asyncio
from pathlib import Path

# Force UTF-8 encoding for Windows terminal compatibility
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt


async def main():
    console = Console()
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]⚡ SemanticRouter API - Interactive CLI Playground[/bold cyan]\n"
            "[dim]Real-time Sub-15ms Prompt Complexity Classification & LLM Forwarding[/dim]",
            border_style="cyan"
        )
    )
    console.print("[dim]Type your prompt query below. Type 'exit', 'quit', or 'q' to stop.[/dim]\n")

    gateway_url = "http://localhost:8000/v1/chat/completions"
    health_url = "http://localhost:8000/health"

    # Health check on startup
    async with httpx.AsyncClient(timeout=3.0) as client:
        try:
            health_res = await client.get(health_url)
            if health_res.status_code == 200:
                h_data = health_res.json()
                console.print(
                    f"[bold green]✓ Connected to Gateway Server[/bold green] | "
                    f"Fast Model: [cyan]{h_data.get('fast_lane_model', 'gemma2:2b')}[/cyan] | "
                    f"Deep Model: [magenta]{h_data.get('deep_lane_model', 'llama-3.3-70b-versatile')}[/magenta]\n"
                )
            else:
                console.print("[bold yellow]⚠️ Gateway server returned non-200 health check status.[/bold yellow]\n")
        except Exception:
            console.print(
                "[bold red]❌ Cannot connect to Gateway Server at http://localhost:8000[/bold red]\n"
                "[yellow]Please make sure the server is running in another terminal:[/yellow]\n"
                "   [bold white]python -m uvicorn app.main:app --port 8000[/bold white]\n"
            )

    while True:
        try:
            user_input = Prompt.ask("\n[bold yellow]Query[/bold yellow]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Exiting interactive CLI playground. Goodbye![/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            console.print("[dim]Exiting interactive CLI playground. Goodbye![/dim]")
            break

        console.print("[dim]Classifying prompt complexity and forwarding to upstream LLM...[/dim]")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                res = await client.post(
                    gateway_url,
                    json={
                        "model": "auto",
                        "messages": [{"role": "user", "content": user_input}]
                    }
                )

            headers = res.headers
            route = headers.get("x-semantic-route", "deep_lane")
            model = headers.get("x-semantic-model", "unknown")
            latency_ms = headers.get("x-semantic-classification-ms", "0.0")
            saved = headers.get("x-estimated-cost-saved", "0.000000")

            if route == "deep_lane":
                if not model or model == "unknown" or "gemini" in model.lower():
                    model = "llama-3.3-70b-versatile"
            elif route == "fast_lane":
                if not model or model == "unknown":
                    model = "gemma2:2b"

            # Render Classification Telemetry Table
            table = Table(
                title="Semantic Routing Decision",
                title_style="bold cyan",
                show_header=True,
                header_style="bold yellow"
            )
            table.add_column("Route Chosen", justify="center")
            table.add_column("Target Model", style="blue")
            table.add_column("Classification Latency", justify="right", style="bright_green")
            table.add_column("Estimated Cost Saved", justify="right", style="green")

            if route == "fast_lane":
                route_badge = Text("FAST_LANE", style="bold green")
            else:
                route_badge = Text("DEEP_LANE", style="bold magenta")

            table.add_row(
                route_badge,
                model,
                f"{float(latency_ms):.2f} ms",
                f"${float(saved):.6f}"
            )
            console.print(table)

            # Extract generated response text or handle error response
            if res.status_code == 200:
                data = res.json()
                choices = data.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "").strip()
                    console.print(
                        Panel(
                            content,
                            title=f"[bold green]Generated Response ({model})[/bold green]",
                            border_style="green"
                        )
                    )
                else:
                    console.print("[yellow]Response returned 200 OK but no text choices found.[/yellow]")
            else:
                error_detail = res.text
                try:
                    err_json = res.json()
                    error_detail = err_json.get("detail", error_detail)
                except Exception:
                    pass

                console.print(
                    Panel(
                        f"[bold red]Upstream Status Code: {res.status_code}[/bold red]\n\n{error_detail}",
                        title="[bold red]Upstream Execution Info[/bold red]",
                        border_style="red"
                    )
                )

        except httpx.RequestError as exc:
            console.print(
                Panel(
                    f"[bold red]Connection Error:[/bold red] {str(exc)}\n\n"
                    "Make sure the gateway server is running at http://localhost:8000.",
                    title="Error",
                    border_style="red"
                )
            )


if __name__ == "__main__":
    asyncio.run(main())
