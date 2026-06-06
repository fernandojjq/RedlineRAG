"""RedlineRAG - Terms of Service Risk Auditor (command line entry point).

A zero-friction CLI: with no arguments it ingests, indexes, and runs a
default risk scan; with arguments the user can pass their own question,
rebuild the index, or query a specific document.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Make `src` importable when running this file directly as a script.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.auditor import MatchLocation  # noqa: E402
from src.pipeline.orchestrator import RagPipeline  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging_setup import configure_logging  # noqa: E402

app = typer.Typer(
    name="redline-rag",
    help="Audit Terms of Service and privacy policies for hidden legal risks.",
    add_completion=False,
    no_args_is_help=False,
)
console = Console()


# -- Default risk-scan questions ------------------------------------------------
# These cover the most common legal-trap families. Running the CLI with no
# arguments performs all of them and produces a consolidated report.

DEFAULT_RISK_QUESTIONS: tuple[str, ...] = (
    "binding arbitration and class action waiver",
    "perpetual irrevocable content license",
    "right to change terms without notice",
    "selling personal data to third parties",
    "tracking pixels and device fingerprinting",
    "account termination and data deletion",
    "liability disclaimer and warranty",
    "user indemnification",
    "international data transfer",
    "payment card data storage",
)


def _render_assessment(result, console: Console) -> None:
    """Print a human-friendly audit report to the terminal."""
    assessment = result.assessment
    severity = assessment.severity_summary

    header = (
        f"[bold]Question:[/bold] {assessment.question}\n"
        f"[dim]Documents scanned:[/dim] {', '.join(assessment.documents_scanned) or '(none)'}\n"
        f"[dim]Chunks retrieved:[/dim] {assessment.retrieved_chunk_count}"
    )
    console.print(Panel(header, title="Risk Audit", border_style="cyan"))

    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Severity", style="bold")
    summary_table.add_column("Count", justify="right")
    for level in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        count = severity.get(level, 0)
        color = {
            "CRITICAL": "red",
            "HIGH": "orange1",
            "MEDIUM": "yellow",
            "LOW": "green",
        }[level]
        summary_table.add_row(f"[{color}]{level}[/{color}]", str(count))
    console.print(summary_table)

    if not assessment.findings:
        console.print(
            Panel(
                "No flagged clauses for this query. Either the corpus is "
                "clean or the query did not match any indexed text.",
                border_style="green",
            )
        )
        return

    for index, finding in enumerate(assessment.findings, start=1):
        color = {
            "CRITICAL": "red",
            "HIGH": "orange1",
            "MEDIUM": "yellow",
            "LOW": "green",
        }[finding.severity.value]
        # Tag primary vs co-located in the title so the user knows which
        # findings are about their actual question and which are nearby
        # clauses that showed up in the same paragraph.
        if finding.match_location == MatchLocation.PRIMARY:
            location_tag = "[cyan]primary[/cyan]"
        else:
            location_tag = "[dim]co-located[/dim]"
        # Off-topic tag: the pattern fired but no key signal matched, so
        # the cited text is probably about a different topic. We show it
        # demoted and clearly marked so the user does not trust it.
        if finding.off_topic:
            topic_tag = "[yellow]off-topic[/yellow]"
        else:
            topic_tag = "[green]on-topic[/green]"
        mitigation_note = (
            "[green]Mitigating language detected in same text.[/green]"
            if finding.mitigations_detected
            else "[red]No mitigating language detected.[/red]"
        )
        body = (
            f"[bold]{finding.family}[/bold]  ({location_tag}  ·  {topic_tag})\n"
            f"[dim]Source:[/dim] {finding.document_title}  "
            f"[dim]Score:[/dim] {finding.score:.3f}\n\n"
            f"{finding.description}\n\n"
            f"[bold]Cited text:[/bold]\n"
            f"[italic]\u201c{finding.excerpt}\u201d[/italic]\n\n"
            f"[dim]Parent paragraph:[/dim]\n"
            f"{finding.parent_excerpt}"
            f"\n\n"
            f"{mitigation_note}"
        )
        console.print(
            Panel(
                body,
                title=f"#{index} [{color}]{finding.severity.value}[/{color}]",
                border_style=color,
            )
        )


# -- Commands -------------------------------------------------------------------


@app.command()
def scan(
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        help="Force a full rebuild of the vector store from scratch.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit a machine-readable JSON report instead of pretty output.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable INFO-level logging (default: warnings only).",
    ),
) -> None:
    """Run the full default risk scan (all common legal-trap families)."""
    configure_logging("INFO" if verbose else "WARNING")
    pipeline = RagPipeline()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task("Ingesting documents...", total=None)
        pipeline.ingest()
        progress.add_task("Indexing corpus...", total=None)
        pipeline.index(force_rebuild=rebuild)
        progress.add_task("Running risk scan...", total=None)
        results = [pipeline.ask(question) for question in DEFAULT_RISK_QUESTIONS]

    if json_output:
        payload = {
            "documents_ingested": results[0].documents_ingested,
            "chunks_indexed": results[0].chunks_indexed,
            "questions": [result.to_dict() for result in results],
        }
        console.print_json(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    console.rule("[bold cyan]RedlineRAG - Default Risk Scan")
    console.print(
        f"[dim]Documents ingested:[/dim] {', '.join(results[0].documents_ingested)}"
    )
    console.print(f"[dim]Chunks indexed:[/dim] {results[0].chunks_indexed}\n")

    for result in results:
        _render_assessment(result, console)
        console.print()


@app.command()
def ask(
    question: str = typer.Argument(..., help="Free-text risk to look for."),
    top_k: int = typer.Option(5, "--top-k", "-k", min=1, max=20),
    rebuild: bool = typer.Option(False, "--rebuild", help="Rebuild the index first."),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable INFO logging."),
) -> None:
    """Run a single custom query against the indexed corpus."""
    configure_logging("INFO" if verbose else "WARNING")
    pipeline = RagPipeline()
    pipeline.ingest()
    pipeline.index(force_rebuild=rebuild)
    result = pipeline.ask(question, top_k=top_k)

    if json_output:
        console.print_json(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return

    console.rule("[bold cyan]RedlineRAG - Custom Query")
    _render_assessment(result, console)


@app.command()
def info() -> None:
    """Print configuration and on-disk layout diagnostics."""
    configure_logging("WARNING")
    config = load_config()
    table = Table(title="RedlineRAG Configuration", header_style="bold magenta")
    table.add_column("Setting", style="bold")
    table.add_column("Value")
    for field_name, value in config.model_dump().items():
        if isinstance(value, Path):
            value = str(value)
        table.add_row(field_name, str(value))
    console.print(table)

    for label, path in (
        ("Raw documents", config.raw_documents_dir),
        ("Mock documents", config.mock_documents_dir),
        ("Vector store", config.vector_store_dir),
    ):
        exists = path.exists()
        marker = "[green]present[/green]" if exists else "[yellow]missing[/yellow]"
        console.print(f"  {label}: {path}  ({marker})")


def main() -> None:
    """Typer entry point."""
    if len(sys.argv) == 1:
        # No subcommand: default to a full scan. Easier for first-time users.
        sys.argv.append("scan")
    app()


if __name__ == "__main__":
    main()
