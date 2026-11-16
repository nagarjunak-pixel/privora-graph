"""Rich & JSON command-line interface for PrivoraGraph (`privora`)."""

from __future__ import annotations

import json
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from privora_graph.api.app import create_app
from privora_graph.audit.ollama_bitemporal import run_real_bitemporal_ollama_audit
from privora_graph.bitemporal.graph_engine import run_bitemporal_ppr_benchmark
from privora_graph.byzantine.robust_fed import run_byzantine_federated_benchmark
from privora_graph.privacy.dp_sgd import run_dp_sgd_vs_mia_benchmark

app = typer.Typer(
    name="privora",
    help="PrivoraGraph: Sovereign DP-SGD (Rényi Accountant), Bi-Temporal PPR GraphRAG & Byzantine Engine",
    add_completion=False,
)
console = Console()


@app.command("train-dp")
def train_dp_cmd(
    sigma: float = typer.Option(1.45, help="Gaussian noise multiplier sigma"),
    clip_c: float = typer.Option(1.0, help="Per-sample L2 clipping norm C"),
    epochs: int = typer.Option(10, help="Training epochs"),
    as_json: bool = typer.Option(False, "--json", help="Emit structured JSON report"),
) -> None:
    """Run per-sample clipped DP-SGD with exact Rényi Moments Accountant & MIA empirical audit."""
    rep = run_dp_sgd_vs_mia_benchmark(clip_c=clip_c, sigma=sigma, epochs=epochs)
    if as_json:
        typer.echo(json.dumps(rep.model_dump(), indent=2))
        return
    tbl = Table(
        title=f"PrivoraGraph Rényi DP-SGD & Membership-Inference Audit (delta={rep.target_delta})"
    )
    tbl.add_column("Metric", style="cyan")
    tbl.add_column("Value", justify="right", style="green")
    tbl.add_row(
        "Certified (epsilon, delta) @ Optimal Rényi alpha*",
        f"eps = {rep.final_certified_epsilon:.4f} (alpha* = {rep.optimal_renyi_alpha:.0f})",
    )
    tbl.add_row(
        "Test Generalization (Non-Private vs DP-SGD)",
        f"{rep.non_private_test_accuracy_pct:.2f}% vs {rep.dp_test_accuracy_pct:.2f}%",
    )
    tbl.add_row(
        "Membership Inference Attack (MIA) AUC: Non-Private",
        f"{rep.non_private_mia_leakage_auc:.4f} (LEAKAGE)",
    )
    tbl.add_row(
        "Membership Inference Attack (MIA) AUC: DP-SGD",
        f"{rep.dp_mia_leakage_auc:.4f} (~0.50 Random)",
    )
    tbl.add_row("MIA Privacy Advantage Eliminated", f"-{rep.mia_privacy_risk_reduction_pct:.2f}%")
    console.print(tbl)


@app.command("query-bitemporal")
def query_bitemporal_cmd(
    valid_at: int = typer.Option(20260501, help="Valid-time query date YYYYMMDD"),
    tx_at: int = typer.Option(20260601, help="Transaction-time audit date YYYYMMDD"),
    as_json: bool = typer.Option(False, "--json", help="Emit structured JSON report"),
) -> None:
    """Execute point-in-time Bi-Temporal CSR Personalized PageRank via native C++20 kernel."""
    rep = run_bitemporal_ppr_benchmark(valid_at=valid_at, tx_at=tx_at)
    if as_json:
        typer.echo(json.dumps(rep.model_dump(), indent=2))
        return
    tbl = Table(
        title=f"Bi-Temporal Personalized PageRank ({rep.kernel_backend} • tx_at={rep.query_tx_at})"
    )
    tbl.add_column("Entity", style="cyan")
    tbl.add_column("PPR Score", justify="right", style="green")
    tbl.add_column("Point-in-Time Grounded Fact")
    for f in rep.top_ranked_entities:
        tbl.add_row(f.entity, f"{f.ppr_score:.6f}", f.fact_summary)
    console.print(tbl)
    console.print(
        f"[bold]Invariants:[/] Retroactive Future Leaks Blocked = {rep.retroactive_future_edges_blocked} | "
        f"Simplex Sum = {rep.simplex_mass_sum:.8f} | C++20 vs NumPy |Delta| = {rep.cpp_vs_numpy_max_diff:.12f}"
    )


@app.command("simulate-byzantine")
def simulate_byzantine_cmd(
    clients: int = typer.Option(9, help="Total federated silos K"),
    byzantine: int = typer.Option(3, help="Byzantine adversarial silos f"),
    as_json: bool = typer.Option(False, "--json", help="Emit structured JSON report"),
) -> None:
    """Compare Naive FedAvg vs Multi-Krum vs Smoothed Weiszfeld Geometric Median (RFA)."""
    rep = run_byzantine_federated_benchmark(total_clients_k=clients, byzantine_clients_f=byzantine)
    if as_json:
        typer.echo(json.dumps(rep.model_dump(), indent=2))
        return
    tbl = Table(
        title=f"Byzantine Federated Aggregation (K={rep.total_clients_k}, f={rep.byzantine_clients_f}, Poison={rep.poison_magnitude_x}x)"
    )
    tbl.add_column("Aggregator", style="cyan")
    tbl.add_column("Cosine to Honest Mean", justify="right", style="green")
    tbl.add_row("Naive FedAvg (Vulnerable Mean)", f"{rep.fedavg_cosine_to_honest:.5f}")
    tbl.add_row("Blanchard et al. Multi-Krum", f"{rep.multikrum_cosine_to_honest:.5f}")
    tbl.add_row(
        "Smoothed Weiszfeld Geometric Median (RFA)",
        f"{rep.geomedian_rfa_cosine_to_honest:.5f} ({rep.corruption_reduction_x:.2f}x L2 reduction)",
    )
    console.print(tbl)


@app.command("eval-ollama")
def eval_ollama_cmd(
    model: str = typer.Option("qwen3.5:4b", help="Ollama model tag"),
    live_only: bool = typer.Option(False, "--live-only", help="Require live Ollama inference"),
    save_path: Path | None = typer.Option(None, help="Optional custom output JSON path"),
    as_json: bool = typer.Option(False, "--json", help="Emit structured JSON report"),
) -> None:
    """Run live Bi-Temporal Point-in-Time Compliance RAG audit on Ollama."""
    rep = run_real_bitemporal_ollama_audit(
        model=model,
        save_path=save_path,
        allow_offline_replay=not live_only,
    )
    if as_json:
        typer.echo(json.dumps(rep.model_dump(), indent=2))
        return
    tbl = Table(title=f"Zero-Mock Bi-Temporal Lookahead Audit ({rep.model_name})")
    tbl.add_column("Case ID", style="cyan")
    tbl.add_column("Truth", justify="right")
    tbl.add_column("Naive GraphRAG", justify="right")
    tbl.add_column("Privora Bi-Temporal", justify="right")
    for o in rep.outcomes:
        n_str = (
            f"[red]{o.naive_unfiltered_answer} (LEAK)[/]"
            if o.naive_leaked_future_info
            else str(o.naive_unfiltered_answer)
        )
        b_str = (
            f"[green]{o.bitemporal_guarded_answer} ✓[/]"
            if o.bitemporal_exact_match
            else f"[red]{o.bitemporal_guarded_answer} ✗[/]"
        )
        tbl.add_row(o.case_id, str(o.ground_truth_code), n_str, b_str)
    console.print(tbl)
    console.print(
        f"[bold]Summary:[/] Naive Unfiltered = {rep.naive_correct_count}/{rep.num_cases} ({rep.naive_accuracy_pct:.1f}%, "
        f"{rep.naive_future_leakage_count} Future Leaks) -> Privora Bi-Temporal = "
        f"{rep.bitemporal_correct_count}/{rep.num_cases} ({rep.bitemporal_accuracy_pct:.1f}%)"
    )


@app.command("demo")
def demo_cmd(
    as_json: bool = typer.Option(False, "--json", help="Emit unified JSON report"),
) -> None:
    """Run all deterministic PrivoraGraph subsystems end-to-end."""
    dp = run_dp_sgd_vs_mia_benchmark()
    bt = run_bitemporal_ppr_benchmark()
    byz = run_byzantine_federated_benchmark()
    payload = {
        "dp_sgd": dp.model_dump(),
        "bitemporal": bt.model_dump(),
        "byzantine": byz.model_dump(),
    }
    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return
    console.print(
        f"[bold cyan]1. Rényi DP-SGD:[/] eps={dp.final_certified_epsilon:.4f} (delta={dp.target_delta}) | MIA AUC {dp.non_private_mia_leakage_auc:.3f} -> {dp.dp_mia_leakage_auc:.3f} (-{dp.mia_privacy_risk_reduction_pct:.1f}% risk)"
    )
    console.print(
        f"[bold magenta]2. Bi-Temporal PPR ({bt.kernel_backend}):[/] Blocked {bt.retroactive_future_edges_blocked} future leaks | Simplex={bt.simplex_mass_sum:.8f}"
    )
    console.print(
        f"[bold green]3. Byzantine RFA (K={byz.total_clients_k}, f={byz.byzantine_clients_f}):[/] FedAvg cos={byz.fedavg_cosine_to_honest:.4f} -> Weiszfeld RFA cos={byz.geomedian_rfa_cosine_to_honest:.4f}"
    )


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8060, help="Bind port"),
) -> None:
    """Launch the PrivoraGraph FastAPI & Sovereign Privacy Workbench server."""
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    app()
