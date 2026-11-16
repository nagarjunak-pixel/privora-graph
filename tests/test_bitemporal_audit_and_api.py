"""Tests for Native C++20 CSR PPR, Bi-Temporal Knowledge Graph, Ollama Audit, API, and CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner

from privora_graph.api.app import create_app
from privora_graph.audit.ollama_bitemporal import (
    build_bitemporal_audit_suite,
    run_real_bitemporal_ollama_audit,
)
from privora_graph.bitemporal.graph_engine import (
    BiTemporalKnowledgeGraph,
    build_sovereign_compliance_kg,
    run_bitemporal_ppr_benchmark,
)
from privora_graph.cli import app as cli_app
from privora_graph.kernels.native_graph import PrivoraKernelBridge, get_privora_bridge


def test_cpp20_vs_numpy_ppr_and_batch_cosine_exact_parity() -> None:
    cpp_b = get_privora_bridge()
    py_b = PrivoraKernelBridge(csrc_path=Path("/tmp/nonexistent_privora.cpp"))
    assert cpp_b.is_native_cpp is True
    assert py_b.is_native_cpp is False

    rng = np.random.default_rng(13)
    q = rng.normal(0.0, 1.0, size=24).astype(np.float32)
    corpus = rng.normal(0.0, 1.0, size=(15, 24)).astype(np.float32)
    np.testing.assert_allclose(
        cpp_b.batch_cosine_similarity(q, corpus),
        py_b.batch_cosine_similarity(q, corpus),
        rtol=1e-5,
        atol=1e-5,
    )
    with pytest.raises(ValueError, match="Query dim"):
        cpp_b.batch_cosine_similarity(np.zeros(5, dtype=np.float32), corpus)

    empty_kg = BiTemporalKnowledgeGraph()
    assert empty_kg.query_ppr_as_of(["NONE"], valid_at=20260101, tx_at=20260101) == ({}, 0, [])

    kg = build_sovereign_compliance_kg()
    assert len(kg.nodes) >= 6
    # Unknown seed entity falls back to uniform teleportation cleanly
    ranks_u, _, _ = kg.query_ppr_as_of(["UNKNOWN_SEED"], valid_at=20260501, tx_at=20260601)
    assert abs(sum(ranks_u.values()) - 1.0) < 1e-7

    # Pre-restatement vs Post-restatement time-travel verification
    rep_pre = run_bitemporal_ppr_benchmark(valid_at=20260501, tx_at=20260601)
    rep_post = run_bitemporal_ppr_benchmark(valid_at=20260501, tx_at=20261001)
    assert rep_pre.retroactive_future_edges_blocked == 2
    assert rep_post.active_bitemporal_edges > rep_pre.active_bitemporal_edges - 1
    assert rep_pre.cpp_vs_numpy_max_diff < 1e-10


@settings(max_examples=25, deadline=None)
@given(
    teleport=st.floats(min_value=0.05, max_value=0.85),
    w_scale=st.floats(min_value=0.2, max_value=5.0),
)
def test_hypothesis_csr_ppr_conserves_probability_simplex_with_dangling_nodes(
    teleport: float, w_scale: float
) -> None:
    bridge = get_privora_bridge()
    # 4-node graph where node 2 and node 3 are dangling sinks (zero out-edges)
    row_ptr = np.array([0, 2, 3, 3, 3], dtype=np.int32)
    col_idx = np.array([1, 2, 3], dtype=np.int32)
    weights = np.array([1.0, 2.0, 1.5], dtype=np.float64) * w_scale
    seed = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    ranks, _ = bridge.csr_personalized_pagerank(
        4, row_ptr, col_idx, weights, seed, teleport_alpha=teleport
    )
    assert float(np.min(ranks)) >= 0.0
    assert abs(float(np.sum(ranks)) - 1.0) < 1e-8


def test_bitemporal_ollama_audit_mocked_and_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = build_bitemporal_audit_suite()
    assert len(suite) == 6
    out_file = tmp_path / "bt_audit.json"

    def fake_get(url: str, timeout: float = 2.0) -> httpx.Response:
        return httpx.Response(200, json={"models": []}, request=httpx.Request("GET", url))

    def fake_post(url: str, json: dict[str, Any], timeout: float = 45.0) -> httpx.Response:
        prompt = str(json.get("prompt", ""))
        req = httpx.Request("POST", url)
        # If future override text leaked into prompt, simulate LLM returning the leaked code
        for c in suite:
            if c.edges[1].fact_summary in prompt:
                return httpx.Response(
                    200,
                    json={"response": f'{{"reason": "override", "code": {c.leaked_future_code}}}'},
                    request=req,
                )
            if c.edges[0].fact_summary in prompt:
                return httpx.Response(
                    200,
                    json={
                        "response": f'{{"reason": "point-in-time", "code": {c.ground_truth_code}}}'
                    },
                    request=req,
                )
        return httpx.Response(200, json={"response": "invalid"}, request=req)

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(httpx, "post", fake_post)

    live_rep = run_real_bitemporal_ollama_audit(save_path=out_file, allow_offline_replay=False)
    assert live_rep.naive_future_leakage_count == 6
    assert live_rep.bitemporal_accuracy_pct == 100.0
    assert out_file.exists()

    def offline_get(url: str, timeout: float = 2.0) -> httpx.Response:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "get", offline_get)
    replay = run_real_bitemporal_ollama_audit(save_path=out_file, allow_offline_replay=True)
    assert replay.bitemporal_correct_count == 6

    with pytest.raises(RuntimeError, match="unreachable"):
        run_real_bitemporal_ollama_audit(
            save_path=tmp_path / "missing.json", allow_offline_replay=False
        )


async def test_fastapi_endpoints_and_cli_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.get("/healthz")).status_code == 200
        assert (await client.get("/api/v1/overview")).status_code == 200
        assert (
            await client.post("/api/v1/privacy/train", json={"sigma": 1.6, "epochs": 4})
        ).status_code == 200
        assert (
            await client.post(
                "/api/v1/bitemporal/query", json={"valid_at": 20260501, "tx_at": 20261001}
            )
        ).status_code == 200
        assert (
            await client.post(
                "/api/v1/byzantine/simulate",
                json={"total_clients_k": 7, "byzantine_clients_f": 2},
            )
        ).status_code == 200
        assert (await client.get("/")).status_code == 200

    headless = create_app(ui_dist_dir=tmp_path / "no_ui")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=headless), base_url="http://test"
    ) as hc:
        assert (await hc.get("/")).json()["mode"] == "headless-api"

    runner = CliRunner()
    for args in (
        ["train-dp", "--epochs", "4"],
        ["train-dp", "--epochs", "4", "--json"],
        ["query-bitemporal"],
        ["query-bitemporal", "--json"],
        ["simulate-byzantine"],
        ["simulate-byzantine", "--json"],
        ["demo"],
        ["demo", "--json"],
    ):
        assert runner.invoke(cli_app, args).exit_code == 0

    saved = tmp_path / "bt_saved.json"
    saved.write_text(
        json.dumps(
            {
                "model_name": "qwen3.5:4b",
                "num_cases": 1,
                "naive_correct_count": 0,
                "naive_accuracy_pct": 0.0,
                "naive_future_leakage_count": 1,
                "bitemporal_correct_count": 1,
                "bitemporal_accuracy_pct": 100.0,
                "future_leakage_eliminated_pct": 100.0,
                "outcomes": [
                    {
                        "case_id": "BT-01",
                        "domain": "Test",
                        "audit_tx_timestamp": 20260601,
                        "ground_truth_code": 10,
                        "naive_unfiltered_answer": 99,
                        "naive_leaked_future_info": True,
                        "bitemporal_guarded_answer": 10,
                        "bitemporal_exact_match": True,
                        "latency_ms": 120.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def offline_get(url: str, timeout: float = 2.0) -> httpx.Response:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "get", offline_get)
    assert runner.invoke(cli_app, ["eval-ollama", "--save-path", str(saved)]).exit_code == 0
    assert (
        runner.invoke(cli_app, ["eval-ollama", "--save-path", str(saved), "--json"]).exit_code == 0
    )
