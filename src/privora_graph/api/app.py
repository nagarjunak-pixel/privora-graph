"""FastAPI control plane and Sovereign Privacy, Bi-Temporal GraphRAG & Byzantine Workbench."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from privora_graph.audit.ollama_bitemporal import default_bitemporal_artifact_path
from privora_graph.bitemporal.graph_engine import run_bitemporal_ppr_benchmark
from privora_graph.byzantine.robust_fed import run_byzantine_federated_benchmark
from privora_graph.kernels.native_graph import get_privora_bridge
from privora_graph.privacy.dp_sgd import run_dp_sgd_vs_mia_benchmark
from privora_graph.types import (
    BiTemporalGraphReport,
    ByzantineAggregationReport,
    DPSGDPrivacyReport,
    RealBiTemporalAuditReport,
)


class DPSGDRequest(BaseModel):
    sigma: float = Field(default=1.45, ge=0.4, le=5.0)
    clip_c: float = Field(default=1.0, ge=0.1, le=5.0)
    epochs: int = Field(default=10, ge=3, le=30)
    seed: int = Field(default=41)


class BiTemporalQueryRequest(BaseModel):
    valid_at: int = Field(default=20260501)
    tx_at: int = Field(default=20260601)


class ByzantineRequest(BaseModel):
    total_clients_k: int = Field(default=9, ge=5, le=21)
    byzantine_clients_f: int = Field(default=3, ge=1, le=9)
    poison_scale: float = Field(default=24.0, ge=2.0, le=100.0)
    seed: int = Field(default=53)


_WORKBENCH_CACHE: dict[str, Any] | None = None


def _load_bitemporal_audit() -> dict[str, Any] | None:
    artifact = default_bitemporal_artifact_path()
    if not artifact.exists():
        return None
    rep = RealBiTemporalAuditReport.model_validate(json.loads(artifact.read_text(encoding="utf-8")))
    return rep.model_dump()


def _get_workbench_cache() -> dict[str, Any]:
    global _WORKBENCH_CACHE
    if _WORKBENCH_CACHE is None:
        bridge = get_privora_bridge()
        dp_rep = run_dp_sgd_vs_mia_benchmark()
        bt_rep = run_bitemporal_ppr_benchmark()
        byz_rep = run_byzantine_federated_benchmark()
        _WORKBENCH_CACHE = {
            "service": "privora-graph",
            "version": "1.0.0",
            "kernel_backend": bridge.backend,
            "is_native_cpp": bridge.is_native_cpp,
            "dp_sgd": dp_rep.model_dump(),
            "bitemporal": bt_rep.model_dump(),
            "byzantine": byz_rep.model_dump(),
            "real_qwen_bitemporal": _load_bitemporal_audit(),
        }
    else:
        _WORKBENCH_CACHE["real_qwen_bitemporal"] = _load_bitemporal_audit()
    return _WORKBENCH_CACHE


def create_app(ui_dist_dir: Path | None = None) -> FastAPI:
    app = FastAPI(
        title="PrivoraGraph Sovereign Differential Privacy, Bi-Temporal GraphRAG & Byzantine Engine",
        version="1.0.0",
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        bridge = get_privora_bridge()
        return {
            "status": "ok",
            "service": "privora-graph",
            "version": "1.0.0",
            "kernel_backend": bridge.backend,
            "is_native_cpp": bridge.is_native_cpp,
        }

    @app.get("/api/v1/overview")
    async def overview() -> dict[str, Any]:
        return _get_workbench_cache()

    @app.post("/api/v1/privacy/train", response_model=DPSGDPrivacyReport)
    async def dp_train(req: DPSGDRequest) -> DPSGDPrivacyReport:
        return run_dp_sgd_vs_mia_benchmark(
            clip_c=req.clip_c,
            sigma=req.sigma,
            epochs=req.epochs,
            seed=req.seed,
        )

    @app.post("/api/v1/bitemporal/query", response_model=BiTemporalGraphReport)
    async def bt_query(req: BiTemporalQueryRequest) -> BiTemporalGraphReport:
        return run_bitemporal_ppr_benchmark(valid_at=req.valid_at, tx_at=req.tx_at)

    @app.post("/api/v1/byzantine/simulate", response_model=ByzantineAggregationReport)
    async def byz_sim(req: ByzantineRequest) -> ByzantineAggregationReport:
        return run_byzantine_federated_benchmark(
            total_clients_k=req.total_clients_k,
            byzantine_clients_f=req.byzantine_clients_f,
            poison_scale=req.poison_scale,
            seed=req.seed,
        )

    resolved_dist = ui_dist_dir or (Path(__file__).resolve().parents[3] / "ui" / "dist")
    assets_dir = resolved_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/", response_model=None)
    async def root_page() -> FileResponse | JSONResponse:
        index_html = resolved_dist / "index.html"
        if index_html.exists():
            return FileResponse(index_html)
        return JSONResponse(
            {
                "service": "privora-graph",
                "mode": "headless-api",
                "docs": "/docs",
                "overview": "/api/v1/overview",
            }
        )

    return app


app = create_app()
