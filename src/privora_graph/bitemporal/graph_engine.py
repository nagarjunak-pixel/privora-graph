"""Bi-Temporal (`Valid-Time` x `Transaction-Time`, Snodgrass SQL:2011) Knowledge Graph &
Teleport-Seeded Personalized PageRank (`PPR`) Retrieval Engine eliminating Lookahead Leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from privora_graph.kernels.native_graph import PrivoraKernelBridge, get_privora_bridge
from privora_graph.types import BiTemporalGraphReport, BiTemporalRankedFact

INF_TS: int = 9999_99_99


@dataclass(frozen=True)
class BiTemporalEdge:
    src: str
    relation: str
    dst: str
    weight: float
    valid_from: int
    valid_to: int
    tx_recorded: int
    tx_superseded: int = INF_TS
    fact_summary: str = ""

    def is_active_as_of(self, *, valid_at: int, tx_at: int) -> bool:
        return (
            self.valid_from <= valid_at < self.valid_to
            and self.tx_recorded <= tx_at < self.tx_superseded
        )


class BiTemporalKnowledgeGraph:
    """Immutable-history bi-temporal relational store with CSR C++20 Personalized PageRank."""

    def __init__(self) -> None:
        self._edges: list[BiTemporalEdge] = []
        self._nodes: list[str] = []
        self._node_to_idx: dict[str, int] = {}

    def _ensure_node(self, name: str) -> int:
        if name not in self._node_to_idx:
            self._node_to_idx[name] = len(self._nodes)
            self._nodes.append(name)
        return self._node_to_idx[name]

    def add_edge(self, edge: BiTemporalEdge) -> None:
        self._ensure_node(edge.src)
        self._ensure_node(edge.dst)
        self._edges.append(edge)

    @property
    def total_edges(self) -> int:
        return len(self._edges)

    @property
    def nodes(self) -> list[str]:
        return list(self._nodes)

    def snapshot_edges(self, *, valid_at: int, tx_at: int) -> list[BiTemporalEdge]:
        return [e for e in self._edges if e.is_active_as_of(valid_at=valid_at, tx_at=tx_at)]

    def build_csr_as_of(
        self, *, valid_at: int, tx_at: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[BiTemporalEdge]]:
        active = self.snapshot_edges(valid_at=valid_at, tx_at=tx_at)
        n = len(self._nodes)
        adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        for e in active:
            u = self._node_to_idx[e.src]
            v = self._node_to_idx[e.dst]
            adj[u].append((v, e.weight))

        row_ptr = np.zeros(n + 1, dtype=np.int32)
        cols: list[int] = []
        weights: list[float] = []
        for u in range(n):
            row_ptr[u] = len(cols)
            for v, w in adj[u]:
                cols.append(v)
                weights.append(w)
        row_ptr[n] = len(cols)
        return (
            row_ptr,
            np.array(cols, dtype=np.int32),
            np.array(weights, dtype=np.float64),
            active,
        )

    def query_ppr_as_of(
        self,
        seed_entities: list[str],
        *,
        valid_at: int,
        tx_at: int,
        teleport_alpha: float = 0.20,
        bridge: PrivoraKernelBridge | None = None,
    ) -> tuple[dict[str, float], int, list[BiTemporalEdge]]:
        if not self._nodes:
            return {}, 0, []
        b = bridge or get_privora_bridge()
        rp, ci, ew, active = self.build_csr_as_of(valid_at=valid_at, tx_at=tx_at)
        n = len(self._nodes)
        seed_vec = np.zeros(n, dtype=np.float64)
        for s in seed_entities:
            if s in self._node_to_idx:
                seed_vec[self._node_to_idx[s]] = 1.0
        if float(np.sum(seed_vec)) == 0.0:
            seed_vec[:] = 1.0 / float(n)
        else:
            seed_vec /= float(np.sum(seed_vec))

        ranks, iters = b.csr_personalized_pagerank(
            n, rp, ci, ew, seed_vec, teleport_alpha=teleport_alpha
        )
        ranked_map = {self._nodes[i]: float(ranks[i]) for i in range(n)}
        return ranked_map, iters, active


def build_sovereign_compliance_kg() -> BiTemporalKnowledgeGraph:
    """Construct a multi-hop M&A / Treasury Sanctions / Clinical Trial Bi-Temporal Knowledge Graph
    containing retroactive restatements and post-audit sanctions updates.
    """
    kg = BiTemporalKnowledgeGraph()
    edges = [
        BiTemporalEdge(
            "ACME_TREASURY",
            "HOLDS_FACILITY",
            "NORDIC_CLEARING_AG",
            1.0,
            valid_from=20250101,
            valid_to=20270101,
            tx_recorded=20250105,
            fact_summary="ACME Treasury holds €140M credit facility via Nordic Clearing AG",
        ),
        BiTemporalEdge(
            "NORDIC_CLEARING_AG",
            "OWNED_BY",
            "HELSINKI_BANCORP",
            1.0,
            valid_from=20240101,
            valid_to=20270101,
            tx_recorded=20240110,
            tx_superseded=20260915,
            fact_summary="Nordic Clearing AG 100% owned by tier-1 compliant Helsinki Bancorp (prior to Sep 2026 retroactive filing)",
        ),
        BiTemporalEdge(
            "HELSINKI_BANCORP",
            "REGULATORY_STATUS",
            "TIER1_APPROVED_COUNTERPARTY",
            1.0,
            valid_from=20240101,
            valid_to=20270101,
            tx_recorded=20240110,
            fact_summary="Helsinki Bancorp certified ECB Tier-1 Approved Counterparty",
        ),
        # Retroactive Beneficial Ownership Restatement recorded in Sep 2026 (tx_recorded=20260915),
        # backdated to valid_from=20260301! An audit at tx_at=20260601 MUST NOT leak this!
        BiTemporalEdge(
            "NORDIC_CLEARING_AG",
            "RETROACTIVE_SHELL_CONTROL",
            "BALTIC_SHADOW_HOLDINGS",
            1.5,
            valid_from=20260301,
            valid_to=20270101,
            tx_recorded=20260915,
            fact_summary="Retroactive Sep-2026 forensic filing: Baltic Shadow Holdings covertly acquired voting stake in Mar-2026",
        ),
        BiTemporalEdge(
            "BALTIC_SHADOW_HOLDINGS",
            "DESIGNATED_ON",
            "OFAC_SDN_WATCHLIST",
            2.0,
            valid_from=20260301,
            valid_to=20270101,
            tx_recorded=20260915,
            fact_summary="Baltic Shadow Holdings placed on OFAC SDN sanctions watchlist in Sep-2026",
        ),
        BiTemporalEdge(
            "ACME_TREASURY",
            "HEDGES_FX_WITH",
            "ZURICH_PRIME_CUSTODY",
            0.8,
            valid_from=20250201,
            valid_to=20270101,
            tx_recorded=20250202,
            fact_summary="ACME Treasury executes CHF/EUR FX hedges via Zurich Prime Custody",
        ),
        BiTemporalEdge(
            "ZURICH_PRIME_CUSTODY",
            "REGULATORY_STATUS",
            "FINMA_AA_VERIFIED",
            1.0,
            valid_from=20250101,
            valid_to=20270101,
            tx_recorded=20250101,
            fact_summary="Zurich Prime Custody verified FINMA AA sovereign custodian",
        ),
    ]
    for e in edges:
        kg.add_edge(e)
    return kg


def run_bitemporal_ppr_benchmark(
    *,
    valid_at: int = 20260501,
    tx_at: int = 20260601,
) -> BiTemporalGraphReport:
    """Benchmark C++20 CSR Personalized PageRank on point-in-time `as_of(valid_at, tx_at)` snapshot
    and prove exact simplex conservation & zero retroactive future-tx leakage.
    """
    kg = build_sovereign_compliance_kg()
    cpp_bridge = get_privora_bridge()
    py_bridge = PrivoraKernelBridge(csrc_path=Path("/tmp/missing_privora_kernel.cpp"))

    ranks_cpp, iters, active = kg.query_ppr_as_of(
        ["ACME_TREASURY"], valid_at=valid_at, tx_at=tx_at, bridge=cpp_bridge
    )
    ranks_py, _, _ = kg.query_ppr_as_of(
        ["ACME_TREASURY"], valid_at=valid_at, tx_at=tx_at, bridge=py_bridge
    )

    max_diff = max(abs(ranks_cpp[k] - ranks_py[k]) for k in ranks_cpp)
    mass_sum = sum(ranks_cpp.values())

    summary_by_dst: dict[str, str] = {e.dst: e.fact_summary for e in active}
    summary_by_dst["ACME_TREASURY"] = "Query Anchor: ACME Corporate Treasury Portfolio"

    sorted_ents = sorted(ranks_cpp.items(), key=lambda kv: kv[1], reverse=True)
    top_facts = [
        BiTemporalRankedFact(
            entity=ent,
            ppr_score=round(sc, 6),
            fact_summary=summary_by_dst.get(ent, "Inactive at point-in-time snapshot"),
        )
        for ent, sc in sorted_ents
        if sc > 0.001
    ][:5]

    return BiTemporalGraphReport(
        kernel_backend=cpp_bridge.backend,
        query_valid_at=valid_at,
        query_tx_at=tx_at,
        total_historical_edges=kg.total_edges,
        active_bitemporal_edges=len(active),
        retroactive_future_edges_blocked=kg.total_edges - len(active),
        ppr_iterations=iters,
        simplex_mass_sum=round(mass_sum, 8),
        cpp_vs_numpy_max_diff=round(max_diff, 12),
        top_ranked_entities=top_facts,
    )
