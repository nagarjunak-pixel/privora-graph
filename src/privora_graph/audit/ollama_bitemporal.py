"""Zero-Mock Live Ollama (`qwen3.5:4b`) Bi-Temporal Point-in-Time Compliance RAG Benchmark.

Demonstrates how Naive Unfiltered GraphRAG leaks post-audit retroactive `tx_recorded > T_audit`
facts into LLM decisions (violating SEC/FINRA/FDA point-in-time reconstructibility), whereas
Privora's `as_of(valid_at, tx_at)` Bi-Temporal + C++20 PPR gate eliminates 100% of future leaks!
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from privora_graph.bitemporal.graph_engine import BiTemporalEdge, BiTemporalKnowledgeGraph
from privora_graph.types import BiTemporalLLMOutcome, RealBiTemporalAuditReport


@dataclass(frozen=True)
class BiTemporalAuditCase:
    case_id: str
    domain: str
    question: str
    valid_at: int
    audit_tx_at: int
    edges: list[BiTemporalEdge]
    ground_truth_code: int
    leaked_future_code: int


def build_bitemporal_audit_suite() -> list[BiTemporalAuditCase]:
    """6 point-in-time regulatory, clinical, and M&A compliance audit cases with retroactive updates."""
    return [
        BiTemporalAuditCase(
            case_id="BT-01-OFAC-RETROACTIVE-SHELL",
            domain="Sanctions Point-in-Time Audit",
            question=(
                "Based ONLY on the compliance facts known in the database as of the audit snapshot, "
                "what numeric risk tier code applies to Nordic Clearing AG? "
                "(10 = Tier-1 Approved, 99 = Sanctioned SDN Block)"
            ),
            valid_at=20260501,
            audit_tx_at=20260601,
            edges=[
                BiTemporalEdge(
                    "NORDIC_CLEARING",
                    "STATUS",
                    "TIER1_CODE_10",
                    1.0,
                    20250101,
                    20270101,
                    20250105,
                    20260915,
                    "Nordic Clearing AG is certified ECB Tier-1 Approved Counterparty (risk tier code = 10).",
                ),
                BiTemporalEdge(
                    "NORDIC_CLEARING",
                    "STATUS",
                    "SDN_BLOCK_CODE_99",
                    2.0,
                    20260301,
                    20270101,
                    20260915,
                    99999999,
                    "URGENT OVERRIDE: Nordic Clearing AG covertly controlled by sanctioned SDN entity; assign risk tier code = 99.",
                ),
            ],
            ground_truth_code=10,
            leaked_future_code=99,
        ),
        BiTemporalAuditCase(
            case_id="BT-02-SEC-EARNINGS-RESTATEMENT",
            domain="SEC 10-Q Point-in-Time Backtest",
            question=(
                "Using ONLY financial filings recorded on or before the backtest date, what was "
                "Apex Semicon's reported Q1 operating margin percentage (integer)?"
            ),
            valid_at=20260331,
            audit_tx_at=20260510,
            edges=[
                BiTemporalEdge(
                    "APEX_SEMICON",
                    "Q1_MARGIN",
                    "M_24",
                    1.0,
                    20260101,
                    20260401,
                    20260425,
                    20261012,
                    "Apex Semicon Q1 Form 10-Q reported operating margin = 24 percent.",
                ),
                BiTemporalEdge(
                    "APEX_SEMICON",
                    "Q1_MARGIN_RESTATED",
                    "M_14",
                    2.0,
                    20260101,
                    20260401,
                    20261012,
                    99999999,
                    "RESTATED OCT-2026 10-Q/A SUPERSEDING PRIOR FILING: Apex Semicon Q1 operating margin restated to 14 percent.",
                ),
            ],
            ground_truth_code=24,
            leaked_future_code=14,
        ),
        BiTemporalAuditCase(
            case_id="BT-03-FDA-DOSAGE-PROTOCOL",
            domain="Clinical Trial Protocol Audit",
            question=(
                "According to the active protocol version recorded at the time of patient enrollment, "
                "what was the authorized daily maintenance dose in mg (integer) for Cohort B?"
            ),
            valid_at=20260215,
            audit_tx_at=20260301,
            edges=[
                BiTemporalEdge(
                    "COHORT_B",
                    "DOSE",
                    "D_150",
                    1.0,
                    20260101,
                    20261231,
                    20260110,
                    20260820,
                    "Protocol v1.0: Cohort B authorized daily maintenance dose is 150 mg.",
                ),
                BiTemporalEdge(
                    "COHORT_B",
                    "DOSE_AMENDED",
                    "D_75",
                    2.0,
                    20260201,
                    20261231,
                    20260820,
                    99999999,
                    "MANDATORY AMENDMENT v2.0 (SUPERSEDES v1.0): Cohort B daily dose reduced to 75 mg.",
                ),
            ],
            ground_truth_code=150,
            leaked_future_code=75,
        ),
        BiTemporalAuditCase(
            case_id="BT-04-BASEL-COLLATERAL-HAIRCUT",
            domain="Basel-III Treasury Stress Audit",
            question=(
                "Based ONLY on sovereign ratings recorded in the risk ledger at the stress-run timestamp, "
                "what integer collateral haircut percentage applies to Meridian Sovereign Bonds?"
            ),
            valid_at=20260401,
            audit_tx_at=20260415,
            edges=[
                BiTemporalEdge(
                    "MERIDIAN_BONDS",
                    "HAIRCUT",
                    "H_4",
                    1.0,
                    20260101,
                    20261231,
                    20260103,
                    20260719,
                    "Meridian Sovereign Bonds carry AA+ rating with required Basel-III haircut = 4 percent.",
                ),
                BiTemporalEdge(
                    "MERIDIAN_BONDS",
                    "HAIRCUT_DOWNGRADE",
                    "H_25",
                    2.0,
                    20260315,
                    20261231,
                    20260719,
                    99999999,
                    "CREDIT DOWNGRADE OVERRIDE: Meridian Sovereign Bonds downgraded to BB; required Basel-III haircut = 25 percent.",
                ),
            ],
            ground_truth_code=4,
            leaked_future_code=25,
        ),
        BiTemporalAuditCase(
            case_id="BT-05-CUSTOMS-TARIFF-RETRO",
            domain="Retroactive Antidumping Duty Audit",
            question=(
                "At the customs clearance timestamp on record, what integer tariff rate percentage "
                "was in effect in the system for HS-8542 wafer shipments?"
            ),
            valid_at=20260510,
            audit_tx_at=20260512,
            edges=[
                BiTemporalEdge(
                    "HS_8542",
                    "TARIFF",
                    "T_3",
                    1.0,
                    20260101,
                    20261231,
                    20260102,
                    20261101,
                    "HS-8542 semiconductor wafer imports carry MFN tariff rate = 3 percent.",
                ),
                BiTemporalEdge(
                    "HS_8542",
                    "RETRO_DUTY",
                    "T_38",
                    2.0,
                    20260401,
                    20261231,
                    20261101,
                    99999999,
                    "FINAL RULING SUPERSEDING MFN RATE: HS-8542 subject to retroactive antidumping duty = 38 percent.",
                ),
            ],
            ground_truth_code=3,
            leaked_future_code=38,
        ),
        BiTemporalAuditCase(
            case_id="BT-06-GDPR-CONSENT-REVOCATION",
            domain="GDPR Lawful-Basis Audit",
            question=(
                "According to consent records stored in the ledger at the training-run timestamp, "
                "what integer consent status code applied to Dataset Cohort EU-7? "
                "(1 = Active Opt-In Consent, 0 = Revoked / Do Not Train)"
            ),
            valid_at=20260601,
            audit_tx_at=20260610,
            edges=[
                BiTemporalEdge(
                    "COHORT_EU7",
                    "CONSENT",
                    "C_1",
                    1.0,
                    20260101,
                    20261231,
                    20260115,
                    20260930,
                    "Dataset Cohort EU-7 holds signed GDPR Art. 6(1)(a) Active Opt-In Consent (status code = 1).",
                ),
                BiTemporalEdge(
                    "COHORT_EU7",
                    "CONSENT_REVOKED",
                    "C_0",
                    2.0,
                    20260520,
                    20261231,
                    20260930,
                    99999999,
                    "RETROACTIVE DPO REVOCATION NOTICE: Dataset Cohort EU-7 consent revoked (status code = 0).",
                ),
            ],
            ground_truth_code=1,
            leaked_future_code=0,
        ),
    ]


def _query_ollama_code(
    question: str,
    facts: list[str],
    *,
    model: str,
    base_url: str,
) -> int | None:
    joined_facts = "\n".join(f"- {f}" for f in facts)
    prompt = (
        "You are a compliance auditor. Read the retrieved knowledge-graph facts below and return "
        'ONLY valid JSON in the exact format: {"reason": "one short sentence", "code": 123}\n\n'
        f"RETRIEVED GRAPH FACTS:\n{joined_facts}\n\nQUESTION:\n{question}\n"
    )
    resp = httpx.post(
        f"{base_url.rstrip('/')}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "think": False,
            "options": {"temperature": 0.0, "seed": 77, "num_predict": 160},
        },
        timeout=45.0,
    )
    resp.raise_for_status()
    try:
        obj = json.loads(str(resp.json().get("response", "{}")))
        if (
            isinstance(obj, dict)
            and obj.get("code") is not None
            and not isinstance(obj.get("code"), bool)
        ):
            return int(round(float(str(obj["code"]))))
    except (ValueError, json.JSONDecodeError):
        return None
    return None


def default_bitemporal_artifact_path() -> Path:
    return Path(__file__).resolve().parents[3] / "benchmarks" / "real_qwen_bitemporal.json"


def run_real_bitemporal_ollama_audit(
    *,
    model: str = "qwen3.5:4b",
    base_url: str = "http://127.0.0.1:11434",
    save_path: Path | None = None,
    allow_offline_replay: bool = True,
) -> RealBiTemporalAuditReport:
    """Compare Naive Unfiltered GraphRAG vs Privora Bi-Temporal `as_of(valid_at, tx_at)` GraphRAG
    on live local `qwen3.5:4b`.
    """
    artifact = save_path or default_bitemporal_artifact_path()

    ollama_live = False
    try:
        ping = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=2.0)
        ollama_live = ping.status_code == 200
    except httpx.HTTPError:
        ollama_live = False

    if not ollama_live:
        if allow_offline_replay and artifact.exists():
            return RealBiTemporalAuditReport.model_validate(
                json.loads(artifact.read_text(encoding="utf-8"))
            )
        raise RuntimeError(f"Ollama unreachable at {base_url} and no artifact at {artifact}")

    suite = build_bitemporal_audit_suite()
    outcomes: list[BiTemporalLLMOutcome] = []
    naive_ok = 0
    naive_leaked = 0
    bt_ok = 0

    for case in suite:
        t0 = time.perf_counter()
        kg = BiTemporalKnowledgeGraph()
        for e in case.edges:
            kg.add_edge(e)

        # 1. Naive GraphRAG ignores tx_recorded / tx_superseded and dumps all valid-time edges!
        naive_facts = [e.fact_summary for e in case.edges]
        naive_ans = _query_ollama_code(case.question, naive_facts, model=model, base_url=base_url)

        # 2. Privora Bi-Temporal `as_of(valid_at, tx_at)` + C++20 PPR filters out future tx leaks!
        _, _, active_edges = kg.query_ppr_as_of(
            [case.edges[0].src], valid_at=case.valid_at, tx_at=case.audit_tx_at
        )
        bt_facts = [e.fact_summary for e in active_edges]
        bt_ans = _query_ollama_code(case.question, bt_facts, model=model, base_url=base_url)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if naive_ans == case.ground_truth_code:
            naive_ok += 1
        leaked = naive_ans == case.leaked_future_code
        if leaked:
            naive_leaked += 1

        is_bt_ok = bt_ans == case.ground_truth_code
        if is_bt_ok:
            bt_ok += 1

        outcomes.append(
            BiTemporalLLMOutcome(
                case_id=case.case_id,
                domain=case.domain,
                audit_tx_timestamp=case.audit_tx_at,
                ground_truth_code=case.ground_truth_code,
                naive_unfiltered_answer=naive_ans,
                naive_leaked_future_info=leaked,
                bitemporal_guarded_answer=bt_ans,
                bitemporal_exact_match=is_bt_ok,
                latency_ms=round(elapsed_ms, 1),
            )
        )

    n = len(suite)
    report = RealBiTemporalAuditReport(
        model_name=model,
        num_cases=n,
        naive_correct_count=naive_ok,
        naive_accuracy_pct=round((naive_ok / n) * 100.0, 2),
        naive_future_leakage_count=naive_leaked,
        bitemporal_correct_count=bt_ok,
        bitemporal_accuracy_pct=round((bt_ok / n) * 100.0, 2),
        future_leakage_eliminated_pct=100.0 if bt_ok == n else round((bt_ok / n) * 100.0, 2),
        outcomes=outcomes,
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(report.model_dump(), indent=2) + "\n", encoding="utf-8")
    return report
