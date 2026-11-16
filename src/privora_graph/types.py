"""Typed domain models for PrivoraGraph DP-SGD, Bi-Temporal PPR, Byzantine Fed, and Ollama Audit."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DPEpochTelemetry(BaseModel):
    model_config = ConfigDict(frozen=True)

    epoch: int
    epsilon_spent: float = Field(ge=0.0)
    dp_test_accuracy_pct: float
    non_private_test_accuracy_pct: float


class DPSGDPrivacyReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    num_records: int
    clipping_norm_c: float
    noise_multiplier_sigma: float
    target_delta: float
    optimal_renyi_alpha: float
    final_certified_epsilon: float
    dp_test_accuracy_pct: float
    non_private_test_accuracy_pct: float
    non_private_mia_leakage_auc: float
    dp_mia_leakage_auc: float
    mia_privacy_risk_reduction_pct: float
    history: list[DPEpochTelemetry]


class BiTemporalRankedFact(BaseModel):
    model_config = ConfigDict(frozen=True)

    entity: str
    ppr_score: float
    fact_summary: str


class BiTemporalGraphReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    kernel_backend: str
    query_valid_at: int
    query_tx_at: int
    total_historical_edges: int
    active_bitemporal_edges: int
    retroactive_future_edges_blocked: int
    ppr_iterations: int
    simplex_mass_sum: float
    cpp_vs_numpy_max_diff: float
    top_ranked_entities: list[BiTemporalRankedFact]


class ByzantineAggregationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_clients_k: int
    byzantine_clients_f: int
    poison_magnitude_x: float
    fedavg_cosine_to_honest: float
    multikrum_cosine_to_honest: float
    geomedian_rfa_cosine_to_honest: float
    fedavg_l2_corruption: float
    geomedian_rfa_l2_corruption: float
    corruption_reduction_x: float


class BiTemporalLLMOutcome(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    domain: str
    audit_tx_timestamp: int
    ground_truth_code: int
    naive_unfiltered_answer: int | None
    naive_leaked_future_info: bool
    bitemporal_guarded_answer: int | None
    bitemporal_exact_match: bool
    latency_ms: float


class RealBiTemporalAuditReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str
    num_cases: int
    naive_correct_count: int
    naive_accuracy_pct: float
    naive_future_leakage_count: int
    bitemporal_correct_count: int
    bitemporal_accuracy_pct: float
    future_leakage_eliminated_pct: float
    outcomes: list[BiTemporalLLMOutcome]
