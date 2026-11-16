"""Unit and Hypothesis property tests for Rényi DP-SGD, MIA AUC, Multi-Krum, and Weiszfeld RFA."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from privora_graph.byzantine.robust_fed import (
    aggregate_fedavg,
    aggregate_geomedian_weiszfeld,
    aggregate_multikrum,
    run_byzantine_federated_benchmark,
)
from privora_graph.privacy.dp_sgd import (
    compute_rdp_epsilon,
    renyi_divergence_subsampled_gaussian,
    run_dp_sgd_vs_mia_benchmark,
)


def test_renyi_accountant_boundary_conditions_and_guards() -> None:
    with pytest.raises(ValueError, match="alpha"):
        renyi_divergence_subsampled_gaussian(1, 0.2, 1.0)
    with pytest.raises(ValueError, match="Subsampling"):
        renyi_divergence_subsampled_gaussian(3, 1.5, 1.0)
    with pytest.raises(ValueError, match="sigma"):
        renyi_divergence_subsampled_gaussian(3, 0.2, 0.0)
    with pytest.raises(ValueError, match="delta"):
        compute_rdp_epsilon(q=0.1, sigma=1.0, steps=10, delta=0.0)

    assert renyi_divergence_subsampled_gaussian(4, 0.0, 1.2) == 0.0
    # Exact Mironov (2017) full-batch Gaussian identity: rho(alpha, q=1, sigma) == alpha / (2 * sigma^2)
    assert abs(renyi_divergence_subsampled_gaussian(6, 1.0, 2.0) - (6.0 / 8.0)) < 1e-12


@settings(max_examples=35, deadline=None)
@given(
    alpha=st.integers(min_value=2, max_value=32),
    q1=st.floats(min_value=0.02, max_value=0.45),
    sigma1=st.floats(min_value=0.6, max_value=2.5),
)
def test_hypothesis_renyi_monotonic_in_subsampling_and_noise(
    alpha: int, q1: float, sigma1: float
) -> None:
    q2 = min(1.0, q1 + 0.20)
    sigma2 = sigma1 + 0.65
    r_base = renyi_divergence_subsampled_gaussian(alpha, q1, sigma1)
    r_more_q = renyi_divergence_subsampled_gaussian(alpha, q2, sigma1)
    r_more_noise = renyi_divergence_subsampled_gaussian(alpha, q1, sigma2)
    assert r_base >= 0.0
    assert r_more_q >= r_base - 1e-12
    assert r_more_noise <= r_base + 1e-12


def test_dp_sgd_eliminates_mia_leakage_while_preserving_utility() -> None:
    rep = run_dp_sgd_vs_mia_benchmark(epochs=8, seed=41)
    assert rep.final_certified_epsilon > 0.0
    assert rep.dp_test_accuracy_pct > 75.0
    assert abs(rep.dp_mia_leakage_auc - 0.5) < 0.08
    assert rep.mia_privacy_risk_reduction_pct > 50.0


def test_byzantine_multikrum_and_weiszfeld_defeat_sign_flipping_attack() -> None:
    with pytest.raises(ValueError, match="2f \\+ 3"):
        aggregate_multikrum(np.zeros((4, 8)), byzantine_f=1)

    rep = run_byzantine_federated_benchmark(total_clients_k=9, byzantine_clients_f=3, seed=53)
    assert rep.fedavg_cosine_to_honest < -0.5
    assert rep.multikrum_cosine_to_honest > 0.99
    assert rep.geomedian_rfa_cosine_to_honest > 0.85
    assert rep.corruption_reduction_x > 8.0

    # Direct check on identical points (early tol break in Weiszfeld)
    const_pts = np.ones((5, 12), dtype=np.float64) * 3.0
    np.testing.assert_allclose(aggregate_fedavg(const_pts), 3.0)
    np.testing.assert_allclose(aggregate_geomedian_weiszfeld(const_pts), 3.0)
