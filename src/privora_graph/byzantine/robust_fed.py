"""Byzantine-Resilient Sovereign Federated Aggregation:
  1) Blanchard et al. (NeurIPS 2017) `Multi-Krum`
  2) Pillutla, Kakade & Harchaoui (IEEE TSP 2022) Smoothed Weiszfeld Geometric Median (`RFA`)
vs. Naive `FedAvg` under `f < K / 2` Malicious Sign-Flipping / Backdoor Silos.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from privora_graph.types import ByzantineAggregationReport

F64Array = npt.NDArray[np.float64]


def aggregate_fedavg(updates: F64Array) -> F64Array:
    """Standard McMahan et al. vulnerable arithmetic mean (`breakdown point = 1/K`)."""
    res: F64Array = np.mean(updates, axis=0)
    return res


def aggregate_multikrum(
    updates: F64Array, byzantine_f: int, select_m: int | None = None
) -> F64Array:
    """Blanchard et al. (NeurIPS 2017) Multi-Krum (`K >= 2f + 3`).

    Scores each client update `i` by the sum of squared Euclidean distances to its `K - f - 2`
    closest peers, and averages the `m` lowest-scoring updates.
    """
    k_total = int(updates.shape[0])
    if k_total < 2 * byzantine_f + 3:
        raise ValueError(f"Multi-Krum requires K >= 2f + 3 (got K={k_total}, f={byzantine_f})")
    neighbors = k_total - byzantine_f - 2
    m = select_m if select_m is not None else (k_total - byzantine_f)

    diffs = updates[:, None, :] - updates[None, :, :]
    sq_dists = np.sum(diffs * diffs, axis=-1)

    krum_scores = np.empty(k_total, dtype=np.float64)
    for i in range(k_total):
        row = np.delete(sq_dists[i], i)
        row_sorted = np.sort(row)
        krum_scores[i] = float(np.sum(row_sorted[:neighbors]))

    chosen = np.argsort(krum_scores)[:m]
    res: F64Array = np.mean(updates[chosen], axis=0)
    return res


def aggregate_geomedian_weiszfeld(
    updates: F64Array,
    *,
    nu_smooth: float = 1e-6,
    max_iters: int = 80,
    tol: float = 1e-8,
) -> F64Array:
    """Pillutla et al. (IEEE TSP 2022) Smoothed Weiszfeld Geometric Median (`RFA`).

    Minimizes `sum_{i=1}^K sqrt(||v - u_i||_2^2 + nu^2)` with breakdown point `0.5` (`f < K/2`),
    monotonically decreasing the convex Fermat-Weber objective at every iteration!
    """
    est = np.median(updates, axis=0).astype(np.float64)
    for _ in range(max_iters):
        dists = np.sqrt(np.sum((updates - est[None, :]) ** 2, axis=-1) + nu_smooth * nu_smooth)
        inv_w = 1.0 / np.maximum(dists, 1e-12)
        weights = inv_w / np.sum(inv_w)
        nxt = np.sum(updates * weights[:, None], axis=0)
        if float(np.linalg.norm(nxt - est)) <= tol:
            est = nxt
            break
        est = nxt
    return est


def _cosine(a: F64Array, b: F64Array) -> float:
    na = max(float(np.linalg.norm(a)), 1e-12)
    nb = max(float(np.linalg.norm(b)), 1e-12)
    return float(np.dot(a, b) / (na * nb))


def run_byzantine_federated_benchmark(
    *,
    total_clients_k: int = 9,
    byzantine_clients_f: int = 3,
    dim: int = 256,
    poison_scale: float = 24.0,
    seed: int = 53,
) -> ByzantineAggregationReport:
    """Simulate `K=9` sovereign hospital/banking silos with `f=3` (`33.3%`) coordinated Byzantine
    sign-flipping adversaries (`-24x * true_grad`), comparing FedAvg, Multi-Krum, and Weiszfeld RFA.
    """
    rng = np.random.default_rng(seed)
    true_grad = rng.normal(0.0, 1.0, size=dim)
    true_grad /= np.linalg.norm(true_grad)

    honest_count = total_clients_k - byzantine_clients_f
    honest_updates = true_grad[None, :] + rng.normal(0.0, 0.08, size=(honest_count, dim))
    honest_mean = np.mean(honest_updates, axis=0)

    byz_updates = -poison_scale * true_grad[None, :] + rng.normal(
        0.0, 0.05, size=(byzantine_clients_f, dim)
    )
    all_updates = np.vstack([honest_updates, byz_updates])

    agg_fedavg = aggregate_fedavg(all_updates)
    agg_krum = aggregate_multikrum(all_updates, byzantine_f=byzantine_clients_f)
    agg_rfa = aggregate_geomedian_weiszfeld(all_updates)

    l2_fed = float(np.linalg.norm(agg_fedavg - honest_mean))
    l2_rfa = float(np.linalg.norm(agg_rfa - honest_mean))

    return ByzantineAggregationReport(
        total_clients_k=total_clients_k,
        byzantine_clients_f=byzantine_clients_f,
        poison_magnitude_x=poison_scale,
        fedavg_cosine_to_honest=round(_cosine(agg_fedavg, honest_mean), 5),
        multikrum_cosine_to_honest=round(_cosine(agg_krum, honest_mean), 5),
        geomedian_rfa_cosine_to_honest=round(_cosine(agg_rfa, honest_mean), 5),
        fedavg_l2_corruption=round(l2_fed, 4),
        geomedian_rfa_l2_corruption=round(l2_rfa, 4),
        corruption_reduction_x=round(l2_fed / max(l2_rfa, 1e-9), 2),
    )
