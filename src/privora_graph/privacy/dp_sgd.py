"""Abadi et al. (CCS 2016) Per-Sample Clipped `DP-SGD`, Exact Mironov-Talwar-Zhang (2019)
Subsampled Gaussian Rényi Differential Privacy (`RDP`) Accountant, and Yeom/Carlini Membership
Inference Attack (`MIA`) Empirical Audit.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from privora_graph.types import DPEpochTelemetry, DPSGDPrivacyReport


def renyi_divergence_subsampled_gaussian(alpha: int, q: float, sigma: float) -> float:
    """Exact integer-order Rényi divergence `D_alpha(M(D) || M(D'))` of the Poisson-subsampled
    Gaussian mechanism (Mironov, Talwar & Zhang, 2019):
      rho(alpha) = (1 / (alpha - 1)) * log( sum_{k=0}^alpha C(alpha, k) (1-q)^(alpha-k) q^k exp(k(k-1)/(2*sigma^2)) )
    Computed in log-space (`log-sum-exp`) to prevent overflow at large orders `alpha <= 64`.
    """
    if alpha < 2:
        raise ValueError("Integer Rényi order alpha must be >= 2")
    if not (0.0 <= q <= 1.0):
        raise ValueError("Subsampling rate q must be in [0, 1]")
    if sigma <= 0.0:
        raise ValueError("Noise multiplier sigma must be > 0")
    if q == 0.0:
        return 0.0
    if q == 1.0:
        return float(alpha) / (2.0 * sigma * sigma)

    log_terms: list[float] = []
    log_q = math.log(q)
    log_1mq = math.log(1.0 - q)
    for k in range(alpha + 1):
        log_binom = math.lgamma(alpha + 1) - math.lgamma(k + 1) - math.lgamma(alpha - k + 1)
        quad = (k * (k - 1)) / (2.0 * sigma * sigma)
        log_terms.append(log_binom + (alpha - k) * log_1mq + k * log_q + quad)

    m_max = max(log_terms)
    lse = m_max + math.log(sum(math.exp(v - m_max) for v in log_terms))
    return max(0.0, lse / float(alpha - 1))


def compute_rdp_epsilon(
    *,
    q: float,
    sigma: float,
    steps: int,
    delta: float = 1e-5,
) -> tuple[float, float]:
    """Convert `steps` compositions of RDP to tight `(epsilon, delta)`-DP (Balle et al., AISTATS 2020)."""
    if delta <= 0.0 or delta >= 1.0:
        raise ValueError("delta must lie in (0, 1)")
    best_eps = float("inf")
    best_alpha = 2.0
    for alpha in range(2, 65):
        rdp_total = float(steps) * renyi_divergence_subsampled_gaussian(alpha, q, sigma)
        # Tight Balle et al. conversion bound
        conv = (
            math.log(1.0 / delta) + (alpha - 1) * math.log(1.0 - 1.0 / alpha) - math.log(alpha)
        ) / float(alpha - 1)
        eps_cand = max(0.0, rdp_total + conv)
        if eps_cand < best_eps:
            best_eps = eps_cand
            best_alpha = float(alpha)
    return round(best_eps, 4), best_alpha


def _roc_auc_score(member_scores: np.ndarray, non_member_scores: np.ndarray) -> float:
    """Exact Mann-Whitney U statistic ROC-AUC for Membership Inference Attack (`MIA`)."""
    diffs = member_scores[:, None] - non_member_scores[None, :]
    wins = float(np.sum(diffs > 0.0)) + 0.5 * float(np.sum(diffs == 0.0))
    total = float(member_scores.size * non_member_scores.size)
    return wins / max(total, 1.0)


class _SensitiveRiskMLP(nn.Module):
    def __init__(self, in_dim: int = 16, hidden: int = 24) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.fc1(x))
        out: torch.Tensor = self.fc2(h)
        return out


class _OverparameterizedMemorizerMLP(nn.Module):
    """High-capacity unregularized baseline illustrating training-record memorization & MIA vulnerability."""

    def __init__(self, in_dim: int = 16, hidden: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1(x))
        h = F.gelu(self.fc2(h))
        out: torch.Tensor = self.fc3(h)
        return out


def _dp_sgd_step(
    model: nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    *,
    clip_c: float,
    sigma: float,
    lr: float,
) -> None:
    """Exact Abadi et al. per-sample L2 gradient clipping + calibrated Gaussian noise injection."""
    bsz = x_batch.shape[0]
    params = [p for p in model.parameters() if p.requires_grad]
    sum_clipped = [torch.zeros_like(p) for p in params]

    for i in range(bsz):
        model.zero_grad(set_to_none=True)
        logits_i = model(x_batch[i : i + 1])
        loss_i = F.cross_entropy(logits_i, y_batch[i : i + 1])
        grads_i = torch.autograd.grad(loss_i, params)

        sq_norm = sum(float(g.pow(2).sum().item()) for g in grads_i)
        grad_norm = math.sqrt(max(sq_norm, 1e-16))
        clip_factor = min(1.0, clip_c / grad_norm)

        for acc, g in zip(sum_clipped, grads_i, strict=True):
            acc.add_(g * clip_factor)

    with torch.no_grad():
        noise_std = sigma * clip_c
        for p, acc in zip(params, sum_clipped, strict=True):
            noisy_grad = (acc + torch.randn_like(acc) * noise_std) / float(bsz)
            p.sub_(lr * noisy_grad)


@torch.no_grad()
def _eval_acc_and_losses(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> tuple[float, np.ndarray]:
    logits = model(x)
    preds = logits.argmax(dim=-1)
    acc = float((preds == y).float().mean().item()) * 100.0
    losses = F.cross_entropy(logits, y, reduction="none").cpu().numpy()
    return round(acc, 2), losses


def run_dp_sgd_vs_mia_benchmark(
    *,
    num_records: int = 320,
    in_dim: int = 16,
    clip_c: float = 1.0,
    sigma: float = 1.45,
    epochs: int = 10,
    batch_size: int = 32,
    delta: float = 1e-5,
    seed: int = 41,
) -> DPSGDPrivacyReport:
    """Train both an unclipped non-private MLP and a per-sample clipped DP-SGD MLP on sensitive
    records, track exact Rényi `(epsilon, delta)` privacy spend, and audit Yeom/Carlini MIA AUC.
    """
    torch.manual_seed(seed)
    w_true = torch.randn(in_dim, 1) * 0.95
    x_train = torch.randn(num_records, in_dim)
    logits_train = (x_train @ w_true).squeeze(-1) + torch.randn(num_records) * 0.85
    y_train = (logits_train > 0.0).long()

    x_test = torch.randn(num_records, in_dim)
    logits_test = (x_test @ w_true).squeeze(-1) + torch.randn(num_records) * 0.85
    y_test = (logits_test > 0.0).long()

    dp_model = _SensitiveRiskMLP(in_dim=in_dim)
    raw_model = _OverparameterizedMemorizerMLP(in_dim=in_dim)

    raw_opt = torch.optim.AdamW(raw_model.parameters(), lr=6e-3, weight_decay=0.0)

    q_rate = batch_size / float(num_records)
    steps_per_epoch = max(1, num_records // batch_size)
    history: list[DPEpochTelemetry] = []
    total_steps = 0

    for ep in range(1, epochs + 1):
        perm = torch.randperm(num_records)
        for s in range(steps_per_epoch):
            idx = perm[s * batch_size : (s + 1) * batch_size]
            xb, yb = x_train[idx], y_train[idx]

            _dp_sgd_step(dp_model, xb, yb, clip_c=clip_c, sigma=sigma, lr=0.25)
            total_steps += 1

            for _ in range(4):
                raw_opt.zero_grad()
                loss_r = F.cross_entropy(raw_model(xb), yb)
                torch.autograd.backward(loss_r)
                raw_opt.step()

        eps_ep, _ = compute_rdp_epsilon(q=q_rate, sigma=sigma, steps=total_steps, delta=delta)
        dp_acc, _ = _eval_acc_and_losses(dp_model, x_test, y_test)
        raw_acc, _ = _eval_acc_and_losses(raw_model, x_test, y_test)
        history.append(
            DPEpochTelemetry(
                epoch=ep,
                epsilon_spent=eps_ep,
                dp_test_accuracy_pct=dp_acc,
                non_private_test_accuracy_pct=raw_acc,
            )
        )

    final_eps, best_alpha = compute_rdp_epsilon(
        q=q_rate, sigma=sigma, steps=total_steps, delta=delta
    )
    dp_test_acc, dp_nonmember_loss = _eval_acc_and_losses(dp_model, x_test, y_test)
    _, dp_member_loss = _eval_acc_and_losses(dp_model, x_train, y_train)

    raw_test_acc, raw_nonmember_loss = _eval_acc_and_losses(raw_model, x_test, y_test)
    _, raw_member_loss = _eval_acc_and_losses(raw_model, x_train, y_train)

    # Attacker predicts membership using higher confidence / lower loss: score = -loss
    raw_mia_auc = _roc_auc_score(-raw_member_loss, -raw_nonmember_loss)
    dp_mia_auc = _roc_auc_score(-dp_member_loss, -dp_nonmember_loss)

    raw_adv = max(0.0, raw_mia_auc - 0.5)
    dp_adv = max(0.0, abs(dp_mia_auc - 0.5))
    risk_red = ((raw_adv - dp_adv) / max(raw_adv, 1e-6)) * 100.0

    return DPSGDPrivacyReport(
        num_records=num_records,
        clipping_norm_c=clip_c,
        noise_multiplier_sigma=sigma,
        target_delta=delta,
        optimal_renyi_alpha=best_alpha,
        final_certified_epsilon=final_eps,
        dp_test_accuracy_pct=dp_test_acc,
        non_private_test_accuracy_pct=raw_test_acc,
        non_private_mia_leakage_auc=round(raw_mia_auc, 4),
        dp_mia_leakage_auc=round(dp_mia_auc, 4),
        mia_privacy_risk_reduction_pct=round(max(0.0, risk_red), 2),
        history=history,
    )
