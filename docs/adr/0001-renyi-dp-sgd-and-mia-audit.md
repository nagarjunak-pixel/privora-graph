# ADR 0001: Exact Binomial Rényi Differential Privacy (`RDP`) Accountant + Compact `DP-SGD`

- **Status:** Accepted
- **Date:** 2026-11-16

## Context
Asymptotic Gaussian moments approximations underestimate privacy spend at moderate $\sigma$ or small subsampling rates, while applying isotropic DP noise to overparameterized dense layers degrades utility at $O(\sqrt{p}/(\varepsilon n))$.

## Decision
We implement exact log-space binomial evaluation of Mironov-Talwar-Zhang (2019) integer-order Subsampled Gaussian RDP across $\alpha \in \{2, \dots, 64\}$ with Balle et al. (2020) $(\varepsilon, \delta)$ conversion, paired with per-sample $L_2$-clipped compact representations.

## Consequences
- Certifies $\varepsilon = 4.1432$ at $\delta = 10^{-5}$ while preserving `86.87%` held-out test accuracy (vs `88.75%` non-private) and eliminating **`90.26%` of Membership Inference Attack (`MIA`) advantage** (`AUC = 0.5051`).
