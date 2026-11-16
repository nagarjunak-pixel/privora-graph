# Security & Privacy Policy

## Architectural Controls in `privora-graph`

1. **Mathematically Certified Per-Example Sensitivity (`DP-SGD`):**
   Every individual sample gradient is bounded strictly to $L_2 \le C$ prior to aggregation, and cumulative $(\varepsilon, \delta)$ privacy loss is accounted via exact log-space Rényi divergence without asymptotic approximations.
2. **Immutable Bi-Temporal Audit Trail:**
   Historical graph assertions are never mutated or deleted in place; superseding facts close `tx_superseded` and append a new `tx_recorded` edge, guaranteeing point-in-time forensic reproducibility.
3. **Byzantine Gradient Filtering:**
   `Multi-Krum` and `Smoothed Weiszfeld RFA` bound adversarial silo influence up to breakdown point $f < K/2$.

## Reporting a Vulnerability
Please report security or privacy issues responsibly to `sendemailtonagarjuna@gmail.com`.
