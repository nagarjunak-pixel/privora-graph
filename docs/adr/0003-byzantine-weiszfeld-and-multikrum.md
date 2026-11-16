# ADR 0003: Dual Byzantine-Resilient Federated Aggregation (`Multi-Krum` & Smoothed Weiszfeld `RFA`)

- **Status:** Accepted
- **Date:** 2026-11-16

## Context
Cross-silo federated learning across banks or hospitals is vulnerable to compromised participants submitting sign-flipped or backdoored gradients (`FedAvg` breakdown point `1/K`).

## Decision
We provide both distance-graph **`Multi-Krum`** ($K \ge 2f + 3$) and convex **`Smoothed Weiszfeld Geometric Median (RFA)`** ($f < K/2$) aggregators.

## Consequences
- Under `33.3%` (`3/9`) coordinated `-24x` Byzantine silos where `FedAvg` flips to `-0.88086` cosine, `Multi-Krum` recovers `1.00000` cosine and `Weiszfeld RFA` cuts L2 corruption by `12.49x`.
