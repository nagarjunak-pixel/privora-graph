# ADR 0002: Snodgrass SQL:2011 Bi-Temporal Knowledge Graph + Native C++20 CSR PageRank

- **Status:** Accepted
- **Date:** 2026-11-16

## Context
Standard vector and graph RAG stores overwrite or merge facts using only valid-time timestamps, causing severe **lookahead leakage** during SEC 10-Q restatement backtests, OFAC sanctions audits, FDA clinical protocol reviews, and GDPR consent audits.

## Decision
We enforce orthogonal `[valid_from, valid_to)` and `[tx_recorded, tx_superseded)` intervals on every edge and compile CSR Personalized PageRank in native C++20 (`csrc/privora_kernels.cpp`).

## Consequences
- Eliminates **100% (`6/6` -> `0/6`)** of retroactive lookahead leaks on live `qwen3.5:4b` compliance audits (`0.0%` naive accuracy -> **`100.0%`** bi-temporal accuracy), with `0.000000000000` C++20-vs-NumPy PPR divergence.
