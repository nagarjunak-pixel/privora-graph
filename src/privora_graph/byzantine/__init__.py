"""Byzantine-Resilient Federated Aggregation (`Multi-Krum` & Smoothed Weiszfeld `RFA`)."""

from privora_graph.byzantine.robust_fed import (
    aggregate_fedavg,
    aggregate_geomedian_weiszfeld,
    aggregate_multikrum,
    run_byzantine_federated_benchmark,
)

__all__ = [
    "aggregate_fedavg",
    "aggregate_geomedian_weiszfeld",
    "aggregate_multikrum",
    "run_byzantine_federated_benchmark",
]
