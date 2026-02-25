"""
FedAvg Aggregator - Standalone Module
McMahan et al., "Communication-Efficient Learning of Deep Networks from
Decentralized Data", AISTATS 2017.

Note: FedAvgAggregator is also available in trust_aware.py.
This standalone module provides the same implementation for use
without importing the full trust_aware module.
"""

import torch
from typing import Dict, List, Optional


class FedAvgAggregator:
    """
    Standard Federated Averaging (FedAvg) aggregator.
    Simple uniform average of all client updates.
    Used as the primary baseline in all experiments.
    """

    def __init__(self, sample_weighted: bool = False):
        """
        Args:
            sample_weighted: If True, weight by number of local samples.
                             Requires 'num_samples' key in metrics.
                             If False, uniform weighting (default).
        """
        self.sample_weighted = sample_weighted

    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
        metrics: Optional[List[Dict[str, float]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate updates via uniform (or sample-weighted) averaging.

        Args:
            updates:    List of model update dicts (delta params)
            client_ids: Corresponding client IDs (unused in FedAvg)
            metrics:    Optional list of training metrics per client

        Returns:
            aggregated: Averaged model update
        """
        n = len(updates)
        assert n > 0, "Need at least one update"

        # Compute weights
        if self.sample_weighted and metrics is not None:
            samples = [m.get('num_samples', 1) for m in metrics]
            total = sum(samples)
            weights = [s / total for s in samples]
        else:
            weights = [1.0 / n] * n

        # Weighted average
        aggregated = {}
        for name in updates[0].keys():
            aggregated[name] = sum(
                w * u[name] for w, u in zip(weights, updates)
            )

        return aggregated