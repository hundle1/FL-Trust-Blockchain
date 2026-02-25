"""
Trimmed Mean Aggregator - Standalone Module
Yin et al., "Byzantine-Robust Distributed Learning: Towards Optimal
Statistical Rates", ICML 2018.

For each parameter coordinate, sorts the values across clients,
removes the top and bottom β fraction, then averages the rest.
"""

import torch
from typing import Dict, List, Optional


class TrimmedMeanAggregator:
    """
    Coordinate-wise Trimmed Mean aggregator.

    For each parameter dimension, removes the trim_ratio fraction of
    smallest and largest values, then computes the mean of the remainder.

    Robust to up to trim_ratio fraction of Byzantine clients.
    """

    def __init__(self, trim_ratio: float = 0.1):
        """
        Args:
            trim_ratio: Fraction to trim from each end [0, 0.5).
                        E.g., 0.1 removes 10% from top and 10% from bottom.
                        Requires: num_clients > 2 * ceil(trim_ratio * num_clients)
        """
        assert 0.0 <= trim_ratio < 0.5, "trim_ratio must be in [0, 0.5)"
        self.trim_ratio = trim_ratio

    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
        metrics: Optional[List[Dict[str, float]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Coordinate-wise trimmed mean.

        Args:
            updates:    List of model update dicts
            client_ids: Client IDs (unused)
            metrics:    Unused

        Returns:
            aggregated: Trimmed mean of all updates
        """
        n = len(updates)
        trim_count = int(n * self.trim_ratio)

        aggregated = {}
        for name in updates[0].keys():
            # Stack: shape (n, *param_shape)
            stacked = torch.stack([u[name] for u in updates], dim=0)

            if trim_count > 0 and n - 2 * trim_count > 0:
                # Sort along client dimension (dim=0)
                sorted_vals, _ = torch.sort(stacked, dim=0)
                # Trim both ends
                trimmed = sorted_vals[trim_count: n - trim_count]
            else:
                trimmed = stacked

            aggregated[name] = trimmed.mean(dim=0)

        return aggregated