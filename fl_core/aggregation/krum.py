"""
Krum Aggregator - Standalone Module
Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant
Gradient Descent", NeurIPS 2017.

Krum selects the single update (or Multi-Krum: top-m updates)
whose sum of distances to its n-f-2 nearest neighbours is minimal.
This makes it robust to up to f Byzantine clients.
"""

import torch
import numpy as np
from typing import Dict, List, Optional


class KrumAggregator:
    """
    Krum Byzantine-robust aggregator.

    Selects the client update with the smallest sum of squared distances
    to its (n - num_malicious - 2) nearest neighbours.

    For Multi-Krum (m > 1), selects the m best updates and averages them.
    """

    def __init__(self, num_malicious: int = 0, multi_krum_m: Optional[int] = None):
        """
        Args:
            num_malicious: Estimated number of malicious clients (f).
                           Must satisfy: n > 2f + 2.
            multi_krum_m:  If set, use Multi-Krum: average top-m updates.
                           If None, use standard Krum (select single best).
        """
        self.num_malicious = num_malicious
        self.multi_krum_m = multi_krum_m

    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
        metrics: Optional[List[Dict[str, float]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Krum / Multi-Krum aggregation.

        Args:
            updates:    List of model update dicts
            client_ids: Client IDs (unused)
            metrics:    Unused

        Returns:
            aggregated: Selected (or averaged) update
        """
        n = len(updates)
        f = self.num_malicious

        # Need n > 2f + 2
        k = n - f - 2
        if k < 1:
            # Not enough clients for Krum; fallback to average
            return self._uniform_mean(updates)

        # Flatten each update to 1D tensor
        flat = [
            torch.cat([p.flatten() for p in u.values()])
            for u in updates
        ]

        # Compute pairwise squared distances
        dist = torch.zeros(n, n)
        for i in range(n):
            for j in range(i + 1, n):
                d = torch.sum((flat[i] - flat[j]) ** 2).item()
                dist[i, j] = d
                dist[j, i] = d

        # Krum score: sum of k smallest distances for each client
        scores = []
        for i in range(n):
            row = sorted(dist[i].tolist())
            scores.append(sum(row[1:k + 1]))   # skip 0-distance (self)

        if self.multi_krum_m is not None:
            # Multi-Krum: average top-m
            m = min(self.multi_krum_m, n)
            top_m_idx = np.argsort(scores)[:m]
            selected = [updates[i] for i in top_m_idx]
            return self._uniform_mean(selected)
        else:
            # Standard Krum: return single best
            best_idx = int(np.argmin(scores))
            return {name: param.clone() for name, param in updates[best_idx].items()}

    def _uniform_mean(self, updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        n = len(updates)
        return {
            name: sum(u[name] for u in updates) / n
            for name in updates[0].keys()
        }