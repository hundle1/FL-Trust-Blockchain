"""
Trust-Aware Aggregation — FIXED VERSION
=========================================
FIX 6: Fallback khi no trusted clients
  Bản gốc: return self._uniform_aggregate(updates)
           → dùng ALL updates kể cả poisoned → nguy hiểm!
  Bản fix:  return None (signal cho caller skip update round này)
  Caller (exp script) giữ nguyên global model, không apply update.

FIX 7: Trust weight normalization
  Nếu tất cả trusted clients có trust gần bằng nhau → weight đều
  Nếu có client trust cao hơn hẳn → được weight nhiều hơn đúng nghĩa
"""

import torch
from typing import Dict, List, Optional
import numpy as np


class TrustAwareAggregator:
    """
    Trust-Aware Federated Aggregation.
    Weights client updates by trust scores and filters below threshold.

    FIX: Khi no trusted clients → trả về None thay vì dùng poisoned updates.
    """

    def __init__(self, trust_manager, enable_filtering: bool = True):
        self.trust_manager    = trust_manager
        self.enable_filtering = enable_filtering
        self.skipped_rounds   = 0   # đếm số round bị skip
        self.filtered_counts  = []  # lịch sử số clients bị filter mỗi round

    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
        metrics: Optional[List[Dict[str, float]]] = None
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Trust-weighted aggregation with client filtering.

        Returns:
            aggregated update dict, hoặc None nếu không có trusted client
            (caller nên skip apply update round này)
        """
        if self.enable_filtering:
            trusted_ids = self.trust_manager.get_trusted_clients(client_ids)
            self.filtered_counts.append(len(client_ids) - len(trusted_ids))

            if len(trusted_ids) == 0:
                # FIX 6: không dùng poisoned updates — skip round
                self.skipped_rounds += 1
                print(f"  [TrustAware] WARNING: No trusted clients this round — "
                      f"skipping update (total skipped: {self.skipped_rounds})")
                return None

            # Filter
            trusted_updates    = []
            trusted_client_ids = []
            for i, cid in enumerate(client_ids):
                if cid in trusted_ids:
                    trusted_updates.append(updates[i])
                    trusted_client_ids.append(cid)

            updates    = trusted_updates
            client_ids = trusted_client_ids

        # Trust-weighted sum
        trust_weights = self.trust_manager.get_trust_weights(client_ids)

        aggregated = {}
        for name in updates[0].keys():
            agg_param = torch.zeros_like(updates[0][name])
            for update, weight in zip(updates, trust_weights):
                agg_param += weight * update[name]
            aggregated[name] = agg_param

        return aggregated

    def _uniform_aggregate(
        self, updates: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        n = len(updates)
        return {
            name: sum(u[name] for u in updates) / n
            for name in updates[0].keys()
        }

    def get_stats(self) -> Dict:
        return {
            'skipped_rounds':   self.skipped_rounds,
            'avg_filtered':     float(np.mean(self.filtered_counts))
                                if self.filtered_counts else 0.0,
            'total_rounds':     len(self.filtered_counts),
        }


class FedAvgAggregator:
    """Standard FedAvg — uniform average (baseline)."""

    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
        metrics: Optional[List[Dict[str, float]]] = None
    ) -> Dict[str, torch.Tensor]:
        n = len(updates)
        aggregated = {}
        for name in updates[0].keys():
            agg = torch.zeros_like(updates[0][name])
            for u in updates:
                agg += u[name] / n
            aggregated[name] = agg
        return aggregated


class KrumAggregator:
    """
    Krum — Byzantine-robust baseline.
    """

    def __init__(self, num_malicious: int = 0):
        self.num_malicious = num_malicious

    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
        metrics: Optional[List[Dict[str, float]]] = None
    ) -> Dict[str, torch.Tensor]:
        n = len(updates)
        m = self.num_malicious
        flat = [torch.cat([p.flatten() for p in u.values()]) for u in updates]

        scores = []
        for i in range(n):
            dists = sorted(
                torch.norm(flat[i] - flat[j]).item()
                for j in range(n) if i != j
            )
            scores.append(sum(dists[:max(1, n - m - 2)]))

        selected_idx = int(np.argmin(scores))
        return updates[selected_idx]


class TrimmedMeanAggregator:
    """
    Trimmed Mean — removes extreme values per coordinate.
    """

    def __init__(self, trim_ratio: float = 0.1):
        self.trim_ratio = trim_ratio

    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
        metrics: Optional[List[Dict[str, float]]] = None
    ) -> Dict[str, torch.Tensor]:
        n           = len(updates)
        trim_count  = int(n * self.trim_ratio)
        aggregated  = {}
        for name in updates[0].keys():
            stacked     = torch.stack([u[name] for u in updates])
            sorted_vals, _ = torch.sort(stacked, dim=0)
            if trim_count > 0 and 2 * trim_count < n:
                trimmed = sorted_vals[trim_count:-trim_count]
            else:
                trimmed = sorted_vals
            aggregated[name] = torch.mean(trimmed, dim=0)
        return aggregated