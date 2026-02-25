import torch
from typing import Dict, List, Optional
import numpy as np


class TrustAwareAggregator:
    """
    Trust-Aware Federated Aggregation.
    Weights client updates by trust scores and filters below threshold.
    """
    
    def __init__(self, trust_manager, enable_filtering: bool = True):
        self.trust_manager = trust_manager
        self.enable_filtering = enable_filtering
    
    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
        metrics: Optional[List[Dict[str, float]]] = None
    ) -> Dict[str, torch.Tensor]:
        """Trust-weighted aggregation with optional client filtering."""
        if self.enable_filtering:
            trusted_ids = self.trust_manager.get_trusted_clients(client_ids)
            
            if len(trusted_ids) == 0:
                print("WARNING: No trusted clients, using uniform aggregation")
                return self._uniform_aggregate(updates)
            
            # Filter to trusted clients only
            trusted_updates, trusted_client_ids = [], []
            for i, cid in enumerate(client_ids):
                if cid in trusted_ids:
                    trusted_updates.append(updates[i])
                    trusted_client_ids.append(cid)
            
            updates = trusted_updates
            client_ids = trusted_client_ids
        
        trust_weights = self.trust_manager.get_trust_weights(client_ids)
        
        aggregated = {}
        for name in updates[0].keys():
            agg_param = torch.zeros_like(updates[0][name])
            for update, weight in zip(updates, trust_weights):
                agg_param += weight * update[name]
            aggregated[name] = agg_param
        
        return aggregated
    
    def _uniform_aggregate(self, updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        n = len(updates)
        aggregated = {}
        for name in updates[0].keys():
            agg = torch.zeros_like(updates[0][name])
            for u in updates:
                agg += u[name] / n
            aggregated[name] = agg
        return aggregated


class FedAvgAggregator:
    """Standard FedAvg - uniform averaging (baseline)."""
    
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
    Krum aggregation - Byzantine-robust baseline.
    Selects the update with smallest sum of distances to its nearest neighbors.
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
        
        # Flatten updates
        flat = [torch.cat([p.flatten() for p in u.values()]) for u in updates]
        
        scores = []
        for i in range(n):
            dists = sorted(
                torch.norm(flat[i] - flat[j]).item()
                for j in range(n) if i != j
            )
            # Sum n - m - 2 smallest distances
            score = sum(dists[:max(1, n - m - 2)])
            scores.append(score)
        
        selected_idx = int(np.argmin(scores))
        return updates[selected_idx]


class TrimmedMeanAggregator:
    """
    Trimmed Mean aggregation - removes extreme values per coordinate.
    """
    
    def __init__(self, trim_ratio: float = 0.1):
        self.trim_ratio = trim_ratio
    
    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
        metrics: Optional[List[Dict[str, float]]] = None
    ) -> Dict[str, torch.Tensor]:
        n = len(updates)
        trim_count = int(n * self.trim_ratio)
        
        aggregated = {}
        for name in updates[0].keys():
            stacked = torch.stack([u[name] for u in updates])  # (n, ...)
            sorted_vals, _ = torch.sort(stacked, dim=0)
            
            if trim_count > 0 and 2 * trim_count < n:
                trimmed = sorted_vals[trim_count:-trim_count]
            else:
                trimmed = sorted_vals
            
            aggregated[name] = torch.mean(trimmed, dim=0)
        
        return aggregated