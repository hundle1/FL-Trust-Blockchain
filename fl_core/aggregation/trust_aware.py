import torch
from typing import Dict, List, Optional
import numpy as np


class TrustAwareAggregator:
    """
    Trust-Aware Federated Aggregation
    Weights client updates by trust scores
    """
    
    def __init__(self, trust_manager, enable_filtering: bool = True):
        """
        Args:
            trust_manager: TrustScoreManager instance
            enable_filtering: Whether to filter clients below threshold
        """
        self.trust_manager = trust_manager
        self.enable_filtering = enable_filtering
    
    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
        metrics: Optional[List[Dict[str, float]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using trust weights
        
        Args:
            updates: List of model updates from clients
            client_ids: List of client IDs corresponding to updates
            metrics: Optional list of client metrics
            
        Returns:
            aggregated_update: Trust-weighted aggregated update
        """
        # Filter clients below trust threshold
        if self.enable_filtering:
            trusted_ids = self.trust_manager.get_trusted_clients(client_ids)
            
            if len(trusted_ids) == 0:
                # No trusted clients → use all with uniform weights
                print("WARNING: No trusted clients, using uniform aggregation")
                return self._uniform_aggregate(updates)
            
            # Filter updates to only trusted clients
            trusted_updates = []
            trusted_client_ids = []
            
            for i, cid in enumerate(client_ids):
                if cid in trusted_ids:
                    trusted_updates.append(updates[i])
                    trusted_client_ids.append(cid)
            
            updates = trusted_updates
            client_ids = trusted_client_ids
        
        # Get trust weights
        trust_weights = self.trust_manager.get_trust_weights(client_ids)
        
        # Aggregate with trust weights
        aggregated_update = {}
        param_names = list(updates[0].keys())
        
        for name in param_names:
            aggregated_param = torch.zeros_like(updates[0][name])
            
            for update, weight in zip(updates, trust_weights):
                aggregated_param += weight * update[name]
            
            aggregated_update[name] = aggregated_param
        
        return aggregated_update
    
    def _uniform_aggregate(self, updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Uniform aggregation (fallback)"""
        aggregated_update = {}
        param_names = list(updates[0].keys())
        n = len(updates)
        
        for name in param_names:
            aggregated_param = torch.zeros_like(updates[0][name])
            for update in updates:
                aggregated_param += update[name] / n
            aggregated_update[name] = aggregated_param
        
        return aggregated_update


class FedAvgAggregator:
    """Standard FedAvg aggregation (baseline)"""
    
    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
        metrics: Optional[List[Dict[str, float]]] = None
    ) -> Dict[str, torch.Tensor]:
        """Uniform averaging"""
        aggregated_update = {}
        param_names = list(updates[0].keys())
        n = len(updates)
        
        for name in param_names:
            aggregated_param = torch.zeros_like(updates[0][name])
            for update in updates:
                aggregated_param += update[name] / n
            aggregated_update[name] = aggregated_param
        
        return aggregated_update


class KrumAggregator:
    """Krum aggregation (baseline defense)"""
    
    def __init__(self, num_malicious: int = 0):
        self.num_malicious = num_malicious
    
    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
        metrics: Optional[List[Dict[str, float]]] = None
    ) -> Dict[str, torch.Tensor]:
        """Select update with smallest distance sum to others"""
        n = len(updates)
        m = self.num_malicious
        
        # Flatten updates for distance computation
        flattened_updates = []
        for update in updates:
            flat = torch.cat([param.flatten() for param in update.values()])
            flattened_updates.append(flat)
        
        # Compute pairwise distances
        scores = []
        for i in range(n):
            distances = []
            for j in range(n):
                if i != j:
                    dist = torch.norm(flattened_updates[i] - flattened_updates[j])
                    distances.append(dist.item())
            
            # Sum of n - m - 2 smallest distances
            distances.sort()
            score = sum(distances[:n - m - 2])
            scores.append(score)
        
        # Select update with minimum score
        selected_idx = np.argmin(scores)
        
        return updates[selected_idx]


class TrimmedMeanAggregator:
    """Trimmed Mean aggregation (baseline defense)"""
    
    def __init__(self, trim_ratio: float = 0.1):
        self.trim_ratio = trim_ratio
    
    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
        metrics: Optional[List[Dict[str, float]]] = None
    ) -> Dict[str, torch.Tensor]:
        """Trimmed mean: remove extreme values"""
        n = len(updates)
        trim_count = int(n * self.trim_ratio)
        
        aggregated_update = {}
        param_names = list(updates[0].keys())
        
        for name in param_names:
            # Stack parameter values from all clients
            param_values = torch.stack([update[name] for update in updates])
            
            # Sort along client dimension and trim
            sorted_values, _ = torch.sort(param_values, dim=0)
            
            if trim_count > 0:
                trimmed = sorted_values[trim_count:-trim_count]
            else:
                trimmed = sorted_values
            
            # Mean of trimmed values
            aggregated_update[name] = torch.mean(trimmed, dim=0)
        
        return aggregated_update