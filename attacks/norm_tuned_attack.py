"""
Norm-Tuned Attack
Attacker tunes gradient norm to match benign clients — makes attack harder to detect.
"""

import torch
import numpy as np
from typing import Dict, Optional


class NormTunedAttack:
    """
    Norm-tuned poisoning: scales attack gradient to match benign norms.
    Harder to detect by norm-based defenses.
    """
    
    def __init__(
        self,
        client_id: int,
        target_norm_ratio: float = 1.5,
        base_poisoning_scale: float = 5.0,
        adaptive: bool = True
    ):
        self.client_id = client_id
        self.target_norm_ratio = target_norm_ratio
        self.base_poisoning_scale = base_poisoning_scale
        self.adaptive = adaptive
        
        self.current_scale = base_poisoning_scale
        self.observed_benign_norms = []
        self.attack_norms = []
    
    def compute_gradient_norm(self, gradient: Dict[str, torch.Tensor]) -> float:
        total = sum(torch.norm(p).item() ** 2 for p in gradient.values())
        return float(np.sqrt(total))
    
    def update_scale(self, benign_norm: float):
        """Adapt poisoning scale to observed benign norm."""
        if not self.adaptive:
            return
        self.observed_benign_norms.append(benign_norm)
        if len(self.observed_benign_norms) > 5:
            avg_benign = np.mean(self.observed_benign_norms[-5:])
            self.current_scale = float(np.clip(
                self.target_norm_ratio * avg_benign / max(avg_benign, 1e-6),
                0.5, 10.0
            ))
    
    def poison_gradient(
        self,
        clean_gradient: Dict[str, torch.Tensor],
        reference_norm: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """Apply norm-tuned poisoning."""
        poisoned = {name: -self.current_scale * p for name, p in clean_gradient.items()}
        
        poisoned_norm = self.compute_gradient_norm(poisoned)
        self.attack_norms.append(poisoned_norm)
        
        # Rescale to match target norm ratio
        if reference_norm is not None and self.adaptive and poisoned_norm > 0:
            target_norm = self.target_norm_ratio * reference_norm
            scale_factor = target_norm / poisoned_norm
            poisoned = {name: p * scale_factor for name, p in poisoned.items()}
        
        return poisoned
    
    def get_statistics(self) -> dict:
        return {
            'type': 'norm_tuned',
            'target_norm_ratio': self.target_norm_ratio,
            'current_scale': self.current_scale,
            'avg_observed_benign_norm': float(np.mean(self.observed_benign_norms)) if self.observed_benign_norms else 0.0,
            'avg_attack_norm': float(np.mean(self.attack_norms)) if self.attack_norms else 0.0,
            'adaptive': self.adaptive
        }


class StealthyAttack:
    """
    Stealthy attack: combines norm-tuning with intermittent strategy.
    Only attacks when trust is high enough, with a near-benign norm profile.
    """
    
    def __init__(
        self,
        client_id: int,
        attack_probability: float = 0.3,
        norm_ratio: float = 1.2,
        trust_threshold: float = 0.6
    ):
        self.client_id = client_id
        self.attack_probability = attack_probability
        self.norm_ratio = norm_ratio
        self.trust_threshold = trust_threshold
        
        self.current_trust = 1.0
        self.attacks_executed = 0
    
    def should_attack(self, round_num: int, estimated_trust: float) -> bool:
        self.current_trust = estimated_trust
        if estimated_trust < self.trust_threshold:
            return False
        adjusted_prob = self.attack_probability * estimated_trust
        return bool(np.random.random() < adjusted_prob)
    
    def poison_gradient(
        self,
        clean_gradient: Dict[str, torch.Tensor],
        reference_norm: float
    ) -> Dict[str, torch.Tensor]:
        self.attacks_executed += 1
        
        poisoned = {name: -p for name, p in clean_gradient.items()}
        poisoned_norm = float(np.sqrt(sum(torch.norm(p).item() ** 2 for p in poisoned.values())))
        
        if poisoned_norm > 0:
            target_norm = self.norm_ratio * reference_norm
            scale = target_norm / poisoned_norm
            poisoned = {name: p * scale for name, p in poisoned.items()}
        
        return poisoned