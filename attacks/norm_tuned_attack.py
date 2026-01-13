"""
Norm-Tuned Attack
Attacker tunes gradient norm to match benign clients
Makes attack harder to detect
"""

import torch
import numpy as np
from typing import Dict, Optional


class NormTunedAttack:
    """
    Norm-tuned poisoning: scale attack to match benign gradient norms
    """
    
    def __init__(
        self,
        client_id: int,
        target_norm_ratio: float = 1.5,
        base_poisoning_scale: float = 5.0,
        adaptive: bool = True
    ):
        """
        Args:
            client_id: Attacker's client ID
            target_norm_ratio: Target ratio of attack norm to benign norm
            base_poisoning_scale: Base scale for poisoning
            adaptive: Whether to adapt based on observed norms
        """
        self.client_id = client_id
        self.target_norm_ratio = target_norm_ratio
        self.base_poisoning_scale = base_poisoning_scale
        self.adaptive = adaptive
        
        self.current_scale = base_poisoning_scale
        self.observed_benign_norms = []
        self.attack_norms = []
    
    def compute_gradient_norm(self, gradient: Dict[str, torch.Tensor]) -> float:
        """Compute L2 norm of gradient"""
        total_norm = 0.0
        
        for param in gradient.values():
            total_norm += torch.norm(param).item() ** 2
        
        return np.sqrt(total_norm)
    
    def update_scale(self, benign_norm: float):
        """
        Update poisoning scale based on observed benign norm
        
        Args:
            benign_norm: Observed norm of benign gradients
        """
        if not self.adaptive:
            return
        
        self.observed_benign_norms.append(benign_norm)
        
        # Adapt scale to match target ratio
        # attack_norm = scale * clean_norm
        # We want: attack_norm / benign_norm = target_ratio
        # So: scale * clean_norm / benign_norm = target_ratio
        
        # Estimate scale adjustment
        if len(self.observed_benign_norms) > 5:
            avg_benign_norm = np.mean(self.observed_benign_norms[-5:])
            
            # Adjust scale to achieve target ratio
            # This is simplified - in practice clean_norm ≈ benign_norm
            self.current_scale = self.target_norm_ratio * avg_benign_norm / (avg_benign_norm + 1e-6)
            self.current_scale = max(0.5, min(10.0, self.current_scale))
    
    def poison_gradient(
        self,
        clean_gradient: Dict[str, torch.Tensor],
        reference_norm: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply norm-tuned poisoning
        
        Args:
            clean_gradient: Clean gradient from training
            reference_norm: Reference benign norm (if available)
            
        Returns:
            poisoned_gradient: Poisoned gradient with tuned norm
        """
        # Flip gradient
        poisoned = {}
        for name, param in clean_gradient.items():
            poisoned[name] = -self.current_scale * param
        
        # Compute poisoned norm
        poisoned_norm = self.compute_gradient_norm(poisoned)
        self.attack_norms.append(poisoned_norm)
        
        # If reference norm provided, rescale to match target ratio
        if reference_norm is not None and self.adaptive:
            target_norm = self.target_norm_ratio * reference_norm
            
            if poisoned_norm > 0:
                scale_factor = target_norm / poisoned_norm
                
                # Rescale poisoned gradient
                for name in poisoned.keys():
                    poisoned[name] = poisoned[name] * scale_factor
        
        return poisoned
    
    def get_statistics(self) -> dict:
        """Get attack statistics"""
        return {
            'type': 'norm_tuned',
            'target_norm_ratio': self.target_norm_ratio,
            'current_scale': self.current_scale,
            'avg_observed_benign_norm': np.mean(self.observed_benign_norms) if self.observed_benign_norms else 0,
            'avg_attack_norm': np.mean(self.attack_norms) if self.attack_norms else 0,
            'adaptive': self.adaptive
        }


class StealthyAttack:
    """
    Combination of norm-tuning and intermittent patterns for stealth
    """
    
    def __init__(
        self,
        client_id: int,
        attack_probability: float = 0.3,
        norm_ratio: float = 1.2,
        trust_threshold: float = 0.6
    ):
        """
        Args:
            client_id: Attacker's client ID
            attack_probability: Base probability of attacking
            norm_ratio: Target norm ratio (closer to 1 = more stealthy)
            trust_threshold: Only attack if trust above this
        """
        self.client_id = client_id
        self.attack_probability = attack_probability
        self.norm_ratio = norm_ratio
        self.trust_threshold = trust_threshold
        
        self.current_trust = 1.0
        self.attacks_executed = 0
    
    def should_attack(self, round_num: int, estimated_trust: float) -> bool:
        """
        Strategic decision considering stealth
        
        Args:
            round_num: Current round
            estimated_trust: Current trust estimate
            
        Returns:
            attack: Whether to attack
        """
        self.current_trust = estimated_trust
        
        # Only attack if:
        # 1. Trust is high enough
        # 2. Random chance succeeds
        if estimated_trust < self.trust_threshold:
            return False
        
        # Attack with probability proportional to trust
        adjusted_prob = self.attack_probability * (estimated_trust / 1.0)
        
        return np.random.random() < adjusted_prob
    
    def poison_gradient(
        self,
        clean_gradient: Dict[str, torch.Tensor],
        reference_norm: float
    ) -> Dict[str, torch.Tensor]:
        """Apply stealthy poisoning"""
        self.attacks_executed += 1
        
        # Flip gradient
        poisoned = {}
        for name, param in clean_gradient.items():
            poisoned[name] = -param
        
        # Compute poisoned norm
        poisoned_norm = 0.0
        for param in poisoned.values():
            poisoned_norm += torch.norm(param).item() ** 2
        poisoned_norm = np.sqrt(poisoned_norm)
        
        # Rescale to match target norm ratio
        target_norm = self.norm_ratio * reference_norm
        
        if poisoned_norm > 0:
            scale_factor = target_norm / poisoned_norm
            
            for name in poisoned.keys():
                poisoned[name] = poisoned[name] * scale_factor
        
        return poisoned