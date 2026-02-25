"""
Trust Decay Module
Implements various trust decay strategies for the TrustScoreManager.
"""

import numpy as np
from typing import List


class TrustDecay:
    """Trust decay mechanisms for controlling trust memory."""
    
    @staticmethod
    def exponential_decay(old_trust: float, new_observation: float, alpha: float) -> float:
        """
        EMA decay: T_new = α * T_old + (1-α) * observation.
        Higher α = slower decay (longer memory).
        """
        return alpha * old_trust + (1 - alpha) * new_observation
    
    @staticmethod
    def linear_decay(old_trust: float, decay_rate: float = 0.01) -> float:
        """Linear decay for idle clients (no participation this round)."""
        return max(0.0, old_trust - decay_rate)
    
    @staticmethod
    def threshold_decay(
        old_trust: float,
        new_observation: float,
        threshold: float = 0.5,
        fast_decay: float = 0.7,
        slow_decay: float = 0.95
    ) -> float:
        """
        Fast decay if observation below threshold (suspicious behavior),
        slow decay otherwise (normal behavior).
        """
        alpha = fast_decay if new_observation < threshold else slow_decay
        return alpha * old_trust + (1 - alpha) * new_observation
    
    @staticmethod
    def windowed_decay(
        trust_history: List[float],
        window: int = 5,
        weight_recent: float = 0.7
    ) -> float:
        """Weighted average of recent trust history, more weight on recent."""
        if not trust_history:
            return 1.0
        recent = trust_history[-window:]
        if len(recent) == 1:
            return recent[0]
        weights = np.linspace(1 - weight_recent, weight_recent, len(recent))
        weights /= weights.sum()
        return float(np.dot(recent, weights))
    
    @staticmethod
    def adaptive_decay(
        old_trust: float,
        new_observation: float,
        variance: float,
        base_alpha: float = 0.9
    ) -> float:
        """
        Adaptive decay: high variance → faster decay (less reliable history).
        Low variance → slower decay (stable behavior).
        """
        alpha = float(np.clip(base_alpha * np.exp(-variance * 10), 0.5, 0.99))
        return alpha * old_trust + (1 - alpha) * new_observation
    
    @staticmethod
    def momentum_decay(
        old_trust: float,
        new_observation: float,
        momentum: float,
        alpha: float = 0.9,
        beta: float = 0.1
    ):
        """Decay with momentum (SGD-style)."""
        new_momentum = beta * momentum + (1 - beta) * (new_observation - old_trust)
        new_trust = float(np.clip(
            alpha * old_trust + (1 - alpha) * new_observation + new_momentum,
            0.0, 1.0
        ))
        return new_trust, new_momentum


class DecayScheduler:
    """Schedules the alpha (decay) parameter over training."""
    
    def __init__(self, initial_alpha: float = 0.9, strategy: str = "constant"):
        self.initial_alpha = initial_alpha
        self.strategy = strategy
    
    def get_alpha(self, round_num: int, total_rounds: int = 100) -> float:
        if self.strategy == "constant":
            return self.initial_alpha
        elif self.strategy == "linear":
            progress = round_num / max(1, total_rounds)
            return self.initial_alpha + (0.99 - self.initial_alpha) * progress
        elif self.strategy == "cosine":
            progress = round_num / max(1, total_rounds)
            return self.initial_alpha + (0.99 - self.initial_alpha) * (1 - np.cos(progress * np.pi)) / 2
        return self.initial_alpha