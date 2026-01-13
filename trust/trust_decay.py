"""
Trust Decay Module
Implements various trust decay strategies
"""

import numpy as np
from typing import List


class TrustDecay:
    """
    Trust decay mechanisms for controlling trust memory
    """
    
    @staticmethod
    def exponential_decay(
        old_trust: float,
        new_observation: float,
        alpha: float
    ) -> float:
        """
        Exponential moving average decay
        
        T_new = α * T_old + (1 - α) * observation
        
        Args:
            old_trust: Previous trust score
            new_observation: New trust observation from current round
            alpha: Decay parameter (higher = slower decay)
            
        Returns:
            new_trust: Updated trust score
        """
        return alpha * old_trust + (1 - alpha) * new_observation
    
    @staticmethod
    def linear_decay(
        old_trust: float,
        decay_rate: float = 0.01
    ) -> float:
        """
        Linear decay over time (for idle clients)
        
        Args:
            old_trust: Previous trust score
            decay_rate: Rate of decay per round
            
        Returns:
            new_trust: Decayed trust score
        """
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
        Threshold-based decay: fast decay if observation below threshold
        
        Args:
            old_trust: Previous trust score
            new_observation: New observation
            threshold: Threshold for switching decay rates
            fast_decay: Alpha for fast decay
            slow_decay: Alpha for slow decay
            
        Returns:
            new_trust: Updated trust score
        """
        if new_observation < threshold:
            alpha = fast_decay
        else:
            alpha = slow_decay
        
        return alpha * old_trust + (1 - alpha) * new_observation
    
    @staticmethod
    def windowed_decay(
        trust_history: List[float],
        window: int = 5,
        weight_recent: float = 0.7
    ) -> float:
        """
        Window-based decay: weighted average of recent history
        
        Args:
            trust_history: Historical trust scores
            window: Size of window to consider
            weight_recent: Weight for most recent values
            
        Returns:
            new_trust: Computed trust from window
        """
        if not trust_history:
            return 1.0
        
        recent = trust_history[-window:]
        
        if len(recent) == 1:
            return recent[0]
        
        # Weighted average with more weight on recent
        weights = np.linspace(1 - weight_recent, weight_recent, len(recent))
        weights = weights / weights.sum()
        
        return float(np.dot(recent, weights))
    
    @staticmethod
    def adaptive_decay(
        old_trust: float,
        new_observation: float,
        variance: float,
        base_alpha: float = 0.9
    ) -> float:
        """
        Adaptive decay based on behavior variance
        
        High variance → faster decay (less reliable)
        Low variance → slower decay (more stable)
        
        Args:
            old_trust: Previous trust score
            new_observation: New observation
            variance: Variance in recent behavior
            base_alpha: Base alpha parameter
            
        Returns:
            new_trust: Updated trust score
        """
        # Adjust alpha based on variance
        # High variance → lower alpha → faster decay
        alpha = base_alpha * np.exp(-variance * 10)
        alpha = max(0.5, min(0.99, alpha))
        
        return alpha * old_trust + (1 - alpha) * new_observation
    
    @staticmethod
    def momentum_decay(
        old_trust: float,
        new_observation: float,
        momentum: float,
        alpha: float = 0.9,
        beta: float = 0.1
    ) -> tuple:
        """
        Decay with momentum (similar to SGD with momentum)
        
        Args:
            old_trust: Previous trust score
            new_observation: New observation
            momentum: Previous momentum
            alpha: Trust decay parameter
            beta: Momentum parameter
            
        Returns:
            new_trust, new_momentum: Updated trust and momentum
        """
        # Update momentum
        new_momentum = beta * momentum + (1 - beta) * (new_observation - old_trust)
        
        # Update trust with momentum
        new_trust = alpha * old_trust + (1 - alpha) * new_observation + new_momentum
        
        # Clip to valid range
        new_trust = max(0.0, min(1.0, new_trust))
        
        return new_trust, new_momentum


class DecayScheduler:
    """
    Schedules decay parameters over training
    """
    
    def __init__(self, initial_alpha: float = 0.9, strategy: str = "constant"):
        """
        Args:
            initial_alpha: Starting alpha value
            strategy: Decay strategy (constant, linear, cosine)
        """
        self.initial_alpha = initial_alpha
        self.strategy = strategy
        self.current_round = 0
    
    def get_alpha(self, round_num: int, total_rounds: int = 100) -> float:
        """
        Get alpha for current round
        
        Args:
            round_num: Current round number
            total_rounds: Total training rounds
            
        Returns:
            alpha: Alpha value for this round
        """
        self.current_round = round_num
        
        if self.strategy == "constant":
            return self.initial_alpha
        
        elif self.strategy == "linear":
            # Linearly increase alpha over time (trust becomes more stable)
            progress = round_num / total_rounds
            return self.initial_alpha + (0.99 - self.initial_alpha) * progress
        
        elif self.strategy == "cosine":
            # Cosine annealing
            progress = round_num / total_rounds
            return self.initial_alpha + (0.99 - self.initial_alpha) * \
                   (1 - np.cos(progress * np.pi)) / 2
        
        else:
            return self.initial_alpha