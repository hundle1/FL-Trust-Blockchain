"""
Intermittent Poisoning Attack
Attacker attacks randomly with certain probability
"""

import torch
import numpy as np
from typing import Dict


class IntermittentPoisoningAttack:
    """
    Intermittent attack: attack with probability p in each round
    """
    
    def __init__(
        self,
        client_id: int,
        attack_probability: float = 0.3,
        poisoning_scale: float = 5.0,
        pattern: str = "random"
    ):
        """
        Args:
            client_id: Attacker's client ID
            attack_probability: Probability of attacking in each round
            poisoning_scale: Scale factor for poisoning
            pattern: Attack pattern (random, periodic, burst)
        """
        self.client_id = client_id
        self.attack_probability = attack_probability
        self.poisoning_scale = poisoning_scale
        self.pattern = pattern
        
        self.current_round = 0
        self.attacks_executed = 0
        self.attack_history = []
    
    def should_attack(self, round_num: int) -> bool:
        """
        Decide whether to attack this round
        
        Args:
            round_num: Current round number
            
        Returns:
            attack: True if should attack
        """
        self.current_round = round_num
        
        if self.pattern == "random":
            # Random attack with fixed probability
            attack = np.random.random() < self.attack_probability
        
        elif self.pattern == "periodic":
            # Attack every N rounds
            period = int(1 / self.attack_probability)
            attack = (round_num % period) == 0
        
        elif self.pattern == "burst":
            # Attack in bursts: 3 consecutive rounds, then idle
            burst_length = 3
            period = int(burst_length / self.attack_probability)
            position_in_period = round_num % period
            attack = position_in_period < burst_length
        
        else:
            # Default: random
            attack = np.random.random() < self.attack_probability
        
        self.attack_history.append(attack)
        
        if attack:
            self.attacks_executed += 1
        
        return attack
    
    def poison_gradient(self, clean_gradient: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply poisoning to gradient
        
        Args:
            clean_gradient: Clean gradient
            
        Returns:
            poisoned_gradient: Poisoned gradient
        """
        poisoned = {}
        
        for name, param in clean_gradient.items():
            # Flip and scale
            poisoned[name] = -self.poisoning_scale * param
        
        return poisoned
    
    def get_statistics(self) -> dict:
        """Get attack statistics"""
        total_rounds = len(self.attack_history)
        attack_rate = self.attacks_executed / total_rounds if total_rounds > 0 else 0.0
        
        # Calculate burstiness
        if len(self.attack_history) > 1:
            consecutive_attacks = 0
            max_consecutive = 0
            for attacked in self.attack_history:
                if attacked:
                    consecutive_attacks += 1
                    max_consecutive = max(max_consecutive, consecutive_attacks)
                else:
                    consecutive_attacks = 0
        else:
            max_consecutive = 0
        
        return {
            'type': 'intermittent',
            'pattern': self.pattern,
            'attack_probability': self.attack_probability,
            'current_round': self.current_round,
            'attacks_executed': self.attacks_executed,
            'attack_rate': attack_rate,
            'max_consecutive_attacks': max_consecutive,
            'attack_history': self.attack_history
        }