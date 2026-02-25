"""
Intermittent Poisoning Attack
Attacker attacks randomly with certain probability per round.
"""

import torch
import numpy as np
from typing import Dict


class IntermittentPoisoningAttack:
    """
    Intermittent attack: attack with probability p in each round.
    Supports random, periodic, and burst patterns.
    """
    
    def __init__(
        self,
        client_id: int,
        attack_probability: float = 0.3,
        poisoning_scale: float = 5.0,
        pattern: str = "random"   # random | periodic | burst
    ):
        self.client_id = client_id
        self.attack_probability = attack_probability
        self.poisoning_scale = poisoning_scale
        self.pattern = pattern
        
        self.current_round = 0
        self.attacks_executed = 0
        self.attack_history = []
    
    def should_attack(self, round_num: int) -> bool:
        self.current_round = round_num
        
        if self.pattern == "random":
            attack = np.random.random() < self.attack_probability
        
        elif self.pattern == "periodic":
            period = int(1 / max(1e-6, self.attack_probability))
            attack = (round_num % period) == 0
        
        elif self.pattern == "burst":
            # Attack in bursts of 3, then rest
            burst_length = 3
            period = int(burst_length / max(1e-6, self.attack_probability))
            attack = (round_num % period) < burst_length
        
        else:
            attack = np.random.random() < self.attack_probability
        
        self.attack_history.append(attack)
        if attack:
            self.attacks_executed += 1
        return attack
    
    def poison_gradient(self, clean_gradient: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {name: -self.poisoning_scale * param for name, param in clean_gradient.items()}
    
    def get_statistics(self) -> dict:
        total_rounds = len(self.attack_history)
        attack_rate = self.attacks_executed / total_rounds if total_rounds > 0 else 0.0
        
        max_consecutive = 0
        consecutive = 0
        for a in self.attack_history:
            if a:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        
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