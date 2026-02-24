"""
Delayed Poisoning Attack
Attacker behaves normally for D rounds to build trust,
then launches a poisoning attack.

Strategy:
    Round 0 … D-1 : behave honestly → accumulate high trust
    Round D … end  : poison every round (or with some probability)
"""

import torch
import numpy as np
from typing import Dict, Optional


class DelayedPoisoningAttack:
    """
    Delayed poisoning attack.

    Phase 1 (warm-up): Submit clean gradients to build trust.
    Phase 2 (attack):  Submit poisoned gradients to degrade model.

    The key insight: by first building trust, the attacker's
    poisoned updates receive higher aggregation weight, making
    the attack more effective even if total attacking rounds are fewer.
    """

    def __init__(
        self,
        client_id: int,
        delay_rounds: int = 15,
        poisoning_scale: float = 5.0,
        attack_after_delay: str = "always",   # always | intermittent
        attack_probability: float = 1.0,      # used if attack_after_delay == intermittent
        smart_timing: bool = True,            # attack when aggregation weight is highest
    ):
        """
        Args:
            client_id:           Attacker's client ID
            delay_rounds:        Number of warm-up rounds before attacking
            poisoning_scale:     Scale factor for gradient poisoning
            attack_after_delay:  Attack strategy after delay: 'always' or 'intermittent'
            attack_probability:  Attack probability for intermittent strategy
            smart_timing:        If True, prefer attacking when recent trust is high
        """
        self.client_id = client_id
        self.delay_rounds = delay_rounds
        self.poisoning_scale = poisoning_scale
        self.attack_after_delay = attack_after_delay
        self.attack_probability = attack_probability
        self.smart_timing = smart_timing

        # State
        self.current_round = 0
        self.in_warmup = True
        self.attacks_executed = 0
        self.warmup_rounds_completed = 0
        self.attack_history: list = []
        self.estimated_trust_history: list = []

    # ------------------------------------------------------------------
    # Attack decision
    # ------------------------------------------------------------------

    def should_attack(
        self,
        round_num: int,
        estimated_trust: Optional[float] = None
    ) -> bool:
        """
        Decide whether to attack this round.

        Args:
            round_num:       Current training round
            estimated_trust: Attacker's estimate of its trust score

        Returns:
            attack: True if should poison gradient
        """
        self.current_round = round_num

        if estimated_trust is not None:
            self.estimated_trust_history.append(estimated_trust)

        # Phase 1: warm-up
        if round_num < self.delay_rounds:
            self.in_warmup = True
            self.warmup_rounds_completed = round_num + 1
            decision = False
        else:
            # Phase 2: attack
            self.in_warmup = False

            if self.attack_after_delay == "always":
                decision = True
            elif self.attack_after_delay == "intermittent":
                # Optionally be smarter: attack more when trust is high
                if self.smart_timing and estimated_trust is not None:
                    # Higher trust → higher effective attack weight → attack
                    adjusted_prob = self.attack_probability * (0.5 + estimated_trust / 2.0)
                    decision = np.random.random() < adjusted_prob
                else:
                    decision = np.random.random() < self.attack_probability
            else:
                decision = True

        self.attack_history.append(decision)
        if decision:
            self.attacks_executed += 1

        return decision

    # ------------------------------------------------------------------
    # Gradient poisoning
    # ------------------------------------------------------------------

    def poison_gradient(
        self,
        clean_gradient: Dict[str, torch.Tensor],
        scale_override: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Poison the gradient (flip + scale).

        Args:
            clean_gradient:  Clean model update from local training
            scale_override:  Override poisoning scale if provided

        Returns:
            poisoned_gradient: Negated and scaled gradient
        """
        scale = scale_override if scale_override is not None else self.poisoning_scale
        return {name: -scale * param for name, param in clean_gradient.items()}

    def get_clean_gradient(
        self,
        gradient: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Return gradient unchanged (during warm-up phase)."""
        return {name: param.clone() for name, param in gradient.items()}

    # ------------------------------------------------------------------
    # Phase info & statistics
    # ------------------------------------------------------------------

    @property
    def phase(self) -> str:
        """Current attack phase: 'warmup' or 'attack'."""
        return "warmup" if self.in_warmup else "attack"

    def get_statistics(self) -> dict:
        """Return attack statistics."""
        total = len(self.attack_history)
        attack_rate = self.attacks_executed / total if total > 0 else 0.0

        avg_trust_at_attack_time = 0.0
        if self.estimated_trust_history:
            attack_trust = [
                self.estimated_trust_history[i]
                for i, attacked in enumerate(self.attack_history)
                if attacked and i < len(self.estimated_trust_history)
            ]
            if attack_trust:
                avg_trust_at_attack_time = float(np.mean(attack_trust))

        return {
            'type': 'delayed',
            'client_id': self.client_id,
            'delay_rounds': self.delay_rounds,
            'poisoning_scale': self.poisoning_scale,
            'current_phase': self.phase,
            'warmup_rounds_completed': self.warmup_rounds_completed,
            'attacks_executed': self.attacks_executed,
            'total_rounds': total,
            'attack_rate': attack_rate,
            'avg_trust_at_attack': avg_trust_at_attack_time,
            'attack_history': self.attack_history
        }