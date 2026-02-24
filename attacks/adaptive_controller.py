"""
Adaptive Attack Controller
Unified controller for all attack strategies in the FL poisoning experiments.

Attack Types:
    - static:       Always attack every round
    - delayed:      Wait N rounds before attacking (build trust first)
    - intermittent: Attack with probability p each round
    - adaptive:     Monitor estimated trust, attack only when safe

Poisoning Methods:
    - gradient_flip:    Negate and scale the gradient
    - random_noise:     Add large Gaussian noise
    - sign_flip:        Flip sign of each element
    - targeted_scale:   Scale specific layers more aggressively
"""

import torch
import numpy as np
from typing import Dict, Optional


class AdaptiveAttackController:
    """
    Adaptive Attack Controller

    Manages attack decisions and gradient poisoning for a single malicious client.
    Supports multiple attack strategies and poisoning methods.
    Can be configured with knowledge about the trust mechanism.
    """

    def __init__(
        self,
        client_id: int,
        attack_type: str = "static",
        poisoning_scale: float = 5.0,
        poison_method: str = "gradient_flip",
        # Intermittent params
        attack_frequency: float = 0.3,
        # Delayed params
        delay_rounds: int = 15,
        # Adaptive params
        trust_threshold: float = 0.7,
        knows_trust_mechanism: bool = False,
        knows_alpha: bool = False,
        alpha_estimate: float = 0.9,
        objective: str = "maximize_asr",   # maximize_asr | stay_hidden
        # Dormant params
        dormant_threshold: float = 0.3,
    ):
        """
        Args:
            client_id:              Attacker's client ID
            attack_type:            Strategy: static | delayed | intermittent | adaptive
            poisoning_scale:        Multiplier for poisoned gradient
            poison_method:          Poisoning method: gradient_flip | random_noise | sign_flip
            attack_frequency:       Probability of attack for intermittent strategy
            delay_rounds:           Rounds to wait before attacking (delayed strategy)
            trust_threshold:        Threshold above which adaptive attacker attacks
            knows_trust_mechanism:  Whether attacker knows trust is being tracked
            knows_alpha:            Whether attacker knows α parameter
            alpha_estimate:         Attacker's estimate of α (if knows_alpha=True)
            objective:              Attacker objective
            dormant_threshold:      Trust below this → enter dormant mode
        """
        self.client_id = client_id
        self.attack_type = attack_type
        self.poisoning_scale = poisoning_scale
        self.poison_method = poison_method

        # Intermittent
        self.attack_frequency = attack_frequency

        # Delayed
        self.delay_rounds = delay_rounds

        # Adaptive
        self.trust_threshold = trust_threshold
        self.knows_trust_mechanism = knows_trust_mechanism
        self.knows_alpha = knows_alpha
        self.alpha_estimate = alpha_estimate
        self.objective = objective
        self.dormant_threshold = dormant_threshold

        # Internal state
        self.dormant = False
        self.dormant_since = -1
        self.recovery_rounds = 10  # rounds to stay dormant before recovery

        # Statistics
        self.total_rounds = 0
        self.total_attacks = 0
        self.attack_history: list = []    # True/False per round
        self.trust_history: list = []     # estimated trust per round

    # ------------------------------------------------------------------
    # Attack decision
    # ------------------------------------------------------------------

    def should_attack(self, round_num: int, estimated_trust: float = 1.0) -> bool:
        """
        Decide whether to attack in this round.

        Args:
            round_num:       Current training round
            estimated_trust: Attacker's estimate of its own trust score

        Returns:
            attack: True if should attack
        """
        self.total_rounds += 1
        self.trust_history.append(estimated_trust)

        if self.attack_type == "static":
            decision = self._static_decision()

        elif self.attack_type == "delayed":
            decision = self._delayed_decision(round_num)

        elif self.attack_type == "intermittent":
            decision = self._intermittent_decision(round_num)

        elif self.attack_type == "adaptive":
            decision = self._adaptive_decision(round_num, estimated_trust)

        else:
            decision = self._static_decision()

        self.attack_history.append(decision)
        if decision:
            self.total_attacks += 1

        return decision

    def _static_decision(self) -> bool:
        """Always attack."""
        return True

    def _delayed_decision(self, round_num: int) -> bool:
        """Attack only after delay_rounds have passed."""
        return round_num >= self.delay_rounds

    def _intermittent_decision(self, round_num: int) -> bool:
        """Attack randomly with probability attack_frequency."""
        return np.random.random() < self.attack_frequency

    def _adaptive_decision(self, round_num: int, estimated_trust: float) -> bool:
        """
        Adaptive strategy that responds to estimated trust level.

        If knows_trust_mechanism:
            - Stay dormant when trust is low (to recover)
            - Attack aggressively when trust is high
        Else:
            - Simple threshold check
        """
        # Check if entering / maintaining dormant phase
        if estimated_trust < self.dormant_threshold:
            if not self.dormant:
                self.dormant = True
                self.dormant_since = round_num
            return False

        # Recovery: stay dormant for recovery_rounds after trust drops
        if self.dormant:
            rounds_dormant = round_num - self.dormant_since
            if rounds_dormant < self.recovery_rounds:
                return False
            else:
                # Enough time has passed, try to recover
                self.dormant = False

        if self.knows_trust_mechanism:
            if self.objective == "stay_hidden":
                # Only attack if trust is very high (minimize detection risk)
                return estimated_trust > 0.85
            else:  # maximize_asr
                # Attack whenever trust is safely above threshold
                return estimated_trust > self.trust_threshold
        else:
            # Blind attacker: just check threshold
            return estimated_trust > self.trust_threshold

    # ------------------------------------------------------------------
    # Gradient poisoning
    # ------------------------------------------------------------------

    def poison_gradient(
        self,
        clean_gradient: Dict[str, torch.Tensor],
        poison_method: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply poisoning to a clean gradient.

        Args:
            clean_gradient: Client's genuine model update
            poison_method:  Override instance's default method if provided

        Returns:
            poisoned_gradient: Modified gradient
        """
        method = poison_method or self.poison_method

        if method == "gradient_flip":
            return self._gradient_flip(clean_gradient)
        elif method == "random_noise":
            return self._random_noise(clean_gradient)
        elif method == "sign_flip":
            return self._sign_flip(clean_gradient)
        elif method == "targeted_scale":
            return self._targeted_scale(clean_gradient)
        else:
            return self._gradient_flip(clean_gradient)

    def _gradient_flip(self, gradient: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Negate gradient and multiply by poisoning_scale."""
        return {name: -self.poisoning_scale * param for name, param in gradient.items()}

    def _random_noise(self, gradient: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Replace gradient with large random noise."""
        poisoned = {}
        for name, param in gradient.items():
            noise = torch.randn_like(param)
            poisoned[name] = self.poisoning_scale * noise
        return poisoned

    def _sign_flip(self, gradient: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Flip the sign of each element (magnitude preserved, direction reversed)."""
        return {name: -param * self.poisoning_scale for name, param in gradient.items()}

    def _targeted_scale(self, gradient: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Aggressively scale specific layers (last layers get higher scale).
        Designed to corrupt decision boundaries while being harder to detect.
        """
        poisoned = {}
        param_names = list(gradient.keys())
        n = len(param_names)

        for i, name in enumerate(param_names):
            # Scale increases for later layers
            layer_scale = self.poisoning_scale * (1.0 + i / max(1, n - 1))
            poisoned[name] = -layer_scale * gradient[name]

        return poisoned

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict:
        """Return attack statistics."""
        attack_rate = self.total_attacks / max(1, self.total_rounds)

        avg_trust_when_attacking = 0.0
        if self.attack_history and self.trust_history:
            attacked_trusts = [
                self.trust_history[i]
                for i, attacked in enumerate(self.attack_history)
                if attacked and i < len(self.trust_history)
            ]
            if attacked_trusts:
                avg_trust_when_attacking = float(np.mean(attacked_trusts))

        return {
            'client_id': self.client_id,
            'attack_type': self.attack_type,
            'poison_method': self.poison_method,
            'total_rounds': self.total_rounds,
            'total_attacks': self.total_attacks,
            'attack_rate': attack_rate,
            'dormant': self.dormant,
            'attack_history': self.attack_history,
            'trust_history': self.trust_history,
            'avg_trust_when_attacking': avg_trust_when_attacking
        }

    def reset(self):
        """Reset attack controller state."""
        self.dormant = False
        self.dormant_since = -1
        self.total_rounds = 0
        self.total_attacks = 0
        self.attack_history = []
        self.trust_history = []