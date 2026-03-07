"""
Adaptive Attack Controller - STRONGER VERSION
Unified controller for all attack strategies in the FL poisoning experiments.

Attack Types:
    - static:       Always attack every round
    - delayed:      Wait N rounds before attacking (build trust first)
    - intermittent: Attack with probability p each round
    - adaptive:     Monitor estimated trust, attack only when safe
    - little:       LIE (Little Is Enough) — small but effective perturbation
    - min_max:      Min-Max attack — maximize deviation while staying under detection

Poisoning Methods:
    - gradient_flip:    Negate and scale the gradient
    - random_noise:     Add large Gaussian noise
    - sign_flip:        Flip sign of each element
    - targeted_scale:   Scale specific layers more aggressively
    - lie:              Little-Is-Enough small perturbation attack
    - inner_product:    Inner Product Manipulation attack
"""

import torch
import numpy as np
from typing import Dict, List, Optional


class AdaptiveAttackController:
    """
    Adaptive Attack Controller — STRONGER VERSION

    Key changes vs original:
      - LIE (Little Is Enough) attack: small perturbation that evades norm checks
      - Boosted gradient_flip with layer-wise amplification
      - Collaborative poisoning: share reference norm info across attackers
      - Min-Max: maximize impact while keeping L2 norm close to benign average
    """

    def __init__(
        self,
        client_id: int,
        attack_type: str = "static",
        poisoning_scale: float = 10.0,          # STRONGER default
        poison_method: str = "gradient_flip",
        # Intermittent params
        attack_frequency: float = 0.5,           # STRONGER: was 0.3
        # Delayed params
        delay_rounds: int = 10,                  # STRONGER: was 15 (shorter warmup)
        # Adaptive params
        trust_threshold: float = 0.6,            # STRONGER: was 0.7 (attack more)
        knows_trust_mechanism: bool = False,
        knows_alpha: bool = False,
        alpha_estimate: float = 0.9,
        objective: str = "maximize_asr",
        # Dormant params
        dormant_threshold: float = 0.3,
        # Collaborative params
        num_attackers: int = 1,                  # NEW: total # of attackers (for coordinated)
        attacker_ids: Optional[List[int]] = None # NEW: IDs of all attackers
    ):
        self.client_id = client_id
        self.attack_type = attack_type
        self.poisoning_scale = poisoning_scale
        self.poison_method = poison_method

        self.attack_frequency = attack_frequency
        self.delay_rounds = delay_rounds
        self.trust_threshold = trust_threshold
        self.knows_trust_mechanism = knows_trust_mechanism
        self.knows_alpha = knows_alpha
        self.alpha_estimate = alpha_estimate
        self.objective = objective
        self.dormant_threshold = dormant_threshold
        self.num_attackers = num_attackers
        self.attacker_ids = attacker_ids or []

        # Internal state
        self.dormant = False
        self.dormant_since = -1
        self.recovery_rounds = 5               # STRONGER: was 10 (faster recovery)

        # Statistics
        self.total_rounds = 0
        self.total_attacks = 0
        self.attack_history: list = []
        self.trust_history: list = []

        # Running benign norm estimate (updated externally or from recent history)
        self._benign_norm_estimate = 1.0

    # ------------------------------------------------------------------
    # Attack decision
    # ------------------------------------------------------------------

    def should_attack(self, round_num: int, estimated_trust: float = 1.0) -> bool:
        self.total_rounds += 1
        self.trust_history.append(estimated_trust)

        if self.attack_type == "static":
            decision = True
        elif self.attack_type == "delayed":
            decision = round_num >= self.delay_rounds
        elif self.attack_type == "intermittent":
            decision = np.random.random() < self.attack_frequency
        elif self.attack_type == "adaptive":
            decision = self._adaptive_decision(round_num, estimated_trust)
        else:
            decision = True

        self.attack_history.append(decision)
        if decision:
            self.total_attacks += 1
        return decision

    def _adaptive_decision(self, round_num: int, estimated_trust: float) -> bool:
        if estimated_trust < self.dormant_threshold:
            if not self.dormant:
                self.dormant = True
                self.dormant_since = round_num
            return False
        if self.dormant:
            if round_num - self.dormant_since < self.recovery_rounds:
                return False
            self.dormant = False
        if self.knows_trust_mechanism and self.objective == "stay_hidden":
            return estimated_trust > 0.80
        return estimated_trust > self.trust_threshold

    # ------------------------------------------------------------------
    # Gradient poisoning — STRONGER IMPLEMENTATIONS
    # ------------------------------------------------------------------

    def poison_gradient(
        self,
        clean_gradient: Dict[str, torch.Tensor],
        poison_method: Optional[str] = None,
        all_updates: Optional[List[Dict[str, torch.Tensor]]] = None,
        benign_norm: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply poisoning to a clean gradient.

        Args:
            clean_gradient: Client's genuine model update
            poison_method:  Override default method if provided
            all_updates:    All updates this round (for min-max / LIE)
            benign_norm:    Observed benign gradient norm (for norm-matching)
        """
        method = poison_method or self.poison_method

        if benign_norm is not None:
            self._benign_norm_estimate = benign_norm

        if method == "gradient_flip":
            return self._gradient_flip_boosted(clean_gradient)
        elif method == "random_noise":
            return self._random_noise(clean_gradient)
        elif method == "sign_flip":
            return self._sign_flip(clean_gradient)
        elif method == "targeted_scale":
            return self._targeted_scale(clean_gradient)
        elif method == "lie":
            return self._lie_attack(clean_gradient, all_updates)
        elif method == "inner_product":
            return self._inner_product_manipulation(clean_gradient)
        elif method == "min_max":
            return self._min_max_attack(clean_gradient, all_updates)
        else:
            return self._gradient_flip_boosted(clean_gradient)

    def _gradient_flip_boosted(self, gradient: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        STRONGER: Negate + scale, with extra amplification on later (decision) layers.
        """
        param_names = list(gradient.keys())
        n = len(param_names)
        poisoned = {}
        for i, name in enumerate(param_names):
            # Later layers get up to 2x extra scale
            layer_boost = 1.0 + (i / max(1, n - 1))
            poisoned[name] = -self.poisoning_scale * layer_boost * gradient[name]
        return poisoned

    def _random_noise(self, gradient: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Replace gradient with large Gaussian noise."""
        return {name: self.poisoning_scale * torch.randn_like(p)
                for name, p in gradient.items()}

    def _sign_flip(self, gradient: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {name: -self.poisoning_scale * p for name, p in gradient.items()}

    def _targeted_scale(self, gradient: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        param_names = list(gradient.keys())
        n = len(param_names)
        poisoned = {}
        for i, name in enumerate(param_names):
            layer_scale = self.poisoning_scale * (1.5 + i / max(1, n - 1))  # stronger
            poisoned[name] = -layer_scale * gradient[name]
        return poisoned

    def _lie_attack(
        self,
        gradient: Dict[str, torch.Tensor],
        all_updates: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        LIE (Little Is Enough) attack — Baruch et al., NeurIPS 2019.
        Computes z-score bounds and perturbs just enough to flip sign of aggregation
        while evading Byzantine-robust defenses.

        Without knowledge of other updates, falls back to boosted sign_flip
        scaled to match benign norms.
        """
        if all_updates and len(all_updates) >= 2:
            # Compute mean and std across provided updates
            stacked = {k: torch.stack([u[k] for u in all_updates]) for k in gradient}
            mean_g = {k: v.mean(dim=0) for k, v in stacked.items()}
            std_g  = {k: v.std(dim=0).clamp(min=1e-6) for k, v in stacked.items()}

            # z_max for n clients, f Byzantine (Theorem 1 in LIE paper)
            n = len(all_updates)
            f = max(1, int(n * 0.2))
            # approximate z_max ≈ 1.5 for typical settings
            z_max = 1.5

            poisoned = {}
            for k in gradient:
                # push values just past the Krum/TrimMean detection boundary
                poisoned[k] = mean_g[k] - z_max * std_g[k]
            return poisoned
        else:
            # Fallback: norm-matched sign flip
            return self._norm_matched_flip(gradient)

    def _inner_product_manipulation(
        self, gradient: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        IPM attack — Xie et al., ICLR 2020.
        Sends -epsilon * gradient to maximally reduce inner product with global model.
        epsilon > n (number of benign clients) ensures the attack dominates.
        """
        epsilon = max(self.poisoning_scale, self.num_attackers + 2.0)
        return {name: -epsilon * p for name, p in gradient.items()}

    def _min_max_attack(
        self,
        gradient: Dict[str, torch.Tensor],
        all_updates: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Min-Max attack — Shejwalkar & Houmansadr, NDSS 2021.
        Maximize deviation from mean while keeping distance to closest benign update small.
        Falls back to boosted flip if no reference updates available.
        """
        if all_updates and len(all_updates) >= 2:
            stacked = {k: torch.stack([u[k] for u in all_updates]) for k in gradient}
            mean_g = {k: v.mean(dim=0) for k, v in stacked.items()}
            # Direction: away from mean
            direction = {k: gradient[k] - mean_g[k] for k in gradient}
            dir_norm = self._compute_norm(direction)
            scale = self.poisoning_scale / max(dir_norm, 1e-6)
            return {k: gradient[k] + scale * direction[k] for k in gradient}
        return self._gradient_flip_boosted(gradient)

    def _norm_matched_flip(self, gradient: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Flip + rescale to match benign norm estimate."""
        flipped = {name: -p for name, p in gradient.items()}
        flip_norm = self._compute_norm(flipped)
        target_norm = self._benign_norm_estimate * self.poisoning_scale
        scale = target_norm / max(flip_norm, 1e-6)
        return {name: scale * p for name, p in flipped.items()}

    @staticmethod
    def _compute_norm(gradient: Dict[str, torch.Tensor]) -> float:
        total = sum(torch.norm(p).item() ** 2 for p in gradient.values())
        return float(np.sqrt(total))

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict:
        attack_rate = self.total_attacks / max(1, self.total_rounds)
        attacked_trusts = [
            self.trust_history[i]
            for i, attacked in enumerate(self.attack_history)
            if attacked and i < len(self.trust_history)
        ]
        avg_trust_when_attacking = float(np.mean(attacked_trusts)) if attacked_trusts else 0.0
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
        self.dormant = False
        self.dormant_since = -1
        self.total_rounds = 0
        self.total_attacks = 0
        self.attack_history = []
        self.trust_history = []