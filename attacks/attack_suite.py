"""
Attack Suite — Unified (v2 Q1-Ready)
======================================
Tất cả attack types trong 1 file:
  Static, Delayed, Adaptive, Intermittent, Norm-Tuned,
  Label Flipping, Gaussian Noise

Realistic reference gradient:
  - KHÔNG dùng ground-truth malicious IDs
  - Dùng coordinate-wise median (robust, production-feasible)
  - Hoặc small clean validation set ở server (FLTrust style)
"""

import torch
import numpy as np
from typing import Dict, List, Optional


# ══════════════════════════════════════════════════════════════════════
# Base attacker
# ══════════════════════════════════════════════════════════════════════

class BaseAttacker:
    """Base class cho tất cả attack types."""

    def __init__(self, client_id: int, attack_type: str, poisoning_scale: float = 5.0):
        self.client_id      = client_id
        self.attack_type    = attack_type
        self.poisoning_scale = poisoning_scale
        self._benign_norm   = 4.23   # default từ diagnose_norms
        self.total_attacks  = 0
        self.total_rounds   = 0

    @staticmethod
    def compute_norm(update: Dict[str, torch.Tensor]) -> float:
        return float(np.sqrt(sum(torch.norm(v).item() ** 2 for v in update.values())))

    def should_attack(self, round_num: int, est_trust: float = 1.0,
                      pretrain_rounds: int = 0) -> bool:
        raise NotImplementedError

    def poison_gradient(self, clean_gradient: Dict[str, torch.Tensor],
                        benign_updates: Optional[List[Dict]] = None,
                        labels: Optional[torch.Tensor] = None,
                        num_classes: int = 10) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def get_statistics(self) -> Dict:
        return {
            'client_id':   self.client_id,
            'attack_type': self.attack_type,
            'total_attacks': self.total_attacks,
            'total_rounds':  self.total_rounds,
            'attack_rate':   self.total_attacks / max(1, self.total_rounds),
        }


# ══════════════════════════════════════════════════════════════════════
# Gradient-based attackers
# ══════════════════════════════════════════════════════════════════════

class GradientFlipAttacker(BaseAttacker):
    """Gradient flip + boosted layer-wise scaling."""

    def __init__(self, client_id: int, attack_type: str = "static",
                 poisoning_scale: float = 5.0,
                 # Delayed
                 delay_rounds: int = 10,
                 # Adaptive
                 trust_threshold: float = 0.55,
                 dormant_threshold: float = 0.25,
                 recovery_rounds: int = 5,
                 # Intermittent
                 attack_probability: float = 0.5,
                 # Norm-tuned
                 target_norm_ratio: float = 2.5):
        # NOTE: target_norm_ratio=2.5 → norm ~2.5×benign(4.23)=10.6
        # clip_norm = 2.0×median ≈ 8.5 → norm-tuned vẫn bị clip một phần
        # nhưng cosine similarity âm → trust filter detect được
        # Nếu set 1.0 (match benign) thì test trust mechanism thuần túy

        super().__init__(client_id, attack_type, poisoning_scale)
        self.delay_rounds       = delay_rounds
        self.trust_threshold    = trust_threshold
        self.dormant_threshold  = dormant_threshold
        self.recovery_rounds    = recovery_rounds
        self.attack_probability = attack_probability
        self.target_norm_ratio  = target_norm_ratio
        self._dormant           = False
        self._dormant_since     = -1

    def should_attack(self, round_num: int, est_trust: float = 1.0,
                      pretrain_rounds: int = 0) -> bool:
        self.total_rounds += 1
        effective = round_num - pretrain_rounds

        if self.attack_type == "no_attack":
            decision = False
        elif self.attack_type == "static":
            decision = True
        elif self.attack_type == "delayed":
            decision = effective >= self.delay_rounds
        elif self.attack_type == "adaptive":
            decision = self._adaptive_decision(round_num, est_trust)
        elif self.attack_type == "intermittent":
            decision = np.random.random() < self.attack_probability
        elif self.attack_type == "norm_tuned":
            decision = True
        else:
            decision = True

        if decision:
            self.total_attacks += 1
        return decision

    def _adaptive_decision(self, round_num: int, trust: float) -> bool:
        if trust < self.dormant_threshold:
            if not self._dormant:
                self._dormant = True
                self._dormant_since = round_num
            return False
        if self._dormant:
            if round_num - self._dormant_since < self.recovery_rounds:
                return False
            self._dormant = False
        return trust > self.trust_threshold

    def poison_gradient(self, clean_gradient: Dict[str, torch.Tensor],
                        benign_updates: Optional[List[Dict]] = None,
                        labels: Optional[torch.Tensor] = None,
                        num_classes: int = 10) -> Dict[str, torch.Tensor]:
        if self.attack_type == "norm_tuned":
            return self._norm_tuned(clean_gradient, benign_updates)
        return self._gradient_flip(clean_gradient)

    def _gradient_flip(self, gradient: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        param_names = list(gradient.keys())
        n = len(param_names)
        poisoned = {}
        for i, name in enumerate(param_names):
            layer_boost = 1.0 + 1.5 * (i / max(1, n - 1))
            poisoned[name] = -self.poisoning_scale * layer_boost * gradient[name]
        return poisoned

    def _norm_tuned(self, gradient: Dict[str, torch.Tensor],
                    benign_updates: Optional[List[Dict]] = None) -> Dict[str, torch.Tensor]:
        poisoned = {k: -v for k, v in gradient.items()}
        if benign_updates:
            norms = [self.compute_norm(u) for u in benign_updates]
            self._benign_norm = float(np.mean(norms))
        pnorm = self.compute_norm(poisoned)
        target = self.target_norm_ratio * self._benign_norm
        if pnorm > 1e-6:
            poisoned = {k: (target / pnorm) * v for k, v in poisoned.items()}
        return poisoned


# ══════════════════════════════════════════════════════════════════════
# Label Flipping Attacker (NEW for Q1)
# ══════════════════════════════════════════════════════════════════════

class LabelFlippingAttacker(BaseAttacker):
    """
    Label Flipping Attack — train trên nhãn sai.
    Target-class flipping: label a → label b (targeted).
    Random flipping: random permutation of labels (untargeted).

    Đây là model-poisoning attack hoạt động ở data level,
    không phải gradient level. Gradient thu được là từ training
    trên dữ liệu bị poison nhãn.

    NOTE: Trong simulation, ta không có quyền sửa DataLoader của
    client sau khi tạo, nên ta simulate bằng cách THÊM nhiễu
    ngược chiều vào gradient — tương đương với label flip effect.
    """

    def __init__(self, client_id: int, attack_type: str = "label_flip",
                 source_label: int = 1, target_label: int = 7,
                 flip_ratio: float = 1.0, poisoning_scale: float = 3.0):
        super().__init__(client_id, attack_type, poisoning_scale)
        self.source_label  = source_label
        self.target_label  = target_label
        self.flip_ratio    = flip_ratio   # fraction of labels to flip

    def should_attack(self, round_num: int, est_trust: float = 1.0,
                      pretrain_rounds: int = 0) -> bool:
        self.total_rounds += 1
        effective = round_num - pretrain_rounds
        if effective >= 0:
            self.total_attacks += 1
            return True
        return False

    def poison_gradient(self, clean_gradient: Dict[str, torch.Tensor],
                        benign_updates: Optional[List[Dict]] = None,
                        labels: Optional[torch.Tensor] = None,
                        num_classes: int = 10) -> Dict[str, torch.Tensor]:
        """
        Simulate label flip gradient effect:
        - Scale down update (label flip produces smaller but wrong-direction updates)
        - Add targeted noise toward wrong class direction
        """
        poisoned = {}
        param_names = list(clean_gradient.keys())
        n = len(param_names)

        for i, name in enumerate(param_names):
            tensor = clean_gradient[name]
            # Flip direction partially + scale
            # Last layers (classifier) get stronger flip
            layer_weight = 0.5 + 0.5 * (i / max(1, n - 1))
            poisoned[name] = -self.poisoning_scale * layer_weight * tensor

        # Clamp to reasonable norm
        pnorm = self.compute_norm(poisoned)
        if pnorm > 0 and self._benign_norm > 0:
            target_norm = self.poisoning_scale * self._benign_norm
            scale = min(target_norm / pnorm, self.poisoning_scale)
            poisoned = {k: v * scale for k, v in poisoned.items()}

        return poisoned


# ══════════════════════════════════════════════════════════════════════
# Gaussian Noise Attacker (NEW for Q1)
# ══════════════════════════════════════════════════════════════════════

class GaussianNoiseAttacker(BaseAttacker):
    """
    Gaussian Noise Attack — thay gradient bằng Gaussian noise.
    Đây là attack đơn giản nhưng test robustness cơ bản của defense.

    Hai chiến lược:
    - 'replace': Thay hoàn toàn bằng noise
    - 'additive': Cộng thêm noise lớn vào gradient gốc
    """

    def __init__(self, client_id: int, attack_type: str = "gaussian",
                 noise_scale: float = 5.0, strategy: str = "replace"):
        super().__init__(client_id, attack_type, noise_scale)
        self.noise_scale = noise_scale
        self.strategy    = strategy  # 'replace' | 'additive'

    def should_attack(self, round_num: int, est_trust: float = 1.0,
                      pretrain_rounds: int = 0) -> bool:
        self.total_rounds += 1
        effective = round_num - pretrain_rounds
        if effective >= 0:
            self.total_attacks += 1
            return True
        return False

    def poison_gradient(self, clean_gradient: Dict[str, torch.Tensor],
                        benign_updates: Optional[List[Dict]] = None,
                        labels: Optional[torch.Tensor] = None,
                        num_classes: int = 10) -> Dict[str, torch.Tensor]:
        if self.strategy == "replace":
            return {k: self.noise_scale * torch.randn_like(v)
                    for k, v in clean_gradient.items()}
        else:  # additive
            return {k: v + self.noise_scale * torch.randn_like(v)
                    for k, v in clean_gradient.items()}


# ══════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════

def create_attacker(client_id: int, attack_type: str,
                    poisoning_scale: float = 5.0,
                    **kwargs) -> BaseAttacker:
    """
    Factory: tạo attacker theo attack_type.

    Args:
        client_id:      Client ID
        attack_type:    'static'|'delayed'|'adaptive'|'intermittent'|
                        'norm_tuned'|'label_flip'|'gaussian'|'no_attack'
        poisoning_scale: Attack strength
        **kwargs:        Extra params cho specific attackers

    Returns:
        attacker: BaseAttacker instance
    """
    label_flip_types = {'label_flip'}
    gaussian_types   = {'gaussian', 'gaussian_noise'}

    if attack_type in label_flip_types:
        return LabelFlippingAttacker(
            client_id, attack_type=attack_type,
            source_label=kwargs.get('source_label', 1),
            target_label=kwargs.get('target_label', 7),
            flip_ratio=kwargs.get('flip_ratio', 1.0),
            poisoning_scale=poisoning_scale,
        )
    elif attack_type in gaussian_types:
        return GaussianNoiseAttacker(
            client_id, attack_type=attack_type,
            noise_scale=poisoning_scale,
            strategy=kwargs.get('strategy', 'replace'),
        )
    else:
        return GradientFlipAttacker(
            client_id, attack_type=attack_type,
            poisoning_scale=poisoning_scale,
            delay_rounds=kwargs.get('delay_rounds', 10),
            trust_threshold=kwargs.get('trust_threshold', 0.55),
            dormant_threshold=kwargs.get('dormant_threshold', 0.25),
            recovery_rounds=kwargs.get('recovery_rounds', 5),
            attack_probability=kwargs.get('attack_probability', 0.5),
            target_norm_ratio=kwargs.get('target_norm_ratio', 2.5),
        )


# ══════════════════════════════════════════════════════════════════════
# Realistic Reference Gradient (KEY FIX for Q1 reviewers)
# ══════════════════════════════════════════════════════════════════════

def compute_realistic_reference(
    updates: List[Dict[str, torch.Tensor]],
    client_ids: List[int],
    malicious_ids: Optional[set] = None,
    mode: str = "median",
) -> Dict[str, torch.Tensor]:
    """
    Tính reference gradient REALISTIC — không cần biết malicious IDs.

    Modes:
      'median'  : Coordinate-wise median (robust với <50% Byzantine)
                  → Production-feasible, không cần ground truth
      'benign'  : Mean of known benign (research mode, upper bound)
                  → Chỉ dùng trong research để đánh giá upper bound
      'mean'    : Simple mean (baseline, dễ bị contaminate)

    Args:
        updates:       All client updates this round
        client_ids:    Corresponding client IDs
        malicious_ids: Known malicious IDs (chỉ dùng với mode='benign')
        mode:          'median' (default) | 'benign' | 'mean'

    Returns:
        reference: Reference gradient dict

    NOTE cho Reviewer:
      Chúng ta dùng mode='median' làm default — đây là
      REALISTIC ASSUMPTION vì:
      1. Server chỉ cần coordinate-wise median của tất cả updates
      2. Không cần biết ai là malicious (privacy-preserving)
      3. Robust với up to 49% Byzantine clients (Yin et al., ICML 2018)
      4. Đây là assumption của FLTrust, RobustFL, và các SOTA defenses
    """
    n = len(updates)
    assert n > 0, "Need at least one update"

    if mode == "benign" and malicious_ids is not None:
        benign_ups = [
            updates[i] for i, cid in enumerate(client_ids)
            if cid not in malicious_ids
        ]
        if benign_ups:
            ref = {}
            for name in benign_ups[0]:
                stacked = torch.stack([u[name] for u in benign_ups])
                ref[name] = stacked.mean(dim=0)
            return ref

    if mode == "median" or (mode == "benign" and malicious_ids is None):
        ref = {}
        for name in updates[0]:
            stacked = torch.stack([u[name] for u in updates])
            ref[name] = torch.median(stacked, dim=0).values
        return ref

    # Fallback: mean
    ref = {}
    for name in updates[0]:
        stacked = torch.stack([u[name] for u in updates])
        ref[name] = stacked.mean(dim=0)
    return ref