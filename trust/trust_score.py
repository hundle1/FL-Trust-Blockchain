"""
Trust Score Manager — v6 CALIBRATED
=====================================
Calibrated từ diagnose_norms.py:
  Benign norm:   mean=4.24, std=0.08, p97=4.40
  Attacker norm: mean=40.56 (ratio 9.57x)

Changes vs v5:
  - absolute_norm_threshold default = 15.0
    (v5 dùng 6.0 → quá thấp, block cả benign norm ~4.2? Không,
     4.2 < 6.0 → benign pass. Vấn đề thực: alpha=0.45 quá thấp
     + cosine_sim thấp ở early rounds → trust benign rơi < tau)
  - alpha default = 0.7 (ổn định hơn)
  - tau default = 0.40
  - initial_trust default = 0.65
  - warmup_rounds default = 5
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from trust.history_buffer import ClientHistoryManager
from trust.trust_decay import TrustDecay


class TrustScoreManager:
    """
    Manages trust scores for all clients in FL training.

    v6 CALIBRATED — defaults based on diagnose_norms.py measurements:
      Benign norm ~4.2, Attacker norm ~40.6, ratio=9.57x
    """

    def __init__(
        self,
        num_clients: int,
        alpha: float = 0.7,
        tau: float = 0.40,
        initial_trust: float = 0.65,
        enable_decay: bool = True,
        decay_strategy: str = "exponential",
        window_size: int = 10,
        min_trust: float = 0.0,
        max_trust: float = 1.0,
        similarity_weight: float = 0.6,
        idle_decay_rate: float = 0.001,
        enable_norm_penalty: bool = True,
        norm_penalty_threshold: float = 3.0,
        norm_penalty_strength: float = 0.90,
        absolute_norm_threshold: float = 15.0,
        warmup_rounds: int = 5,
    ):
        self.num_clients             = num_clients
        self.alpha                   = alpha
        self.tau                     = tau
        self.initial_trust           = initial_trust
        self.enable_decay            = enable_decay
        self.decay_strategy          = decay_strategy
        self.min_trust               = min_trust
        self.max_trust               = max_trust
        self.similarity_weight       = similarity_weight
        self.idle_decay_rate         = idle_decay_rate
        self.enable_norm_penalty     = enable_norm_penalty
        self.norm_penalty_threshold  = norm_penalty_threshold
        self.norm_penalty_strength   = norm_penalty_strength
        self.absolute_norm_threshold = absolute_norm_threshold
        self.warmup_rounds           = warmup_rounds

        self.trust_scores = np.full(num_clients, initial_trust, dtype=np.float64)
        self.history_manager     = ClientHistoryManager(num_clients, window_size)
        self.last_participated   = np.full(num_clients, -1, dtype=int)
        self.participation_count = np.zeros(num_clients, dtype=int)

        # Norm history — chỉ chứa norms của updates rõ ràng là benign
        self._norm_history: List[float] = []
        self._norm_window_size = 30

        self.update_count  = 0
        self.round_history: List[Dict] = []

    # ------------------------------------------------------------------
    # Core similarity
    # ------------------------------------------------------------------

    def compute_gradient_similarity(
        self,
        client_update: Dict[str, torch.Tensor],
        reference_gradient: Dict[str, torch.Tensor]
    ) -> float:
        """Cosine similarity, mapped [-1,1] → [0,1]."""
        client_flat = torch.cat([v.flatten() for v in client_update.values()])
        ref_flat    = torch.cat([v.flatten() for v in reference_gradient.values()])

        norm_client = torch.norm(client_flat)
        norm_ref    = torch.norm(ref_flat)

        if norm_client < 1e-10 or norm_ref < 1e-10:
            return 0.0

        cosine_sim = torch.dot(client_flat, ref_flat) / (norm_client * norm_ref)
        similarity = (cosine_sim.item() + 1.0) / 2.0
        return float(np.clip(similarity, 0.0, 1.0))

    def _compute_norm(self, update: Dict[str, torch.Tensor]) -> float:
        total = sum(torch.norm(v).item() ** 2 for v in update.values())
        return float(np.sqrt(total))

    # ------------------------------------------------------------------
    # Norm penalty
    # ------------------------------------------------------------------

    def _norm_penalty(self, client_norm: float) -> float:
        """
        Tính norm penalty.

        Early rounds (< 3 obs in history):
          Dùng absolute_norm_threshold để detect outlier ngay.
          Benign ~4.2, threshold=15 → benign luôn penalty=1.0
          Attacker ~40 >> 15 → bị penalize mạnh

        Sau khi có history:
          Dùng median-based penalty với norm_penalty_threshold
        """
        if not self.enable_norm_penalty:
            return 1.0

        if len(self._norm_history) < 3:
            # Early: absolute threshold
            if client_norm > self.absolute_norm_threshold:
                # Attacker ~40, threshold=15 → ratio = 40/15 = 2.67
                # penalty = 1 - 0.9 * min(1, (2.67-1)/2) = 1 - 0.9*0.835 = 0.25
                ratio = client_norm / self.absolute_norm_threshold
                excess = min(1.0, (ratio - 1.0) / 2.0)
                penalty = 1.0 - self.norm_penalty_strength * excess
                return float(np.clip(penalty, 1.0 - self.norm_penalty_strength, 1.0))
            return 1.0

        # History-based: median
        benign_norm_est = float(np.median(self._norm_history))
        if benign_norm_est < 1e-8:
            return 1.0

        ratio = client_norm / benign_norm_est

        if ratio <= self.norm_penalty_threshold:
            return 1.0

        # ratio=3x → no penalty (at threshold)
        # ratio=6x → penalty = 1 - 0.9 * min(1, 3/3) = 0.1
        # ratio=9.57x (attacker) → max penalty = 1-0.9 = 0.1
        excess = (ratio - self.norm_penalty_threshold) / self.norm_penalty_threshold
        penalty = 1.0 - min(self.norm_penalty_strength,
                            self.norm_penalty_strength * excess)
        return float(np.clip(penalty, 1.0 - self.norm_penalty_strength, 1.0))

    # ------------------------------------------------------------------
    # Core trust update
    # ------------------------------------------------------------------

    def update_trust(
        self,
        client_id: int,
        client_update: Dict[str, torch.Tensor],
        reference_gradient: Dict[str, torch.Tensor],
        metrics: Optional[Dict[str, float]] = None,
        round_num: int = 0
    ) -> float:
        old_trust = self.trust_scores[client_id]

        # 1. Cosine similarity
        similarity = self.compute_gradient_similarity(client_update, reference_gradient)

        # 2. Observation = weighted sum of signals
        loss_weight = 1.0 - self.similarity_weight
        if metrics is not None and metrics.get('loss') is not None:
            loss_signal = self._loss_signal(client_id, metrics['loss'])
            observation = self.similarity_weight * similarity + loss_weight * loss_signal
        else:
            observation = similarity

        # 3. Norm penalty
        client_norm = self._compute_norm(client_update)
        penalty     = self._norm_penalty(client_norm)
        observation = observation * penalty

        # 4. Update norm_history — chỉ accept norm < absolute_norm_threshold
        #    Đảm bảo benign norms (~4.2) luôn vào history
        #    Attacker norms (~40) không vào
        if client_norm < self.absolute_norm_threshold:
            self._norm_history.append(client_norm)
            if len(self._norm_history) > self._norm_window_size:
                self._norm_history.pop(0)

        # 5. EMA decay
        if self.enable_decay:
            if self.decay_strategy == "threshold":
                new_trust = TrustDecay.threshold_decay(old_trust, observation)
            elif self.decay_strategy == "adaptive":
                variance  = np.var(
                    self.history_manager.get_gradient_history(client_id) or [0.0]
                )
                new_trust = TrustDecay.adaptive_decay(
                    old_trust, observation, variance, self.alpha
                )
            else:
                new_trust = TrustDecay.exponential_decay(old_trust, observation, self.alpha)
        else:
            new_trust = self.alpha * old_trust + (1 - self.alpha) * observation

        # 6. Clip
        new_trust = float(np.clip(new_trust, self.min_trust, self.max_trust))

        # 7. Store
        self.trust_scores[client_id]        = new_trust
        self.last_participated[client_id]   = round_num
        self.participation_count[client_id] += 1
        self.update_count += 1

        # 8. History
        self.history_manager.add_gradient_similarity(client_id, similarity)
        self.history_manager.add_trust(client_id, new_trust)
        if metrics:
            self.history_manager.add_loss(client_id, metrics.get('loss', 0.0))
            self.history_manager.add_accuracy(client_id, metrics.get('accuracy', 0.0))

        return new_trust

    def _loss_signal(self, client_id: int, loss: Optional[float]) -> float:
        if loss is None:
            return 0.5
        history = self.history_manager.get_loss_history(client_id)
        if not history:
            return 0.5
        avg_loss = np.mean(history)
        if avg_loss < 1e-10:
            return 0.5
        ratio  = loss / avg_loss
        signal = 1.0 / (1.0 + max(0.0, ratio - 1.0))
        return float(np.clip(signal, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Idle decay
    # ------------------------------------------------------------------

    def apply_idle_decay(
        self,
        active_client_ids: List[int],
        round_num: int,
        decay_rate: float = None
    ):
        if not self.enable_decay:
            return
        rate       = decay_rate if decay_rate is not None else self.idle_decay_rate
        active_set = set(active_client_ids)
        for cid in range(self.num_clients):
            if cid not in active_set:
                self.trust_scores[cid] = TrustDecay.linear_decay(
                    self.trust_scores[cid], rate
                )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_trust_score(self, client_id: int) -> float:
        return float(self.trust_scores[client_id])

    def get_all_trust_scores(self) -> np.ndarray:
        return self.trust_scores.copy()

    def get_trusted_clients(
        self,
        client_ids: List[int],
        round_num: int = 999
    ) -> List[int]:
        """Bypass filter trong warmup_rounds đầu."""
        if round_num < self.warmup_rounds:
            return list(client_ids)
        return [cid for cid in client_ids if self.trust_scores[cid] >= self.tau]

    def get_trust_weights(self, client_ids: List[int]) -> List[float]:
        scores = np.array([self.trust_scores[cid] for cid in client_ids])
        total  = scores.sum()
        if total < 1e-10:
            return [1.0 / len(client_ids)] * len(client_ids)
        return (scores / total).tolist()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict:
        scores       = self.trust_scores
        trusted_mask = scores >= self.tau
        return {
            'mean_trust':         float(np.mean(scores)),
            'std_trust':          float(np.std(scores)),
            'min_trust':          float(np.min(scores)),
            'max_trust':          float(np.max(scores)),
            'num_trusted':        int(trusted_mask.sum()),
            'num_untrusted':      int((~trusted_mask).sum()),
            'trusted_ratio':      float(trusted_mask.sum() / self.num_clients),
            'tau':                self.tau,
            'alpha':              self.alpha,
            'total_updates':      self.update_count,
            'norm_history_size':  len(self._norm_history),
            'benign_norm_est':    float(np.median(self._norm_history))
                                  if self._norm_history else 0.0,
        }

    def get_trust_separation(
        self,
        benign_ids: List[int],
        malicious_ids: List[int]
    ) -> Dict[str, float]:
        benign_scores    = [self.trust_scores[i] for i in benign_ids
                            if i < self.num_clients]
        malicious_scores = [self.trust_scores[i] for i in malicious_ids
                            if i < self.num_clients]
        if not benign_scores or not malicious_scores:
            return {'separation': 0.0, 'avg_benign': 0.0, 'avg_malicious': 0.0}
        avg_b = float(np.mean(benign_scores))
        avg_m = float(np.mean(malicious_scores))
        return {
            'separation':    avg_b - avg_m,
            'avg_benign':    avg_b,
            'avg_malicious': avg_m,
            'benign_std':    float(np.std(benign_scores)),
            'malicious_std': float(np.std(malicious_scores))
        }

    def get_client_history(self, client_id: int) -> Dict:
        return self.history_manager.get_statistics(client_id)

    def reset_client(self, client_id: int):
        self.trust_scores[client_id]        = self.initial_trust
        self.participation_count[client_id] = 0
        self.history_manager.clear_client(client_id)

    def print_trust_table(self, top_n: int = 20):
        print(f"\n{'='*55}")
        print(f"{'Client':<10} {'Trust':>10} {'Status':>10} {'Rounds':>10}")
        print(f"{'-'*55}")
        indices = np.argsort(self.trust_scores)[::-1][:top_n]
        for cid in indices:
            score  = self.trust_scores[cid]
            status = "TRUSTED" if score >= self.tau else "BLOCKED"
            rounds = self.participation_count[cid]
            print(f"{cid:<10} {score:>10.4f} {status:>10} {rounds:>10}")
        print(f"{'='*55}")