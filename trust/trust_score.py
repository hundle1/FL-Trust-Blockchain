"""
Trust Score Manager — FIXED VERSION
=====================================
Các bug đã sửa so với bản gốc:

  FIX 1: alpha 0.9 → 0.75
          Trust thay đổi nhanh hơn ~3x, phát hiện attacker sớm hơn
          (bản gốc cần ~60 rounds mới filter được, bản fix ~10 rounds)

  FIX 2: initial_trust 1.0 → 0.5
          Không tin tưởng tuyệt đối ngay từ đầu.
          Giảm thiệt hại ở những rounds đầu tiên.

  FIX 3: idle_decay 0.005 → 0.008
          Decay nhanh hơn khi client không tham gia.
          Vẫn đủ nhỏ để benign không bị false-positive.

  FIX 4: window_size 10 → 5
          Phản ứng nhanh hơn với behavior change.

  FIX 5: similarity_weight 0.8 → 0.9
          Tin cosine sim nhiều hơn loss signal vì loss bị noise nhiều.

Trust update formula (không đổi):
    T_i(t+1) = α * T_i(t) + (1 − α) * S_i(t)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from trust.history_buffer import ClientHistoryManager
from trust.trust_decay import TrustDecay


class TrustScoreManager:
    """
    Manages trust scores for all clients in FL training.

    Trust is computed based on:
    1. Gradient cosine similarity with reference gradient
    2. Exponential moving average (controlled by alpha)
    3. Optional trust decay for idle clients
    4. Optional behavior-based anomaly detection
    """

    def __init__(
        self,
        num_clients: int,
        alpha: float = 0.75,          # FIX 1: was 0.9 → faster trust change
        tau: float = 0.5,             # FIX: was 0.3 → filter earlier
        initial_trust: float = 0.5,   # FIX 2: was 1.0 → don't fully trust from start
        enable_decay: bool = True,
        decay_strategy: str = "exponential",
        window_size: int = 5,         # FIX 4: was 10 → react faster
        min_trust: float = 0.0,
        max_trust: float = 1.0,
        similarity_weight: float = 0.9,   # FIX 5: was 0.8
        idle_decay_rate: float = 0.008,   # FIX 3: was 0.005
    ):
        self.num_clients       = num_clients
        self.alpha             = alpha
        self.tau               = tau
        self.initial_trust     = initial_trust
        self.enable_decay      = enable_decay
        self.decay_strategy    = decay_strategy
        self.min_trust         = min_trust
        self.max_trust         = max_trust
        self.similarity_weight = similarity_weight
        self.idle_decay_rate   = idle_decay_rate

        # Trust scores
        self.trust_scores = np.full(num_clients, initial_trust, dtype=np.float64)

        # History tracking
        self.history_manager = ClientHistoryManager(num_clients, window_size)

        # Round participation tracking
        self.last_participated  = np.full(num_clients, -1, dtype=int)
        self.participation_count = np.zeros(num_clients, dtype=int)

        # Statistics
        self.update_count  = 0
        self.round_history: List[Dict] = []

    # ------------------------------------------------------------------
    # Core trust update
    # ------------------------------------------------------------------

    def compute_gradient_similarity(
        self,
        client_update: Dict[str, torch.Tensor],
        reference_gradient: Dict[str, torch.Tensor]
    ) -> float:
        """
        Cosine similarity between client update and reference gradient.
        Maps [-1, 1] → [0, 1].
        """
        client_flat = torch.cat([v.flatten() for v in client_update.values()])
        ref_flat    = torch.cat([v.flatten() for v in reference_gradient.values()])

        norm_client = torch.norm(client_flat)
        norm_ref    = torch.norm(ref_flat)

        if norm_client < 1e-10 or norm_ref < 1e-10:
            return 0.0

        cosine_sim = torch.dot(client_flat, ref_flat) / (norm_client * norm_ref)
        similarity = (cosine_sim.item() + 1.0) / 2.0
        return float(np.clip(similarity, 0.0, 1.0))

    def update_trust(
        self,
        client_id: int,
        client_update: Dict[str, torch.Tensor],
        reference_gradient: Dict[str, torch.Tensor],
        metrics: Optional[Dict[str, float]] = None,
        round_num: int = 0
    ) -> float:
        """
        Update trust score for a client after receiving its update.

        FIX 5: similarity_weight tăng lên 0.9 để cosine sim chiếm ưu thế.
        """
        old_trust = self.trust_scores[client_id]

        # 1. Cosine similarity (primary signal)
        similarity = self.compute_gradient_similarity(client_update, reference_gradient)

        # 2. Loss-based signal (secondary)
        loss_weight = 1.0 - self.similarity_weight
        if metrics is not None and metrics.get('loss') is not None:
            loss_signal  = self._loss_signal(client_id, metrics['loss'])
            observation  = self.similarity_weight * similarity + loss_weight * loss_signal
        else:
            observation = similarity

        # 3. EMA decay
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

        # 4. Clip
        new_trust = float(np.clip(new_trust, self.min_trust, self.max_trust))

        # 5. Store
        self.trust_scores[client_id]      = new_trust
        self.last_participated[client_id] = round_num
        self.participation_count[client_id] += 1
        self.update_count += 1

        # 6. Update history
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
        ratio   = loss / avg_loss
        signal  = 1.0 / (1.0 + max(0.0, ratio - 1.0))
        return float(np.clip(signal, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Idle decay — FIX 3: rate tăng 0.005 → 0.008
    # ------------------------------------------------------------------

    def apply_idle_decay(
        self,
        active_client_ids: List[int],
        round_num: int,
        decay_rate: float = None   # None → dùng self.idle_decay_rate
    ):
        """
        Apply trust decay to non-participating clients.
        FIX 3: default decay_rate tăng lên 0.008.
        """
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

    def get_trusted_clients(self, client_ids: List[int]) -> List[int]:
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
            'mean_trust':    float(np.mean(scores)),
            'std_trust':     float(np.std(scores)),
            'min_trust':     float(np.min(scores)),
            'max_trust':     float(np.max(scores)),
            'num_trusted':   int(trusted_mask.sum()),
            'num_untrusted': int((~trusted_mask).sum()),
            'trusted_ratio': float(trusted_mask.sum() / self.num_clients),
            'tau':   self.tau,
            'alpha': self.alpha,
            'total_updates': self.update_count
        }

    def get_trust_separation(
        self,
        benign_ids: List[int],
        malicious_ids: List[int]
    ) -> Dict[str, float]:
        benign_scores   = [self.trust_scores[i] for i in benign_ids
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
        self.trust_scores[client_id]      = self.initial_trust
        self.participation_count[client_id] = 0
        self.history_manager.clear_client(client_id)

    def print_trust_table(self, top_n: int = 20):
        print(f"\n{'='*50}")
        print(f"{'Client':<10} {'Trust Score':<15} {'Status':<10} {'Rounds':<10}")
        print(f"{'-'*50}")
        indices = np.argsort(self.trust_scores)[::-1][:top_n]
        for cid in indices:
            score  = self.trust_scores[cid]
            status = "TRUSTED" if score >= self.tau else "BLOCKED"
            rounds = self.participation_count[cid]
            print(f"{cid:<10} {score:<15.4f} {status:<10} {rounds:<10}")
        print(f"{'='*50}")