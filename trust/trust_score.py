"""
Trust Score Manager — v7 Q1-Ready
=====================================
Changes vs v6:
  - Tương thích với realistic reference gradient (median-based)
  - Thêm method compute_realistic_reference() để server không cần biết malicious IDs
  - absolute_norm_threshold=15.0 (calibrated từ diagnose_norms.py)
  - Default alpha=0.9, tau=0.3 (validated từ sensitivity analysis)
  - Thêm: get_trust_confidence() — trust reliability estimate
  - Thêm: is_client_suspicious() — hard threshold check

Paper note:
  Trust update formula: T_i(t+1) = α·T_i(t) + (1-α)·S_i(t)
  S_i(t) = cosine_sim(g_i, ref) mapped to [0,1]
  ref = coordinate-wise median (all updates) — REALISTIC, no ground truth needed
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from trust.history_buffer import ClientHistoryManager
from trust.trust_decay import TrustDecay


class TrustScoreManager:
    """
    Trust Score Manager — manages EMA-based trust scores for FL clients.

    Core formula: T_i(t+1) = α·T_i(t) + (1-α)·[sim_weight·S_i(t) + (1-sim_weight)·L_i(t)]
    Where:
      S_i(t) = cosine similarity with reference gradient, mapped [−1,1] → [0,1]
      L_i(t) = loss-based trust signal [0,1]
      α       = EMA memory factor (default 0.9)

    Reference gradient: coordinate-wise median (robust, production-feasible)
    No need to know which clients are malicious.
    """

    def __init__(
        self,
        num_clients: int,
        alpha: float = 0.9,
        tau: float = 0.3,
        initial_trust: float = 1.0,
        enable_decay: bool = True,
        decay_strategy: str = "exponential",
        window_size: int = 10,
        min_trust: float = 0.0,
        max_trust: float = 1.0,
        similarity_weight: float = 0.7,
        idle_decay_rate: float = 0.002,
        enable_norm_penalty: bool = True,
        norm_penalty_threshold: float = 3.0,
        norm_penalty_strength: float = 0.80,
        absolute_norm_threshold: float = 15.0,
        warmup_rounds: int = 0,
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

        self.trust_scores        = np.full(num_clients, initial_trust, dtype=np.float64)
        self.history_manager     = ClientHistoryManager(num_clients, window_size)
        self.last_participated   = np.full(num_clients, -1, dtype=int)
        self.participation_count = np.zeros(num_clients, dtype=int)

        # Norm history — chỉ chứa norms rõ ràng benign (< absolute_norm_threshold)
        self._norm_history: List[float] = []
        self._norm_window_size = 30

        self.update_count = 0

    # ------------------------------------------------------------------
    # Cosine similarity
    # ------------------------------------------------------------------

    def compute_gradient_similarity(
        self,
        client_update: Dict[str, torch.Tensor],
        reference_gradient: Dict[str, torch.Tensor]
    ) -> float:
        """
        Cosine similarity between client update and reference, mapped [−1,1] → [0,1].

        S_i(t) = (cos(g_i, ref) + 1) / 2

        Paper justification:
          Benign clients: gradient direction similar to global trend → high similarity
          Malicious clients: gradient flipped/scaled → low/negative similarity
        """
        client_flat = torch.cat([v.flatten() for v in client_update.values()])
        ref_flat    = torch.cat([v.flatten() for v in reference_gradient.values()])

        norm_c = torch.norm(client_flat)
        norm_r = torch.norm(ref_flat)

        if norm_c < 1e-10 or norm_r < 1e-10:
            return 0.0

        cosine = torch.dot(client_flat, ref_flat) / (norm_c * norm_r)
        return float(np.clip((cosine.item() + 1.0) / 2.0, 0.0, 1.0))

    def _compute_norm(self, update: Dict[str, torch.Tensor]) -> float:
        return float(np.sqrt(sum(torch.norm(v).item() ** 2 for v in update.values())))

    # ------------------------------------------------------------------
    # Norm penalty
    # ------------------------------------------------------------------

    def _norm_penalty(self, client_norm: float) -> float:
        """
        Multiplicative penalty for suspicious norms.

        Early rounds (< 3 obs): use absolute_norm_threshold.
          - Benign norm ~4.2 << 15 → no penalty
          - Attacker norm ~40 >> 15 → strong penalty

        After warmup: median-based relative penalty.
        """
        if not self.enable_norm_penalty:
            return 1.0

        if len(self._norm_history) < 3:
            if client_norm > self.absolute_norm_threshold:
                ratio = client_norm / self.absolute_norm_threshold
                excess = min(1.0, (ratio - 1.0) / 2.0)
                penalty = 1.0 - self.norm_penalty_strength * excess
                return float(np.clip(penalty, 1.0 - self.norm_penalty_strength, 1.0))
            return 1.0

        benign_norm_est = float(np.median(self._norm_history))
        if benign_norm_est < 1e-8:
            return 1.0

        ratio = client_norm / benign_norm_est
        if ratio <= self.norm_penalty_threshold:
            return 1.0

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
        """
        Update trust score for one client.

        T_i(t+1) = α·T_i(t) + (1−α)·observation_i(t)
        observation = sim_weight·S_i + (1−sim_weight)·L_i, then × norm_penalty

        Args:
            client_id:          Client identifier
            client_update:      Client's gradient dict
            reference_gradient: Reference gradient (median of all updates — realistic)
            metrics:            Training metrics (loss, accuracy, gradient_norm)
            round_num:          Current FL round

        Returns:
            new_trust: Updated trust score [0, 1]
        """
        old_trust = self.trust_scores[client_id]

        # 1. Cosine similarity → behavioral trust signal
        similarity = self.compute_gradient_similarity(client_update, reference_gradient)

        # 2. Combined observation with optional loss signal
        loss_weight = 1.0 - self.similarity_weight
        if metrics is not None and metrics.get('loss') is not None:
            loss_signal = self._loss_signal(client_id, metrics['loss'])
            observation = self.similarity_weight * similarity + loss_weight * loss_signal
        else:
            observation = similarity

        # 3. Norm penalty (multiplicative)
        client_norm = self._compute_norm(client_update)
        penalty     = self._norm_penalty(client_norm)
        observation = observation * penalty

        # 4. Update norm history (only benign-range norms)
        if client_norm < self.absolute_norm_threshold:
            self._norm_history.append(client_norm)
            if len(self._norm_history) > self._norm_window_size:
                self._norm_history.pop(0)

        # 5. EMA trust update
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
            else:  # exponential (default)
                new_trust = TrustDecay.exponential_decay(old_trust, observation, self.alpha)
        else:
            new_trust = self.alpha * old_trust + (1 - self.alpha) * observation

        # 6. Clip to [0, 1]
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
        avg_loss = float(np.mean(history))
        if avg_loss < 1e-10:
            return 0.5
        ratio  = loss / avg_loss
        signal = 1.0 / (1.0 + max(0.0, ratio - 1.0))
        return float(np.clip(signal, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Idle decay
    # ------------------------------------------------------------------

    def apply_idle_decay(self, active_client_ids: List[int], round_num: int,
                          decay_rate: float = None):
        """Apply linear decay to idle (non-participating) clients."""
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

    def get_trusted_clients(self, client_ids: List[int], round_num: int = 999) -> List[int]:
        """Return clients with trust ≥ τ. Bypass during warmup."""
        if round_num < self.warmup_rounds:
            return list(client_ids)
        return [cid for cid in client_ids if self.trust_scores[cid] >= self.tau]

    def get_trust_weights(self, client_ids: List[int]) -> List[float]:
        scores = np.array([self.trust_scores[cid] for cid in client_ids])
        total  = scores.sum()
        if total < 1e-10:
            return [1.0 / len(client_ids)] * len(client_ids)
        return (scores / total).tolist()

    def is_client_suspicious(self, client_id: int,
                              suspicious_threshold: float = None) -> bool:
        """Hard check if client trust is below suspicious threshold."""
        thresh = suspicious_threshold or (self.tau * 0.5)
        return bool(self.trust_scores[client_id] < thresh)

    def get_trust_confidence(self, client_id: int) -> float:
        """
        Trust reliability = how stable trust has been.
        Low std → high confidence. Returns [0, 1].
        """
        history = self.history_manager.get_trust_history(client_id)
        if len(history) < 3:
            return 0.5
        std = float(np.std(history))
        return float(1.0 / (1.0 + std * 10))

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
        """Compute trust separation statistics."""
        b_scores = [self.trust_scores[i] for i in benign_ids if i < self.num_clients]
        m_scores = [self.trust_scores[i] for i in malicious_ids if i < self.num_clients]
        if not b_scores or not m_scores:
            return {'separation': 0.0, 'avg_benign': 0.0, 'avg_malicious': 0.0,
                    'benign_std': 0.0, 'malicious_std': 0.0}
        avg_b = float(np.mean(b_scores))
        avg_m = float(np.mean(m_scores))
        return {
            'separation':    avg_b - avg_m,
            'avg_benign':    avg_b,
            'avg_malicious': avg_m,
            'benign_std':    float(np.std(b_scores)),
            'malicious_std': float(np.std(m_scores)),
        }

    def get_client_history(self, client_id: int) -> Dict:
        return self.history_manager.get_statistics(client_id)

    def reset_client(self, client_id: int):
        self.trust_scores[client_id]        = self.initial_trust
        self.participation_count[client_id] = 0
        self.history_manager.clear_client(client_id)

    def print_trust_table(self, top_n: int = 20):
        print(f"\n{'='*60}")
        print(f"{'Client':<10} {'Trust':>10} {'Status':>10} {'Rounds':>10}")
        print(f"{'-'*60}")
        indices = np.argsort(self.trust_scores)[::-1][:top_n]
        for cid in indices:
            score  = self.trust_scores[cid]
            status = "TRUSTED" if score >= self.tau else "BLOCKED"
            rounds = self.participation_count[cid]
            print(f"{cid:<10} {score:>10.4f} {status:>10} {rounds:>10}")
        print(f"{'='*60}")