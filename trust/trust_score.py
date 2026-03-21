"""
Trust Score Manager — v8 Q1-Target
=====================================
3 cải tiến chính để đạt Q1 targets:

FIX 1 — Direction Consistency (norm-tuned sep: 0.047 → ≥0.10)
  Norm-tuned attack norm nhỏ (~6 ≈ benign) nhưng HƯỚNG gradient luôn ngược.
  Thêm direction_weight × direction_consistency vào observation.
  direction_consistency = cosine(current_grad, mean_of_own_history) → [0,1]
  Benign: hướng ổn định → consistency cao (~0.7)
  Norm-tuned: hướng luôn bị flip → consistency thấp (~0.2)

FIX 2 — Temporal Smoothing (intermittent det: 5% → ≥20%)
  Intermittent: cứ xen kẽ attack/clean → trust oscillate, không bao giờ drop đủ.
  smooth_obs = β × current_obs + (1-β) × mean(recent_window)
  Rolling mean kéo observation xuống bền vững → trust dần giảm dưới tau.

FIX 3 — Sustained Low Signal Penalty (adaptive det: 0% → ≥10%)
  Adaptive attacker giữ trust ≈ 0.53 bằng cách chỉ attack khi trust cao.
  Cosine similarity khi attack ≈ 0.3, khi không attack ≈ 0.75.
  Nếu rolling_sim < 0.45 liên tục 3 round → penalty thêm 0.15.
  Điều này nudge trust xuống dưới tau=0.25.

Paper formula (v8):
  obs_i(t) = w_sim × cosine_sim(g_i, ref)
           + w_dir × direction_consistency(g_i, hist_i)
           + w_loss × loss_signal(loss_i)
  obs_i(t) ×= norm_penalty(norm_i) × sustained_penalty(sim_i)
  smooth_obs = β × obs_i + (1-β) × mean(obs_window)
  T_i(t+1) = α × T_i(t) + (1-α) × smooth_obs
"""

import torch
import numpy as np
from collections import deque
from typing import Dict, List, Optional
from trust.history_buffer import ClientHistoryManager
from trust.trust_decay import TrustDecay


class TrustScoreManager:
    """
    Trust Score Manager v8 — Q1-Target.

    Enhanced 3-signal observation:
      obs = w_sim × cosine_sim      ← direction vs reference
          + w_dir × dir_consistency ← direction vs own history [FIX1]
          + w_loss × loss_signal    ← loss anomaly
      obs ×= norm_penalty × sustained_penalty
      smooth_obs = temporal_smooth(obs)   [FIX2]
      T_new = α × T_old + (1-α) × smooth_obs
    """

    def __init__(
        self,
        num_clients: int,
        alpha: float = 0.9,
        tau: float = 0.25,               # FIX3: 0.30→0.25 for adaptive det≥10%
        initial_trust: float = 1.0,
        enable_decay: bool = True,
        decay_strategy: str = "exponential",
        window_size: int = 10,
        min_trust: float = 0.0,
        max_trust: float = 1.0,
        # Signal weights (normalized internally to sum=1)
        similarity_weight: float = 0.55,  # cosine similarity
        direction_weight: float = 0.25,   # FIX1: direction consistency
        loss_weight: float = 0.20,        # loss signal
        # Temporal smoothing (FIX2)
        smoothing_beta: float = 0.7,      # weight: current vs window mean
        smoothing_window: int = 5,
        # Idle decay
        idle_decay_rate: float = 0.002,
        # Norm penalty
        enable_norm_penalty: bool = True,
        norm_penalty_threshold: float = 3.0,
        norm_penalty_strength: float = 0.80,
        absolute_norm_threshold: float = 15.0,
        # Sustained low signal penalty (FIX3)
        enable_sustained_penalty: bool = True,
        sustained_threshold: float = 0.45,
        sustained_window: int = 3,
        sustained_penalty_strength: float = 0.15,
        # Warmup
        warmup_rounds: int = 0,
    ):
        self.num_clients               = num_clients
        self.alpha                     = alpha
        self.tau                       = tau
        self.initial_trust             = initial_trust
        self.enable_decay              = enable_decay
        self.decay_strategy            = decay_strategy
        self.min_trust                 = min_trust
        self.max_trust                 = max_trust
        self.idle_decay_rate           = idle_decay_rate
        self.enable_norm_penalty       = enable_norm_penalty
        self.norm_penalty_threshold    = norm_penalty_threshold
        self.norm_penalty_strength     = norm_penalty_strength
        self.absolute_norm_threshold   = absolute_norm_threshold
        self.enable_sustained_penalty  = enable_sustained_penalty
        self.sustained_threshold       = sustained_threshold
        self.sustained_window          = sustained_window
        self.sustained_penalty_strength = sustained_penalty_strength
        self.smoothing_beta            = smoothing_beta
        self.smoothing_window          = smoothing_window
        self.warmup_rounds             = warmup_rounds

        # Normalize signal weights to sum=1
        _total = similarity_weight + direction_weight + loss_weight
        self._w_sim  = similarity_weight / _total
        self._w_dir  = direction_weight  / _total
        self._w_loss = loss_weight       / _total

        # Core state
        self.trust_scores        = np.full(num_clients, initial_trust, dtype=np.float64)
        self.history_manager     = ClientHistoryManager(num_clients, window_size)
        self.last_participated   = np.full(num_clients, -1, dtype=int)
        self.participation_count = np.zeros(num_clients, dtype=int)
        self.update_count        = 0

        # Norm history (benign-range norms only, for median-based penalty)
        self._norm_history: List[float] = []
        self._norm_window_size = 30

        # FIX1: Per-client gradient direction history (normalized, last 5)
        self._grad_dir_history: Dict[int, deque] = {
            i: deque(maxlen=5) for i in range(num_clients)
        }

        # FIX2: Per-client rolling observation window for temporal smoothing
        self._obs_window: Dict[int, deque] = {
            i: deque(maxlen=smoothing_window) for i in range(num_clients)
        }

        # FIX3: Per-client rolling cosine similarity for sustained penalty
        self._sim_window: Dict[int, deque] = {
            i: deque(maxlen=sustained_window) for i in range(num_clients)
        }

    # ──────────────────────────────────────────────────────────────────
    # Signal 1: Cosine similarity vs reference
    # ──────────────────────────────────────────────────────────────────

    def compute_gradient_similarity(
        self,
        client_update: Dict[str, torch.Tensor],
        reference_gradient: Dict[str, torch.Tensor]
    ) -> float:
        """
        Cosine similarity between client gradient and reference, mapped [-1,1]→[0,1].
        Reference = coordinate-wise median of all updates (realistic).
        """
        c_flat = torch.cat([v.flatten() for v in client_update.values()])
        r_flat = torch.cat([v.flatten() for v in reference_gradient.values()])
        nc = torch.norm(c_flat)
        nr = torch.norm(r_flat)
        if nc < 1e-10 or nr < 1e-10:
            return 0.0
        cos = torch.dot(c_flat, r_flat) / (nc * nr)
        return float(np.clip((cos.item() + 1.0) / 2.0, 0.0, 1.0))

    # ──────────────────────────────────────────────────────────────────
    # Signal 2: Direction consistency vs own history [FIX1]
    # ──────────────────────────────────────────────────────────────────

    def compute_direction_consistency(
        self,
        client_id: int,
        client_update: Dict[str, torch.Tensor]
    ) -> float:
        """
        Cosine similarity between current gradient direction and
        mean of this client's own recent gradient directions.

        WHY THIS CATCHES NORM-TUNED:
          Norm-tuned rescales to small norm but direction is always flipped.
          Benign: stable direction over rounds → high consistency ~0.7
          Norm-tuned: direction always opposite to own history → low ~0.2

        Returns 0.5 if insufficient history (neutral).
        """
        history = self._grad_dir_history[client_id]
        if len(history) < 2:
            return 0.5

        c_flat = torch.cat([v.flatten() for v in client_update.values()])
        nc = torch.norm(c_flat)
        if nc < 1e-10:
            return 0.5

        stacked  = torch.stack(list(history))
        mean_dir = stacked.mean(dim=0)
        nm = torch.norm(mean_dir)
        if nm < 1e-10:
            return 0.5

        cos = torch.dot(c_flat / nc, mean_dir / nm)
        return float(np.clip((cos.item() + 1.0) / 2.0, 0.0, 1.0))

    def _store_direction(self, client_id: int, client_update: Dict[str, torch.Tensor]):
        """Store normalized gradient direction for FIX1."""
        flat = torch.cat([v.flatten() for v in client_update.values()])
        n = torch.norm(flat)
        if n > 1e-10:
            self._grad_dir_history[client_id].append(
                (flat / n).detach().cpu()
            )

    # ──────────────────────────────────────────────────────────────────
    # Signal 3: Loss signal
    # ──────────────────────────────────────────────────────────────────

    def _loss_signal(self, client_id: int, loss: Optional[float]) -> float:
        """Loss-based trust: high loss vs own history → low signal."""
        if loss is None:
            return 0.5
        history = self.history_manager.get_loss_history(client_id)
        if not history:
            return 0.5
        avg = float(np.mean(history))
        if avg < 1e-10:
            return 0.5
        ratio = loss / avg
        return float(np.clip(1.0 / (1.0 + max(0.0, ratio - 1.0)), 0.0, 1.0))

    # ──────────────────────────────────────────────────────────────────
    # Norm penalty
    # ──────────────────────────────────────────────────────────────────

    def _compute_norm(self, update: Dict[str, torch.Tensor]) -> float:
        return float(np.sqrt(sum(torch.norm(v).item() ** 2 for v in update.values())))

    def _norm_penalty(self, client_norm: float) -> float:
        """
        Multiplicative norm penalty.
        Early (<3 history): absolute threshold (15.0).
          Benign ~4.2 << 15 → no penalty.
          Attacker ~40 >> 15 → strong penalty.
        Later: median-based relative penalty.
        """
        if not self.enable_norm_penalty:
            return 1.0
        if len(self._norm_history) < 3:
            if client_norm > self.absolute_norm_threshold:
                ratio  = client_norm / self.absolute_norm_threshold
                excess = min(1.0, (ratio - 1.0) / 2.0)
                return float(np.clip(
                    1.0 - self.norm_penalty_strength * excess,
                    1.0 - self.norm_penalty_strength, 1.0
                ))
            return 1.0
        benign_est = float(np.median(self._norm_history))
        if benign_est < 1e-8:
            return 1.0
        ratio = client_norm / benign_est
        if ratio <= self.norm_penalty_threshold:
            return 1.0
        excess  = (ratio - self.norm_penalty_threshold) / self.norm_penalty_threshold
        penalty = 1.0 - min(self.norm_penalty_strength,
                            self.norm_penalty_strength * excess)
        return float(np.clip(penalty, 1.0 - self.norm_penalty_strength, 1.0))

    # ──────────────────────────────────────────────────────────────────
    # Temporal smoothing [FIX2 — intermittent detection]
    # ──────────────────────────────────────────────────────────────────

    def _temporal_smooth(self, client_id: int, current_obs: float) -> float:
        """
        smooth = beta×current + (1-beta)×mean(recent_window)

        WHY THIS CATCHES INTERMITTENT:
          50% attack rate: poison (obs≈0.1) alternates clean (obs≈0.8).
          Without: trust oscillates 0.3↔0.7, never drops below tau.
          With: rolling mean drags observation to ~0.45 persistently.
          trust eventually falls below tau=0.25 after enough attack rounds.
        """
        win = self._obs_window[client_id]
        if len(win) == 0:
            smoothed = current_obs
        else:
            win_mean = float(np.mean(list(win)))
            smoothed = self.smoothing_beta * current_obs + (1 - self.smoothing_beta) * win_mean
        win.append(current_obs)
        return float(np.clip(smoothed, 0.0, 1.0))

    # ──────────────────────────────────────────────────────────────────
    # Sustained low signal penalty [FIX3 — adaptive detection]
    # ──────────────────────────────────────────────────────────────────

    def _sustained_penalty(self, client_id: int, current_sim: float) -> float:
        """
        If cosine_sim below sustained_threshold for ALL of last sustained_window
        rounds → apply extra multiplicative penalty.

        WHY THIS CATCHES ADAPTIVE:
          Adaptive equilibrium: trust ≈ 0.53 from live logs.
          Cosine sim when attacking  ≈ 0.3 (below 0.45 threshold).
          Cosine sim when dormant    ≈ 0.75 (above threshold).
          After 3 consecutive below-threshold → penalty 0.15 applied.
          Combined with temporal smoothing → trust nudged below tau=0.25.
        """
        if not self.enable_sustained_penalty:
            return 1.0
        win = self._sim_window[client_id]
        win.append(current_sim)
        if len(win) < self.sustained_window:
            return 1.0
        if all(s < self.sustained_threshold for s in win):
            return 1.0 - self.sustained_penalty_strength
        return 1.0

    # ──────────────────────────────────────────────────────────────────
    # Core trust update
    # ──────────────────────────────────────────────────────────────────

    def update_trust(
        self,
        client_id: int,
        client_update: Dict[str, torch.Tensor],
        reference_gradient: Dict[str, torch.Tensor],
        metrics: Optional[Dict[str, float]] = None,
        round_num: int = 0
    ) -> float:
        """
        Full v8 trust update formula:
          obs = w_sim·S_cos + w_dir·S_dir + w_loss·S_loss
          obs ×= norm_penalty × sustained_penalty
          smooth = β·obs + (1-β)·mean(obs_window)
          T_new = α·T_old + (1-α)·smooth
        """
        old_trust = self.trust_scores[client_id]

        # Signal 1: cosine similarity vs reference
        s_cos = self.compute_gradient_similarity(client_update, reference_gradient)

        # Signal 2: direction consistency vs own history [FIX1]
        s_dir = self.compute_direction_consistency(client_id, client_update)

        # Signal 3: loss signal
        s_loss = (self._loss_signal(client_id, metrics['loss'])
                  if metrics and metrics.get('loss') is not None else 0.5)

        # Combine
        obs = self._w_sim * s_cos + self._w_dir * s_dir + self._w_loss * s_loss

        # Norm penalty
        client_norm = self._compute_norm(client_update)
        obs *= self._norm_penalty(client_norm)

        # Sustained low signal penalty [FIX3]
        obs *= self._sustained_penalty(client_id, s_cos)

        # Update norm history (benign-range only)
        if client_norm < self.absolute_norm_threshold:
            self._norm_history.append(client_norm)
            if len(self._norm_history) > self._norm_window_size:
                self._norm_history.pop(0)

        # Update direction history [FIX1]
        self._store_direction(client_id, client_update)

        # Temporal smoothing [FIX2]
        smooth_obs = self._temporal_smooth(client_id, obs)

        # EMA trust update
        if self.enable_decay:
            if self.decay_strategy == "threshold":
                new_trust = TrustDecay.threshold_decay(old_trust, smooth_obs)
            elif self.decay_strategy == "adaptive":
                variance  = np.var(
                    self.history_manager.get_gradient_history(client_id) or [0.0]
                )
                new_trust = TrustDecay.adaptive_decay(
                    old_trust, smooth_obs, variance, self.alpha
                )
            else:
                new_trust = TrustDecay.exponential_decay(old_trust, smooth_obs, self.alpha)
        else:
            new_trust = self.alpha * old_trust + (1 - self.alpha) * smooth_obs

        new_trust = float(np.clip(new_trust, self.min_trust, self.max_trust))

        # Persist state
        self.trust_scores[client_id]        = new_trust
        self.last_participated[client_id]   = round_num
        self.participation_count[client_id] += 1
        self.update_count += 1

        self.history_manager.add_gradient_similarity(client_id, s_cos)
        self.history_manager.add_trust(client_id, new_trust)
        if metrics:
            self.history_manager.add_loss(client_id, metrics.get('loss', 0.0))
            self.history_manager.add_accuracy(client_id, metrics.get('accuracy', 0.0))

        return new_trust

    # ──────────────────────────────────────────────────────────────────
    # Idle decay
    # ──────────────────────────────────────────────────────────────────

    def apply_idle_decay(
        self,
        active_client_ids: List[int],
        round_num: int,
        decay_rate: float = None
    ):
        """Linear decay for clients not participating this round."""
        if not self.enable_decay:
            return
        rate       = decay_rate if decay_rate is not None else self.idle_decay_rate
        active_set = set(active_client_ids)
        for cid in range(self.num_clients):
            if cid not in active_set:
                self.trust_scores[cid] = TrustDecay.linear_decay(
                    self.trust_scores[cid], rate
                )

    # ──────────────────────────────────────────────────────────────────
    # Querying
    # ──────────────────────────────────────────────────────────────────

    def get_trust_score(self, client_id: int) -> float:
        return float(self.trust_scores[client_id])

    def get_all_trust_scores(self) -> np.ndarray:
        return self.trust_scores.copy()

    def get_trusted_clients(
        self,
        client_ids: List[int],
        round_num: int = 999
    ) -> List[int]:
        """Return clients with trust >= tau. Bypass during warmup."""
        if round_num < self.warmup_rounds:
            return list(client_ids)
        return [cid for cid in client_ids if self.trust_scores[cid] >= self.tau]

    def get_trust_weights(self, client_ids: List[int]) -> List[float]:
        scores = np.array([self.trust_scores[cid] for cid in client_ids])
        total  = scores.sum()
        if total < 1e-10:
            return [1.0 / len(client_ids)] * len(client_ids)
        return (scores / total).tolist()

    def is_client_suspicious(
        self, client_id: int, suspicious_threshold: float = None
    ) -> bool:
        thresh = (suspicious_threshold if suspicious_threshold is not None
                  else self.tau * 0.5)
        return bool(self.trust_scores[client_id] < thresh)

    def get_trust_confidence(self, client_id: int) -> float:
        h = self.history_manager.get_trust_history(client_id)
        if len(h) < 3:
            return 0.5
        return float(1.0 / (1.0 + float(np.std(h)) * 10))

    # ──────────────────────────────────────────────────────────────────
    # Statistics
    # ──────────────────────────────────────────────────────────────────

    def get_statistics(self) -> Dict:
        scores       = self.trust_scores
        trusted_mask = scores >= self.tau
        return {
            'mean_trust':        float(np.mean(scores)),
            'std_trust':         float(np.std(scores)),
            'min_trust':         float(np.min(scores)),
            'max_trust':         float(np.max(scores)),
            'num_trusted':       int(trusted_mask.sum()),
            'num_untrusted':     int((~trusted_mask).sum()),
            'trusted_ratio':     float(trusted_mask.sum() / self.num_clients),
            'tau':               self.tau,
            'alpha':             self.alpha,
            'total_updates':     self.update_count,
            'norm_history_size': len(self._norm_history),
            'benign_norm_est':   (float(np.median(self._norm_history))
                                  if self._norm_history else 0.0),
            'signal_weights':    {
                'cosine':    round(self._w_sim, 3),
                'direction': round(self._w_dir, 3),
                'loss':      round(self._w_loss, 3),
            },
        }

    def get_trust_separation(
        self,
        benign_ids: List[int],
        malicious_ids: List[int]
    ) -> Dict[str, float]:
        b = [self.trust_scores[i] for i in benign_ids    if i < self.num_clients]
        m = [self.trust_scores[i] for i in malicious_ids if i < self.num_clients]
        if not b or not m:
            return {
                'separation': 0.0, 'avg_benign': 0.0,
                'avg_malicious': 0.0, 'benign_std': 0.0, 'malicious_std': 0.0,
            }
        avg_b, avg_m = float(np.mean(b)), float(np.mean(m))
        return {
            'separation':    avg_b - avg_m,
            'avg_benign':    avg_b,
            'avg_malicious': avg_m,
            'benign_std':    float(np.std(b)),
            'malicious_std': float(np.std(m)),
        }

    def get_client_history(self, client_id: int) -> Dict:
        return self.history_manager.get_statistics(client_id)

    def reset_client(self, client_id: int):
        self.trust_scores[client_id]        = self.initial_trust
        self.participation_count[client_id] = 0
        self.history_manager.clear_client(client_id)
        self._grad_dir_history[client_id].clear()
        self._obs_window[client_id].clear()
        self._sim_window[client_id].clear()

    def print_trust_table(self, top_n: int = 20):
        print(f"\n{'='*60}")
        print(f"{'Client':<10} {'Trust':>10} {'Status':>10} {'Rounds':>10}")
        print(f"{'-'*60}")
        for cid in np.argsort(self.trust_scores)[::-1][:top_n]:
            score  = self.trust_scores[cid]
            status = "TRUSTED" if score >= self.tau else "BLOCKED"
            print(f"{cid:<10} {score:>10.4f} {status:>10} "
                  f"{self.participation_count[cid]:>10}")
        print(f"{'='*60}")