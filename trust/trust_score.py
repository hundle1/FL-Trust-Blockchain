"""
Trust Score Manager — FINAL v2 (Paper-Ready, All 3 Upgrades)
=============================================================
Thêm 3 upgrade quan trọng sau review:

UPGRADE 1 — MAD Normalization (thay vì f/mu_t):
  mu_t  = EMA_beta( median(f^t) )
  sigma_t = EMA_beta( median(|f_i^t - mu_t|) )   [MAD]
  z_i   = (f_i^t - mu_t) / (sigma_t + eps)
  f̃_i^t = clip( (z_i + c) / (2c), 0, 1 )         [rescale to [0,1]]
  → Robust against collective drift, không cần lower_bound trick

UPGRADE 2 — Variance-based alpha (bắt slow poisoning):
  m_t = mean(sim_window[-W:])
  m_prev = mean(sim_window[-2W:-W])
  v_t = var(sim_window[-W:])
  if m_t < m_prev - delta AND v_t < var_thresh:
      alpha = alpha_slow_poison  # 0.50
  → Bắt attacker "pro": giảm từ từ + ổn định (variance thấp)

UPGRADE 3 — Z-score Norm Detection (thay vì ratio > 3.0):
  mu_n = running_median(norms)
  sigma_n = running_MAD(norms)
  z_i = (n_i - mu_n) / (sigma_n + eps)
  P_norm = sigmoid(|z_i| - tau_n)   [smooth penalty, reviewer-friendly]
  → Không bypass được bằng ratio=2.9, statistical anomaly detection
"""

import torch
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
from trust.history_buffer import ClientHistoryManager
from trust.trust_decay import TrustDecay


class TrustScoreManager:
    """
    Trust Score Manager — FINAL v2

    3-layer framework + 3 statistical upgrades:
      Layer 1: f_i^t  = additive scoring (S_cos + S_dir + S_loss - P_norm - P_sus)
      Layer 2: f̃_i^t = MAD normalization (robust, reviewer-friendly)
      Layer 3: T_i^t  = adaptive EMA with dynamic alpha
                         (sudden drop + trend + variance-based slow poison detection)
    """

    def __init__(
        self,
        num_clients: int,

        # ── Layer 3: EMA params ───────────────────────────────────
        alpha: float = 0.75,             # default alpha
        alpha_sudden: float = 0.50,      # sudden drop (|drop| > sudden_thresh)
        alpha_trend: float = 0.60,       # sustained degradation
        alpha_slow_poison: float = 0.50, # slow poisoning (low variance + mean drop)
        tau: float = 0.35,               # trust threshold
        initial_trust: float = 1.0,
        enable_decay: bool = True,
        window_size: int = 10,
        min_trust: float = 0.05,         # trust floor
        max_trust: float = 1.0,

        # ── Layer 1: scoring weights (must sum = 1.00) ────────────
        w_cos: float = 0.40,
        w_dir: float = 0.30,             # increased from 0.25 — key for norm-tuned
        w_loss: float = 0.15,
        w_norm: float = 0.10,
        w_sus: float = 0.05,

        # ── Layer 2: MAD normalization ────────────────────────────
        mu_beta: float = 0.9,            # EMA smoothing for mu and sigma
        mu_init: float = 0.5,            # initial mu_t
        sigma_init: float = 0.15,        # initial sigma_t
        mad_eps: float = 1e-4,           # numerical stability
        mad_clip: float = 10.0,          # clip z to [-c, c] — c=10 needed with sigma~0.05

        # ── Dynamic alpha: detection params ───────────────────────
        sudden_drop_threshold: float = 0.30,   # |drop in 1 round| → sudden
        trend_threshold: float = -0.07,         # mean trend over W rounds → degrading
        trend_window: int = 3,
        # Variance-based slow poison detection (UPGRADE 2)
        var_window: int = 5,                    # W for mean/var computation
        var_mean_delta: float = 0.08,           # m_t < m_prev - delta
        var_thresh: float = 0.01,               # v_t < var_thresh → suspicious

        # ── Sustained penalty ─────────────────────────────────────
        sustained_threshold: float = 0.45,
        sustained_window: int = 2,

        # ── UPGRADE 3: Z-score norm penalty ───────────────────────
        norm_z_tau: float = 2.0,               # |z| > tau_n → penalty starts
        norm_z_smooth: bool = True,             # True=sigmoid, False=hard threshold
        absolute_norm_threshold: float = 15.0, # early-round absolute fallback

        # ── Idle decay ────────────────────────────────────────────
        idle_decay_rate: float = 0.002,
        warmup_rounds: int = 0,
    ):
        # Weight validation
        w_total = w_cos + w_dir + w_loss + w_norm + w_sus
        assert abs(w_total - 1.0) < 1e-5, \
            f"Weights must sum to 1.0, got {w_total:.5f}"

        self.num_clients = num_clients
        self.alpha = alpha
        self.alpha_sudden = alpha_sudden
        self.alpha_trend = alpha_trend
        self.alpha_slow_poison = alpha_slow_poison
        self.tau = tau
        self.initial_trust = initial_trust
        self.enable_decay = enable_decay
        self.min_trust = min_trust
        self.max_trust = max_trust
        self.idle_decay_rate = idle_decay_rate
        self.warmup_rounds = warmup_rounds

        # Layer 1 weights
        self.w_cos  = w_cos
        self.w_dir  = w_dir
        self.w_loss = w_loss
        self.w_norm = w_norm
        self.w_sus  = w_sus

        # Layer 2: MAD normalization state
        self.mu_beta   = mu_beta
        self.mad_eps   = mad_eps
        self.mad_clip  = mad_clip
        self.mu_t      = mu_init     # running median of f
        self.sigma_t   = sigma_init  # running MAD of f

        # Dynamic alpha detection
        self.sudden_drop_threshold = sudden_drop_threshold
        self.trend_threshold       = trend_threshold
        self.trend_window          = trend_window
        self.var_window            = var_window
        self.var_mean_delta        = var_mean_delta
        self.var_thresh            = var_thresh

        # Sustained penalty
        self.sustained_threshold = sustained_threshold
        self.sustained_window    = sustained_window

        # Upgrade 3: Z-score norm penalty
        self.norm_z_tau            = norm_z_tau
        self.norm_z_smooth         = norm_z_smooth
        self.absolute_norm_threshold = absolute_norm_threshold

        # Core state
        self.trust_scores        = np.full(num_clients, initial_trust, dtype=np.float64)
        self.history_manager     = ClientHistoryManager(num_clients, window_size)
        self.last_participated   = np.full(num_clients, -1, dtype=int)
        self.participation_count = np.zeros(num_clients, dtype=int)
        self.update_count        = 0

        # Per-client direction history (for S_dir)
        self._grad_dir_history: Dict[int, deque] = {
            i: deque(maxlen=5) for i in range(num_clients)
        }

        # Per-client cosine similarity window (for alpha + sustained)
        self._sim_window: Dict[int, deque] = {
            i: deque(maxlen=max(sustained_window, trend_window, var_window * 2) + 2)
            for i in range(num_clients)
        }

        # Running norm statistics for z-score (UPGRADE 3)
        self._norm_mu    = 4.0    # initial estimate (MNIST benign ≈ 4.2)
        self._norm_sigma = 1.0    # initial MAD estimate
        self._norm_history: List[float] = []
        self._norm_window_size = 50

    # ══════════════════════════════════════════════════════
    # LAYER 1: Signal computation
    # ══════════════════════════════════════════════════════

    def compute_cosine_similarity(
        self,
        client_update: Dict[str, torch.Tensor],
        reference_gradient: Dict[str, torch.Tensor],
    ) -> float:
        """S_cos: cosine similarity vs reference, mapped to [0, 1]."""
        c_flat = torch.cat([v.flatten() for v in client_update.values()])
        r_flat = torch.cat([v.flatten() for v in reference_gradient.values()])
        r_flat = r_flat.to(c_flat.device)  # fix: reference may be on CPU
        nc = torch.norm(c_flat)
        nr = torch.norm(r_flat)
        if nc < 1e-10 or nr < 1e-10:
            return 0.0
        cos = torch.dot(c_flat, r_flat) / (nc * nr)
        return float(np.clip((cos.item() + 1.0) / 2.0, 0.0, 1.0))

    def compute_direction_consistency(
        self,
        client_id: int,
        client_update: Dict[str, torch.Tensor],
    ) -> float:
        """
        S_dir: cosine vs client's OWN gradient history.
        Key signal for norm-tuned attack (direction flips, norm ≈ benign).
        Returns 0.5 (neutral) when history < 2.
        """
        history = self._grad_dir_history[client_id]
        if len(history) < 2:
            return 0.5

        c_flat = torch.cat([v.flatten() for v in client_update.values()])
        nc = torch.norm(c_flat)
        if nc < 1e-10:
            return 0.5

        # History stored on CPU — move to same device as client update
        stacked  = torch.stack(list(history)).to(c_flat.device)
        mean_dir = stacked.mean(dim=0)
        nm       = torch.norm(mean_dir)
        if nm < 1e-10:
            return 0.5

        cos = torch.dot(c_flat / nc, mean_dir / nm)
        return float(np.clip((cos.item() + 1.0) / 2.0, 0.0, 1.0))

    def _compute_loss_signal(
        self, client_id: int, loss: Optional[float]
    ) -> float:
        """S_loss: normalized inverse loss vs client's own history."""
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

    @staticmethod
    def _compute_norm(update: Dict[str, torch.Tensor]) -> float:
        return float(np.sqrt(
            sum(torch.norm(v).item() ** 2 for v in update.values())
        ))

    # ── UPGRADE 3: Z-score norm penalty ──────────────────────────

    def _update_norm_statistics(self, norms: List[float]) -> None:
        """
        Update running mu_n and sigma_n (MAD) from current round's norms.
        Only uses norms below absolute_norm_threshold for benign estimation.
        """
        benign_norms = [n for n in norms if n < self.absolute_norm_threshold]
        if not benign_norms:
            return

        round_median = float(np.median(benign_norms))
        round_mad    = float(np.median(np.abs(
            np.array(benign_norms) - round_median
        ))) + self.mad_eps

        # EMA update
        self._norm_mu    = self.mu_beta * self._norm_mu    + (1 - self.mu_beta) * round_median
        self._norm_sigma = self.mu_beta * self._norm_sigma + (1 - self.mu_beta) * round_mad

    def _norm_penalty_zscore(self, client_norm: float) -> float:
        """
        UPGRADE 3: Z-score based norm penalty.
        P_norm = sigmoid(|z| - tau_n)   [smooth, reviewer-friendly]

        Early rounds (< 5 norms seen): absolute threshold fallback.
        Later: statistical z-score detection.

        Returns value in [0, 1] — will be weighted by w_norm.
        """
        # Early rounds fallback
        if len(self._norm_history) < 5:
            if client_norm > self.absolute_norm_threshold:
                ratio  = client_norm / self.absolute_norm_threshold
                excess = min((ratio - 1.0) / 3.0, 1.0)
                return float(excess)
            return 0.0

        # Z-score detection
        z = abs(client_norm - self._norm_mu) / (self._norm_sigma + self.mad_eps)

        if self.norm_z_smooth:
            # Smooth sigmoid penalty: P = sigmoid(|z| - tau_n)
            # = 0 when |z| << tau_n, → 1 when |z| >> tau_n
            penalty = 1.0 / (1.0 + np.exp(-(z - self.norm_z_tau)))
        else:
            # Hard threshold
            penalty = float(np.clip((z - self.norm_z_tau) / self.norm_z_tau, 0.0, 1.0))

        return float(penalty)

    def _sustained_penalty_additive(
        self, client_id: int, s_cos: float
    ) -> float:
        """
        P_sus: penalty for sustained low cosine similarity.
        Fires when ALL of last K rounds have sim < threshold.
        Returns value in [0, 1] — weighted by w_sus.
        """
        win = self._sim_window[client_id]
        if len(win) < self.sustained_window:
            return 0.0

        recent = list(win)[-self.sustained_window:]
        if all(s < self.sustained_threshold for s in recent):
            avg_deficit = np.mean([
                max(0.0, self.sustained_threshold - s) for s in recent
            ])
            return float(np.clip(
                avg_deficit / self.sustained_threshold, 0.0, 1.0
            ))
        return 0.0

    def _compute_f(
        self,
        s_cos: float,
        s_dir: float,
        s_loss: float,
        p_norm: float,
        p_sus: float,
    ) -> float:
        """
        Layer 1 — additive scoring formula (matches paper):
        f_i^t = w_cos*S_cos + w_dir*S_dir + w_loss*S_loss
                - w_norm*P_norm - w_sus*P_sus
        NOT clipped here — clipping happens after MAD normalize.
        """
        return (self.w_cos  * s_cos
                + self.w_dir  * s_dir
                + self.w_loss * s_loss
                - self.w_norm * p_norm
                - self.w_sus  * p_sus)

    # ══════════════════════════════════════════════════════
    # LAYER 2: MAD Normalization (UPGRADE 1)
    # ══════════════════════════════════════════════════════

    def _update_mad_statistics(self, f_values: List[float]) -> None:
        """
        UPGRADE 1: Update running mu_t (median) and sigma_t (MAD).

        mu_t    = beta * mu_{t-1}    + (1-beta) * median(f^t)
        sigma_t = beta * sigma_{t-1} + (1-beta) * median(|f_i^t - median(f^t)|)

        MAD is more robust than std: resistant to outliers from attackers.
        """
        if not f_values:
            return

        f_arr        = np.array(f_values)
        round_median = float(np.median(f_arr))
        round_mad    = float(np.median(np.abs(f_arr - round_median))) + self.mad_eps

        self.mu_t    = self.mu_beta * self.mu_t    + (1 - self.mu_beta) * round_median
        self.sigma_t = self.mu_beta * self.sigma_t + (1 - self.mu_beta) * round_mad

    def _normalize_f_mad(self, f: float) -> float:
        """
        UPGRADE 1: MAD-based normalization.

        z = (f - mu_t) / (sigma_t + eps)
        f_tilde = clip( (z + c) / (2c), 0, 1 )

        Properties:
          - Benign clients near mu_t → z ≈ 0 → f_tilde ≈ 0.5
          - Strong attackers far below mu_t → z ≈ -c → f_tilde ≈ 0
          - Better benign → z slightly positive → f_tilde > 0.5
        """
        z = (f - self.mu_t) / (self.sigma_t + self.mad_eps)
        z = float(np.clip(z, -self.mad_clip, self.mad_clip))
        # Rescale from [-c, c] to [0, 1]
        return float((z + self.mad_clip) / (2.0 * self.mad_clip))

    # ══════════════════════════════════════════════════════
    # LAYER 3: Dynamic alpha (with UPGRADE 2)
    # ══════════════════════════════════════════════════════

    def _get_dynamic_alpha(self, client_id: int, s_cos: float) -> float:
        """
        Dynamic alpha with 4 levels:

        1. alpha_sudden (0.50):     sudden drop > sudden_drop_threshold in 1 round
                                    → delayed attack, sudden flip
        2. alpha_slow_poison (0.50): UPGRADE 2 — mean drop + low variance
                                    → slow poisoning, pro attacker
        3. alpha_trend (0.60):      sustained negative trend over trend_window
                                    → adaptive attacker gradual degradation
        4. alpha (0.75):            normal behavior → stable trust
        """
        hist = list(self._sim_window[client_id])

        # ── Check 1: Sudden drop (1 round) ───────────────────────
        if len(hist) >= 1:
            drop = hist[-1] - s_cos
            if drop > self.sudden_drop_threshold:
                return self.alpha_sudden  # 0.50

        # ── Check 2: UPGRADE 2 — Slow poison (mean↓ + var low) ──
        if len(hist) >= self.var_window * 2:
            window_now  = hist[-self.var_window:]
            window_prev = hist[-self.var_window * 2:-self.var_window]

            m_now  = float(np.mean(window_now))
            m_prev = float(np.mean(window_prev))
            v_now  = float(np.var(window_now))

            if (m_now < m_prev - self.var_mean_delta
                    and v_now < self.var_thresh):
                # Mean dropped significantly AND variance is low
                # → attacker is "pro": smoothly degrading, not random
                return self.alpha_slow_poison  # 0.50

        # ── Check 3: Sustained trend ──────────────────────────────
        if len(hist) >= self.trend_window:
            recent = hist[-self.trend_window:]
            trend  = float(np.mean(np.diff(recent)))
            if trend < self.trend_threshold:  # -0.07
                return self.alpha_trend  # 0.60

        return self.alpha  # 0.75 default

    # ══════════════════════════════════════════════════════
    # CORE: update_trust_batch — recommended entry point
    # ══════════════════════════════════════════════════════

    def update_trust_batch(
        self,
        updates: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
        reference_gradient: Dict[str, torch.Tensor],
        metrics_list: Optional[List[Optional[Dict[str, float]]]] = None,
        round_num: int = 0,
    ) -> List[float]:
        """
        RECOMMENDED: Update trust for all clients in one round.

        Correct Layer 2 behavior: mu_t and sigma_t are updated using
        ALL f values from the round before normalizing any individual client.

        Args:
            updates:           All client model update dicts
            client_ids:        Corresponding client IDs
            reference_gradient: Robust reference (coordinate-wise median)
            metrics_list:      Training metrics per client (loss, accuracy, ...)
            round_num:         Current FL round number

        Returns:
            List of updated trust scores (same order as client_ids)
        """
        if metrics_list is None:
            metrics_list = [None] * len(client_ids)

        # ── Step 1: Compute ALL signals and raw f values ──────────
        raw_f_values = []
        all_norms    = []
        signals_cache = []  # (s_cos, s_dir, s_loss, p_norm_raw, p_sus, norm, f_raw)

        # First pass: compute norms (needed for z-score update)
        norms = [self._compute_norm(u) for u in updates]
        all_norms = norms

        # Update norm statistics with this round's norms
        self._update_norm_statistics(norms)

        for i, (client_id, update) in enumerate(zip(client_ids, updates)):
            metrics = metrics_list[i]

            s_cos  = self.compute_cosine_similarity(update, reference_gradient)
            s_dir  = self.compute_direction_consistency(client_id, update)
            s_loss = self._compute_loss_signal(
                client_id,
                metrics.get('loss') if metrics else None
            )

            # UPGRADE 3: z-score norm penalty
            p_norm = self._norm_penalty_zscore(norms[i])
            p_sus  = self._sustained_penalty_additive(client_id, s_cos)

            f_raw = self._compute_f(s_cos, s_dir, s_loss, p_norm, p_sus)
            raw_f_values.append(f_raw)
            signals_cache.append((s_cos, s_dir, s_loss, p_norm, p_sus, norms[i], f_raw))

        # ── Step 2: Update MAD statistics with ALL f values ───────
        # CRITICAL: must happen before normalizing any individual client
        self._update_mad_statistics(raw_f_values)

        # ── Step 3: Normalize + EMA update per client ─────────────
        new_trusts = []
        for i, client_id in enumerate(client_ids):
            s_cos, s_dir, s_loss, p_norm, p_sus, norm_i, f_raw = signals_cache[i]
            update  = updates[i]
            metrics = metrics_list[i]

            # Layer 2: MAD normalize
            f_tilde = self._normalize_f_mad(f_raw)

            # Layer 3: dynamic alpha + EMA
            alpha_t   = self._get_dynamic_alpha(client_id, s_cos)
            old_trust = self.trust_scores[client_id]
            new_trust = alpha_t * old_trust + (1 - alpha_t) * f_tilde
            new_trust = float(np.clip(new_trust, self.min_trust, self.max_trust))

            # ── Update internal state ─────────────────────────────
            self.trust_scores[client_id]        = new_trust
            self.last_participated[client_id]   = round_num
            self.participation_count[client_id] += 1
            self.update_count += 1

            # Store normalized direction for S_dir
            flat = torch.cat([v.flatten() for v in update.values()])
            n    = torch.norm(flat)
            if n > 1e-10:
                self._grad_dir_history[client_id].append(
                    (flat / n).detach().cpu()
                )

            # Update sim window (for alpha detection)
            self._sim_window[client_id].append(s_cos)

            # Update norm history (benign range)
            if norm_i < self.absolute_norm_threshold:
                self._norm_history.append(norm_i)
                if len(self._norm_history) > self._norm_window_size:
                    self._norm_history.pop(0)

            # History manager
            self.history_manager.add_gradient_similarity(client_id, s_cos)
            self.history_manager.add_trust(client_id, new_trust)
            if metrics:
                self.history_manager.add_loss(
                    client_id, metrics.get('loss', 0.0))
                self.history_manager.add_accuracy(
                    client_id, metrics.get('accuracy', 0.0))

            new_trusts.append(new_trust)

        return new_trusts

    def update_trust(
        self,
        client_id: int,
        client_update: Dict[str, torch.Tensor],
        reference_gradient: Dict[str, torch.Tensor],
        metrics: Optional[Dict[str, float]] = None,
        round_num: int = 0,
        all_f_values: Optional[List[float]] = None,
    ) -> float:
        """
        Single-client update. Prefer update_trust_batch() for correct
        Layer 2 behavior. Use this only when calling one client at a time.
        """
        s_cos  = self.compute_cosine_similarity(client_update, reference_gradient)
        s_dir  = self.compute_direction_consistency(client_id, client_update)
        s_loss = self._compute_loss_signal(
            client_id, metrics.get('loss') if metrics else None
        )

        client_norm = self._compute_norm(client_update)
        self._update_norm_statistics([client_norm])
        p_norm = self._norm_penalty_zscore(client_norm)
        p_sus  = self._sustained_penalty_additive(client_id, s_cos)

        f_raw = self._compute_f(s_cos, s_dir, s_loss, p_norm, p_sus)

        if all_f_values is not None:
            self._update_mad_statistics(all_f_values)
        else:
            self._update_mad_statistics([f_raw])

        f_tilde   = self._normalize_f_mad(f_raw)
        alpha_t   = self._get_dynamic_alpha(client_id, s_cos)
        old_trust = self.trust_scores[client_id]
        new_trust = alpha_t * old_trust + (1 - alpha_t) * f_tilde
        new_trust = float(np.clip(new_trust, self.min_trust, self.max_trust))

        self.trust_scores[client_id]        = new_trust
        self.last_participated[client_id]   = round_num
        self.participation_count[client_id] += 1
        self.update_count += 1

        flat = torch.cat([v.flatten() for v in client_update.values()])
        nn   = torch.norm(flat)
        if nn > 1e-10:
            self._grad_dir_history[client_id].append(
                (flat / nn).detach().cpu()
            )
        self._sim_window[client_id].append(s_cos)

        if client_norm < self.absolute_norm_threshold:
            self._norm_history.append(client_norm)
            if len(self._norm_history) > self._norm_window_size:
                self._norm_history.pop(0)

        self.history_manager.add_gradient_similarity(client_id, s_cos)
        self.history_manager.add_trust(client_id, new_trust)
        if metrics:
            self.history_manager.add_loss(client_id, metrics.get('loss', 0.0))
            self.history_manager.add_accuracy(
                client_id, metrics.get('accuracy', 0.0))

        return new_trust

    # ══════════════════════════════════════════════════════
    # Idle decay
    # ══════════════════════════════════════════════════════

    def apply_idle_decay(
        self,
        active_client_ids: List[int],
        round_num: int,
        decay_rate: Optional[float] = None,
    ):
        """Linear decay for clients not selected this round."""
        if not self.enable_decay:
            return
        rate       = decay_rate if decay_rate is not None else self.idle_decay_rate
        active_set = set(active_client_ids)
        for cid in range(self.num_clients):
            if cid not in active_set:
                self.trust_scores[cid] = max(
                    self.min_trust,
                    TrustDecay.linear_decay(self.trust_scores[cid], rate)
                )

    # ══════════════════════════════════════════════════════
    # Querying
    # ══════════════════════════════════════════════════════

    def get_trust_score(self, client_id: int) -> float:
        return float(self.trust_scores[client_id])

    def get_all_trust_scores(self) -> np.ndarray:
        return self.trust_scores.copy()

    def get_trusted_clients(
        self, client_ids: List[int], round_num: int = 999
    ) -> List[int]:
        if round_num < self.warmup_rounds:
            return list(client_ids)
        return [cid for cid in client_ids
                if self.trust_scores[cid] >= self.tau]

    def get_trust_weights(self, client_ids: List[int]) -> List[float]:
        scores = np.array([self.trust_scores[cid] for cid in client_ids])
        total  = scores.sum()
        if total < 1e-10:
            return [1.0 / len(client_ids)] * len(client_ids)
        return (scores / total).tolist()

    def is_client_suspicious(
        self, client_id: int, threshold: Optional[float] = None
    ) -> bool:
        thresh = threshold if threshold is not None else self.tau * 0.7
        return bool(self.trust_scores[client_id] < thresh)

    def get_trust_separation(
        self,
        benign_ids: List[int],
        malicious_ids: List[int],
    ) -> Dict:
        b = [self.trust_scores[i] for i in benign_ids  if i < self.num_clients]
        m = [self.trust_scores[i] for i in malicious_ids if i < self.num_clients]
        if not b or not m:
            return {'separation': 0.0, 'avg_benign': 0.0,
                    'avg_malicious': 0.0, 'benign_std': 0.0,
                    'malicious_std': 0.0}
        avg_b = float(np.mean(b))
        avg_m = float(np.mean(m))
        return {
            'separation':    avg_b - avg_m,
            'avg_benign':    avg_b,
            'avg_malicious': avg_m,
            'benign_std':    float(np.std(b)),
            'malicious_std': float(np.std(m)),
        }

    def get_statistics(self) -> Dict:
        scores       = self.trust_scores
        trusted_mask = scores >= self.tau
        return {
            'mean_trust':      float(np.mean(scores)),
            'std_trust':       float(np.std(scores)),
            'min_trust':       float(np.min(scores)),
            'max_trust':       float(np.max(scores)),
            'num_trusted':     int(trusted_mask.sum()),
            'num_untrusted':   int((~trusted_mask).sum()),
            'trusted_ratio':   float(trusted_mask.sum() / self.num_clients),
            'tau':             self.tau,
            'alpha_default':   self.alpha,
            # Layer 2 state
            'mu_t':            round(float(self.mu_t), 4),
            'sigma_t':         round(float(self.sigma_t), 4),
            # Norm statistics
            'norm_mu':         round(float(self._norm_mu), 4),
            'norm_sigma':      round(float(self._norm_sigma), 4),
            'total_updates':   self.update_count,
            'benign_norm_est': float(np.median(self._norm_history))
                               if self._norm_history else 0.0,
            'signal_weights':  {
                'cos':      self.w_cos,
                'dir':      self.w_dir,
                'loss':     self.w_loss,
                'norm_pen': self.w_norm,
                'sus_pen':  self.w_sus,
            },
        }

    def reset_client(self, client_id: int):
        self.trust_scores[client_id]   = self.initial_trust
        self.participation_count[client_id] = 0
        self.history_manager.clear_client(client_id)
        self._grad_dir_history[client_id].clear()
        self._sim_window[client_id].clear()

    def print_trust_table(self, top_n: int = 20):
        print(f"\n{'='*70}")
        print(f"  mu_t={self.mu_t:.4f}  sigma_t={self.sigma_t:.4f}  "
              f"tau={self.tau}  alpha={self.alpha}")
        print(f"  norm_mu={self._norm_mu:.3f}  norm_sigma={self._norm_sigma:.3f}")
        print(f"{'='*70}")
        print(f"  {'Client':<10} {'Trust':>10} {'Status':>10} {'Rounds':>10}")
        print(f"  {'-'*55}")
        for cid in np.argsort(self.trust_scores)[::-1][:top_n]:
            score  = self.trust_scores[cid]
            status = "TRUSTED" if score >= self.tau else "BLOCKED"
            print(f"  {cid:<10} {score:>10.4f} {status:>10} "
                  f"{self.participation_count[cid]:>10}")
        print(f"{'='*70}")