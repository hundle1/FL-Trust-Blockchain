"""
Trust-Aware Aggregation — v5 FIXED
=====================================
Fixes vs v4:

  FIX 1: Reference gradient cho trust update phải ROBUST hơn.
    Cũ: dùng mean(benign_selected_updates) — nếu không có benign trong round
        thì fallback về mean(all) → ref bị contaminate bởi attackers.
    Fix: dùng MEDIAN thay mean để tính reference gradient.
         Median robust với outliers: dù 30-40% updates là poisoned,
         median vẫn gần với benign direction.
         Đây là reference chỉ dùng cho trust scoring, không phải aggregation.

  FIX 2: Norm clipping warmup multiplier phải CHẶT hơn cho static attack.
    Static attacker có norm 10-25x benign.
    warmup_clip_multiplier=1.2 → clip rất chặt ngay từ đầu.
    Sau warmup: clip_multiplier=1.5 (vẫn chặt).

  FIX 3: get_stats() expose thêm info để debug.

Pipeline mỗi round (không đổi):
  1. Norm clipping (median-based, trước filter)
  2. Trust filtering (sau warmup)
  3. Coordinate-wise median hoặc weighted mean
  4. Return None nếu 0 trusted → giữ model
"""

import torch
from typing import Dict, List, Optional
import numpy as np


# ── Helper: tính robust reference gradient bằng coordinate-wise median ──

def _robust_reference(
    updates: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    FIX: Tính reference gradient bằng coordinate-wise median.

    Dùng để compute cosine similarity cho trust scoring.
    Median robust với Byzantine updates: dù 40% updates là poisoned,
    median vẫn nằm trong benign region (Yin et al., ICML 2018).

    Khác với _coordinate_median (dùng cho aggregation):
      - Hàm này chạy trên TẤT CẢ updates (cả poisoned) để tạo reference
      - _coordinate_median chạy chỉ trên TRUSTED updates

    Args:
        updates: List of gradient dicts (mix of benign + malicious)

    Returns:
        median_ref: Coordinate-wise median → robust reference
    """
    if len(updates) == 1:
        return {k: v.clone() for k, v in updates[0].items()}

    result = {}
    for name in updates[0].keys():
        stacked = torch.stack([u[name] for u in updates], dim=0)
        result[name] = torch.median(stacked, dim=0).values
    return result


def _mean_reference(
    updates: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """Fallback: mean reference (dùng khi chỉ có benign updates)."""
    n = len(updates)
    result = {}
    for name in updates[0].keys():
        result[name] = torch.mean(
            torch.stack([u[name] for u in updates], dim=0), dim=0
        )
    return result


# ── Helper: norm clipping ────────────────────────────────────────────

def _clip_update_by_norm(
    update: Dict[str, torch.Tensor],
    clip_norm: float
) -> Dict[str, torch.Tensor]:
    """Clip gradient về clip_norm nếu vượt quá."""
    current_norm = float(np.sqrt(
        sum(torch.norm(v).item() ** 2 for v in update.values())
    ))
    if current_norm <= clip_norm or current_norm < 1e-8:
        return update
    scale = clip_norm / current_norm
    return {k: v * scale for k, v in update.items()}


# ── Helper: coordinate-wise median (cho aggregation) ─────────────────

def _coordinate_median(
    updates: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """Coordinate-wise median của trusted + clipped updates."""
    if len(updates) == 1:
        return {k: v.clone() for k, v in updates[0].items()}

    result = {}
    for name in updates[0].keys():
        stacked = torch.stack([u[name] for u in updates], dim=0)
        result[name] = torch.median(stacked, dim=0).values
    return result


# ── Main aggregator ───────────────────────────────────────────────────

class TrustAwareAggregator:
    """
    Trust-Aware Federated Aggregation — v5 FIXED.

    Pipeline mỗi round:
      1. Norm clipping (median-based, trước filter)
      2. Trust filtering (sau warmup_rounds)
      3. Coordinate-wise median (trusted + clipped updates)
      4. Return None nếu 0 trusted → giữ model

    NEW: compute_reference() method — tính robust reference gradient
         cho trust scoring bên ngoài aggregator.
    """

    def __init__(
        self,
        trust_manager,
        enable_filtering: bool = True,
        enable_norm_clip: bool = True,
        clip_multiplier: float = 1.5,
        warmup_clip_multiplier: float = 1.2,
        warmup_rounds: int = 1,            # FIX: từ 3 → 1 (match trust_score.py)
        use_median: bool = False,
        min_trusted_for_median: int = 2,
    ):
        self.trust_manager          = trust_manager
        self.enable_filtering       = enable_filtering
        self.enable_norm_clip       = enable_norm_clip
        self.clip_multiplier        = clip_multiplier
        self.warmup_clip_multiplier = warmup_clip_multiplier
        self.warmup_rounds          = warmup_rounds
        self.use_median             = use_median
        self.min_trusted_for_median = min_trusted_for_median

        # Tracking stats
        self.skipped_rounds  = 0
        self.filtered_counts = []
        self.clip_counts     = []
        self.median_rounds   = 0
        self.mean_rounds     = 0

    # ------------------------------------------------------------------
    # NEW: Compute robust reference gradient for trust scoring
    # ------------------------------------------------------------------

    def compute_reference(
        self,
        updates: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
        malicious_ids: Optional[set] = None,
        use_robust: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        FIX: Tính reference gradient robust cho trust scoring.

        Strategy (theo thứ tự ưu tiên):
          1. Nếu có benign_ids và đủ benign: mean(benign_updates)
             — chính xác nhất vì chỉ dùng benign
          2. Nếu use_robust=True: median(all_updates)
             — robust với up to 49% Byzantine
          3. Fallback: mean(all_updates)
             — như cũ, dễ bị contaminate

        Args:
            updates:       All updates this round
            client_ids:    Corresponding client IDs
            malicious_ids: Known malicious IDs (nếu biết, chỉ dùng cho research)
            use_robust:    True = dùng median nếu không có benign info

        Returns:
            ref_gradient: Robust reference gradient
        """
        if malicious_ids is not None:
            # Research mode: biết malicious IDs → dùng chỉ benign updates
            benign_ups = [
                updates[i] for i, cid in enumerate(client_ids)
                if cid not in malicious_ids
            ]
            if benign_ups:
                return _mean_reference(benign_ups)

        if use_robust and len(updates) >= 2:
            # Production mode: không biết malicious → dùng median (robust)
            return _robust_reference(updates)

        # Fallback: mean of all
        return _mean_reference(updates)

    # ------------------------------------------------------------------
    # Main aggregation pipeline
    # ------------------------------------------------------------------

    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
        metrics: Optional[List[Dict[str, float]]] = None,
        round_num: int = 0
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Aggregate client updates với full defense pipeline.

        Args:
            updates:    List of model update dicts
            client_ids: Corresponding client IDs
            metrics:    Training metrics per client (unused in median)
            round_num:  Current round number

        Returns:
            Aggregated update dict, hoặc None nếu 0 trusted clients
        """
        n = len(updates)
        if n == 0:
            return None

        # ── STEP 1: Norm clipping ──────────────────────────────────────
        clipped_updates = updates
        num_clipped = 0

        if self.enable_norm_clip:
            norms = [
                float(np.sqrt(sum(torch.norm(v).item() ** 2 for v in u.values())))
                for u in updates
            ]
            median_norm = float(np.median(norms))
            effective_clip = (
                self.warmup_clip_multiplier
                if round_num < self.warmup_rounds
                else self.clip_multiplier
            )
            clip_norm = effective_clip * max(median_norm, 1e-6)

            clipped_updates = []
            for u, norm in zip(updates, norms):
                if norm > clip_norm:
                    clipped_updates.append(_clip_update_by_norm(u, clip_norm))
                    num_clipped += 1
                else:
                    clipped_updates.append(u)

        self.clip_counts.append(num_clipped)

        # ── STEP 2: Trust filtering ────────────────────────────────────
        if self.enable_filtering:
            trusted_ids = self.trust_manager.get_trusted_clients(client_ids, round_num)
            self.filtered_counts.append(len(client_ids) - len(trusted_ids))

            if len(trusted_ids) == 0:
                self.skipped_rounds += 1
                print(f"  [TrustAware v5] WARNING: No trusted clients round {round_num} "
                      f"— skipping update (total skipped: {self.skipped_rounds})")
                return None

            trusted_updates    = []
            trusted_client_ids = []
            for i, cid in enumerate(client_ids):
                if cid in trusted_ids:
                    trusted_updates.append(clipped_updates[i])
                    trusted_client_ids.append(cid)

            clipped_updates = trusted_updates
            client_ids      = trusted_client_ids
        else:
            self.filtered_counts.append(0)

        # ── STEP 3: Aggregation ────────────────────────────────────────
        n_trusted = len(clipped_updates)

        if self.use_median and n_trusted >= self.min_trusted_for_median:
            aggregated = _coordinate_median(clipped_updates)
            self.median_rounds += 1
        else:
            # Trust-weighted mean
            raw_weights = np.array([
                self.trust_manager.get_trust_score(cid)
                for cid in client_ids
            ])
            total = raw_weights.sum()
            if total < 1e-10:
                trust_weights = [1.0 / n_trusted] * n_trusted
            else:
                trust_weights = (raw_weights / total).tolist()

            aggregated = {}
            for name in clipped_updates[0].keys():
                agg_param = torch.zeros_like(clipped_updates[0][name])
                for update, weight in zip(clipped_updates, trust_weights):
                    agg_param += weight * update[name]
                aggregated[name] = agg_param

            self.mean_rounds += 1

        return aggregated

    def get_stats(self) -> Dict:
        return {
            'skipped_rounds': self.skipped_rounds,
            'avg_filtered':   float(np.mean(self.filtered_counts))
                              if self.filtered_counts else 0.0,
            'avg_clipped':    float(np.mean(self.clip_counts))
                              if self.clip_counts else 0.0,
            'total_rounds':   len(self.filtered_counts),
            'median_rounds':  self.median_rounds,
            'mean_rounds':    self.mean_rounds,
        }


# ── Baseline aggregators ─────────────────────────────────────────────

class FedAvgAggregator:
    """Standard FedAvg — uniform average (baseline, không có defense)."""

    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
        metrics: Optional[List[Dict[str, float]]] = None,
        round_num: int = 0
    ) -> Dict[str, torch.Tensor]:
        n = len(updates)
        aggregated = {}
        for name in updates[0].keys():
            agg = torch.zeros_like(updates[0][name])
            for u in updates:
                agg += u[name] / n
            aggregated[name] = agg
        return aggregated

    def get_stats(self) -> Dict:
        return {
            'skipped_rounds': 0, 'avg_filtered': 0.0,
            'avg_clipped': 0.0,  'total_rounds': 0,
            'median_rounds': 0,  'mean_rounds': 0,
        }


class KrumAggregator:
    """Krum — Byzantine-robust baseline."""

    def __init__(self, num_malicious: int = 0):
        self.num_malicious = num_malicious

    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
        metrics: Optional[List[Dict[str, float]]] = None,
        round_num: int = 0
    ) -> Dict[str, torch.Tensor]:
        n    = len(updates)
        m    = self.num_malicious
        flat = [torch.cat([p.flatten() for p in u.values()]) for u in updates]

        scores = []
        for i in range(n):
            dists = sorted(
                torch.norm(flat[i] - flat[j]).item()
                for j in range(n) if i != j
            )
            scores.append(sum(dists[:max(1, n - m - 2)]))

        selected_idx = int(np.argmin(scores))
        return updates[selected_idx]

    def get_stats(self) -> Dict:
        return {
            'skipped_rounds': 0, 'avg_filtered': 0.0,
            'avg_clipped': 0.0,  'total_rounds': 0,
            'median_rounds': 0,  'mean_rounds': 0,
        }


class TrimmedMeanAggregator:
    """Trimmed Mean — removes extreme values per coordinate."""

    def __init__(self, trim_ratio: float = 0.1):
        self.trim_ratio = trim_ratio

    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
        metrics: Optional[List[Dict[str, float]]] = None,
        round_num: int = 0
    ) -> Dict[str, torch.Tensor]:
        n          = len(updates)
        trim_count = int(n * self.trim_ratio)
        aggregated = {}
        for name in updates[0].keys():
            stacked        = torch.stack([u[name] for u in updates])
            sorted_vals, _ = torch.sort(stacked, dim=0)
            if trim_count > 0 and 2 * trim_count < n:
                trimmed = sorted_vals[trim_count:-trim_count]
            else:
                trimmed = sorted_vals
            aggregated[name] = torch.mean(trimmed, dim=0)
        return aggregated

    def get_stats(self) -> Dict:
        return {
            'skipped_rounds': 0, 'avg_filtered': 0.0,
            'avg_clipped': 0.0,  'total_rounds': 0,
            'median_rounds': 0,  'mean_rounds': 0,
        }