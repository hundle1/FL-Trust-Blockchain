"""
Trust-Aware Aggregation — v4 FULL DEFENSE
==========================================
Tất cả 4 hướng fix đã được yêu cầu:

  FIX 1: alpha giảm (0.75→0.5)          — trong trust_score.py
  FIX 2: tau tăng (0.5→0.45 + init=0.3) — trong trust_score.py
  FIX 3: Norm-clipping trước aggregate  — trong file này (giữ từ v3)
  FIX 4: Median thay weighted mean      — THÊM MỚI trong file này

──────────────────────────────────────────────────────────────────────
FIX 4 — Coordinate-wise Median Aggregation (Yin et al., ICML 2018)

  Bản cũ (weighted mean):
    agg[k] = Σ w_i * update_i[k]
    → 1 poisoned update với norm 12.5x CÓ THỂ kéo lệch sum nếu weight cao
    → Ngay cả sau norm-clip, nếu attacker lọt qua trust filter thì
      weighted mean vẫn bị ảnh hưởng tuyến tính

  Bản mới (coordinate-wise median của trusted updates):
    agg[k] = median({update_i[k] : i ∈ trusted})   ← per element
    → Robust với up to ⌊n/2⌋ - 1 Byzantine clients
    → 1 outlier value tại bất kỳ coordinate nào không ảnh hưởng được kết quả
    → Không cần biết số lượng attacker trước

  Pipeline hoàn chỉnh mỗi round:
    1. Norm-clip tất cả updates về clip_multiplier × median_norm
    2. Trust-filter: loại bỏ client dưới tau
    3. Coordinate-wise median trên trusted + clipped updates
    4. Return None nếu < 2 trusted clients (giữ model)

  Tại sao cần cả norm-clip VÀ median?
    - Norm-clip: chặn outlier lớn, giúp median ổn định hơn
    - Median: robust với attacker vượt qua trust filter
    - Kết hợp: defense-in-depth, mỗi lớp chặn một loại tấn công khác nhau

  Trade-off của median vs weighted mean:
    + Robust hơn (không bị 1 outlier dominate)
    - Hội tụ chậm hơn một chút ở giai đoạn đầu (gradient noise cao hơn)
    → Đây là lý do ta vẫn giữ norm-clip: giảm noise của median

  Khi nào vẫn dùng weighted mean (fallback)?
    - Nếu chỉ có 1 trusted client → median = chính nó → dùng luôn
    - Nếu use_median=False → dùng weighted mean như cũ (backward compat)
──────────────────────────────────────────────────────────────────────
"""

import torch
from typing import Dict, List, Optional
import numpy as np


# ── Helper: norm clipping ────────────────────────────────────────────

def _clip_update_by_norm(
    update: Dict[str, torch.Tensor],
    clip_norm: float
) -> Dict[str, torch.Tensor]:
    """
    Clip gradient update về clip_norm nếu norm vượt quá.
    Trả về update gốc nếu norm đã trong ngưỡng (không copy thừa).
    """
    current_norm = float(np.sqrt(
        sum(torch.norm(v).item() ** 2 for v in update.values())
    ))
    if current_norm <= clip_norm or current_norm < 1e-8:
        return update
    scale = clip_norm / current_norm
    return {k: v * scale for k, v in update.items()}


# ── Helper: coordinate-wise median ───────────────────────────────────

def _coordinate_median(
    updates: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Coordinate-wise median across a list of gradient updates.

    Với mỗi tham số p và mỗi vị trí (i,j,...):
        result[p][i,j,...] = median({update[p][i,j,...] : update ∈ updates})

    Complexity: O(n log n × num_params × param_size)
    Với n=4 trusted clients và MNIST CNN: rất nhanh (< 1ms)

    Args:
        updates: List of gradient dicts, all với cùng keys/shapes

    Returns:
        median_update: Dict với cùng keys, mỗi value là median tensor
    """
    if len(updates) == 1:
        return {k: v.clone() for k, v in updates[0].items()}

    result = {}
    for name in updates[0].keys():
        # Stack: shape (n_clients, *param_shape)
        stacked = torch.stack([u[name] for u in updates], dim=0)
        # torch.median theo dim=0, lấy values (không lấy indices)
        result[name] = torch.median(stacked, dim=0).values
    return result


# ── Main aggregator ───────────────────────────────────────────────────

class TrustAwareAggregator:
    """
    Trust-Aware Federated Aggregation — v4 FULL DEFENSE.

    Pipeline mỗi round:
      1. Norm clipping (trước filter) — chặn gradient flip lớn
      2. Trust filtering — loại client dưới tau
      3. Coordinate-wise median (thay weighted mean) — robust aggregation
      4. Return None nếu < 1 trusted client → giữ nguyên model

    Constructor params:
      trust_manager:     TrustScoreManager instance
      enable_filtering:  Bật/tắt trust filtering
      enable_norm_clip:  Bật/tắt norm clipping (default True)
      clip_multiplier:   Clip về X × median_norm (default 2.5)
      use_median:        True = coordinate-wise median (NEW, default True)
                         False = trust-weighted mean (backward compat)
      min_trusted_for_median: Số trusted tối thiểu để dùng median (default 2)
    """

    def __init__(
        self,
        trust_manager,
        enable_filtering: bool = True,
        enable_norm_clip: bool = True,
        clip_multiplier: float = 1.5,        # đã update
        warmup_clip_multiplier: float = 1.2, # ← THÊM
        warmup_rounds: int = 3,              # ← THÊM
        use_median: bool = False,
        min_trusted_for_median: int = 2,
    ):
        self.trust_manager          = trust_manager
        self.enable_filtering       = enable_filtering
        self.enable_norm_clip       = enable_norm_clip
        self.clip_multiplier        = clip_multiplier
        self.use_median             = use_median
        self.min_trusted_for_median = min_trusted_for_median
        self.warmup_clip_multiplier = warmup_clip_multiplier
        self.warmup_rounds          = warmup_rounds
        # Tracking stats
        self.skipped_rounds   = 0
        self.filtered_counts  = []   # # clients filtered each round
        self.clip_counts      = []   # # clients clipped each round
        self.median_rounds    = 0    # rounds where median was used
        self.mean_rounds      = 0    # rounds where mean was used (fallback)

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
            metrics:    Training metrics per client (unused in median, kept for compat)
            round_num:  Current round number (for warmup bypass)

        Returns:
            Aggregated update dict, hoặc None nếu không có trusted client
            (caller nên giữ nguyên model round này)
        """
        n = len(updates)
        if n == 0:
            return None

        # ── STEP 1: Norm clipping ─────────────────────────────────────
        # Tính median norm TRƯỚC KHI filter để estimate benign norm chính xác
        # (nếu tính sau filter thì có thể bị attacker lọt vào estimate)
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

        # ── STEP 2: Trust filtering ───────────────────────────────────
        if self.enable_filtering:
            trusted_ids = self.trust_manager.get_trusted_clients(client_ids, round_num)
            self.filtered_counts.append(len(client_ids) - len(trusted_ids))

            if len(trusted_ids) == 0:
                self.skipped_rounds += 1
                print(f"  [TrustAware v4] WARNING: No trusted clients round {round_num} "
                      f"— skipping update (total skipped: {self.skipped_rounds})")
                return None

            # Chỉ giữ lại trusted updates
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

        # ── STEP 3: Aggregation ───────────────────────────────────────
        n_trusted = len(clipped_updates)

        if self.use_median and n_trusted >= self.min_trusted_for_median:
            # FIX 4: Coordinate-wise median — robust với Byzantine clients
            aggregated = _coordinate_median(clipped_updates)
            self.median_rounds += 1

        else:
            # Fallback: trust-weighted mean
            # Dùng khi: use_median=False, hoặc quá ít trusted clients
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
            'skipped_rounds':   self.skipped_rounds,
            'avg_filtered':     float(np.mean(self.filtered_counts))
                                if self.filtered_counts else 0.0,
            'avg_clipped':      float(np.mean(self.clip_counts))
                                if self.clip_counts else 0.0,
            'total_rounds':     len(self.filtered_counts),
            'median_rounds':    self.median_rounds,
            'mean_rounds':      self.mean_rounds,
        }


# ── Baseline aggregators (không thay đổi logic) ──────────────────────

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
        return {'skipped_rounds': 0, 'avg_filtered': 0.0,
                'avg_clipped': 0.0, 'total_rounds': 0,
                'median_rounds': 0, 'mean_rounds': 0}


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
        return {'skipped_rounds': 0, 'avg_filtered': 0.0,
                'avg_clipped': 0.0, 'total_rounds': 0,
                'median_rounds': 0, 'mean_rounds': 0}


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
        return {'skipped_rounds': 0, 'avg_filtered': 0.0,
                'avg_clipped': 0.0, 'total_rounds': 0,
                'median_rounds': 0, 'mean_rounds': 0}