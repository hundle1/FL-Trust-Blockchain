"""
Trust-Aware Aggregation — FINAL VERSION (Paper-Ready)
======================================================
Matches the 3-layer trust framework in trust_score.py.

Key changes from previous version:
  1. Fallback top-k median (NO MORE return None → collapse fixed)
  2. clip_multiplier = 1.5 (from 2.0 — tighter for static/label flip)
  3. Uses update_trust_batch() for correct mu_t update
  4. Exposes get_stats() for ablation reporting
"""

import torch
from typing import Dict, List, Optional, Set
import numpy as np


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _compute_norm(update: Dict[str, torch.Tensor]) -> float:
    return float(np.sqrt(
        sum(torch.norm(v).item() ** 2 for v in update.values())
    ))


def _clip_by_norm(
    update: Dict[str, torch.Tensor], clip_norm: float
) -> Dict[str, torch.Tensor]:
    n = _compute_norm(update)
    if n <= clip_norm or n < 1e-8:
        return update
    scale = clip_norm / n
    return {k: v * scale for k, v in update.items()}


def _coordinate_median(
    updates: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """Coordinate-wise median — robust reference and fallback aggregation."""
    if len(updates) == 1:
        return {k: v.clone() for k, v in updates[0].items()}
    result = {}
    for name in updates[0].keys():
        stacked = torch.stack([u[name] for u in updates], dim=0)
        result[name] = torch.median(stacked, dim=0).values
    return result


def _weighted_mean(
    updates: List[Dict[str, torch.Tensor]],
    weights: List[float],
) -> Dict[str, torch.Tensor]:
    result = {}
    for name in updates[0].keys():
        agg = torch.zeros_like(updates[0][name])
        for update, w in zip(updates, weights):
            agg += w * update[name]
        result[name] = agg
    return result


# ══════════════════════════════════════════════════════════════════════
# Main aggregator
# ══════════════════════════════════════════════════════════════════════

class TrustAwareAggregator:
    """
    Trust-Aware Aggregation — FINAL

    Pipeline per round:
      1. Norm clipping  (median-based, clip_multiplier=1.5)
      2. Trust filtering (T >= tau)
      3. Aggregation:
           - Normal:   trust-weighted mean of trusted clients
           - Fallback: coordinate-wise median of top-k (NO collapse)
    """

    def __init__(
        self,
        trust_manager,
        enable_filtering: bool = True,
        enable_norm_clip: bool = True,
        clip_multiplier: float = 1.5,       # tighter than before (was 2.0)
        warmup_rounds: int = 0,
        fallback_top_k_ratio: float = 0.3,  # used when trusted = empty
        use_median_aggregation: bool = False,
        min_trusted_for_median: int = 2,
    ):
        self.trust_manager = trust_manager
        self.enable_filtering = enable_filtering
        self.enable_norm_clip = enable_norm_clip
        self.clip_multiplier = clip_multiplier
        self.warmup_rounds = warmup_rounds
        self.fallback_top_k_ratio = fallback_top_k_ratio
        self.use_median_aggregation = use_median_aggregation
        self.min_trusted_for_median = min_trusted_for_median

        # Stats for ablation reporting
        self._total_rounds = 0
        self._fallback_rounds = 0
        self._total_filtered = 0
        self._total_clipped = 0
        self._median_rounds = 0

    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
        metrics: Optional[List[Dict[str, float]]] = None,
        round_num: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates with full defense pipeline.

        CRITICAL: Never returns None — fallback ensures model always updates.
        """
        n = len(updates)
        assert n > 0, "Need at least one update"
        self._total_rounds += 1

        # ── Step 1: Norm clipping ────────────────────────────────
        clipped_updates = updates
        num_clipped = 0

        if self.enable_norm_clip:
            norms = [_compute_norm(u) for u in updates]
            median_norm = float(np.median(norms))
            clip_norm = self.clip_multiplier * max(median_norm, 1e-6)

            clipped_updates = []
            for u, norm in zip(updates, norms):
                if norm > clip_norm:
                    clipped_updates.append(_clip_by_norm(u, clip_norm))
                    num_clipped += 1
                else:
                    clipped_updates.append(u)

        self._total_clipped += num_clipped

        # ── Step 2: Trust filtering ──────────────────────────────
        trusted_ids: List[int] = client_ids  # default: no filtering
        filtered_count = 0

        if self.enable_filtering and round_num >= self.warmup_rounds:
            trusted_ids = self.trust_manager.get_trusted_clients(
                client_ids, round_num
            )
            filtered_count = len(client_ids) - len(trusted_ids)
            self._total_filtered += filtered_count

            if len(trusted_ids) == 0:
                # ── FALLBACK: top-k by trust score ───────────────
                self._fallback_rounds += 1
                k = max(1, int(len(client_ids) * self.fallback_top_k_ratio))
                ranked = sorted(
                    range(len(client_ids)),
                    key=lambda i: self.trust_manager.get_trust_score(client_ids[i]),
                    reverse=True,
                )
                top_k_indices = ranked[:k]
                fallback_updates = [clipped_updates[i] for i in top_k_indices]
                result = _coordinate_median(fallback_updates)
                return self._safety_clip(result, fallback_updates)

        # Build trusted update list
        trusted_updates = []
        trusted_client_ids = []
        for i, cid in enumerate(client_ids):
            if cid in set(trusted_ids):
                trusted_updates.append(clipped_updates[i])
                trusted_client_ids.append(cid)

        # ── Step 3: Aggregation ──────────────────────────────────
        n_trusted = len(trusted_updates)

        if (self.use_median_aggregation
                and n_trusted >= self.min_trusted_for_median):
            self._median_rounds += 1
            return _coordinate_median(trusted_updates)

        # Trust-weighted mean (default)
        weights = self.trust_manager.get_trust_weights(trusted_client_ids)
        aggregated = _weighted_mean(trusted_updates, weights)
        return self._safety_clip(aggregated, trusted_updates)

    def compute_reference(
        self,
        updates: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
        malicious_ids: Optional[Set[int]] = None,
        use_robust: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute robust reference gradient for trust scoring.

        Production mode (default): coordinate-wise median of ALL updates.
        Research mode: mean of known benign updates (upper bound).
        """
        if malicious_ids is not None:
            benign = [updates[i] for i, cid in enumerate(client_ids)
                      if cid not in malicious_ids]
            if benign:
                n = len(benign)
                ref = {}
                for name in benign[0]:
                    ref[name] = torch.mean(
                        torch.stack([u[name] for u in benign]), dim=0
                    )
                return ref

        if use_robust and len(updates) >= 2:
            return _coordinate_median(updates)

        n = len(updates)
        ref = {}
        for name in updates[0]:
            ref[name] = torch.mean(
                torch.stack([u[name] for u in updates]), dim=0
            )
        return ref

    def _safety_clip(
        self,
        aggregated: Dict[str, torch.Tensor],
        reference_updates: List[Dict[str, torch.Tensor]],
        max_multiplier: float = 3.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Post-aggregation safety guard.
        If aggregated update norm >> median of input norms, scale it down.
        Prevents model destruction from adversarial median/mean in edge cases.
        """
        agg_norm = _compute_norm(aggregated)
        if agg_norm < 1e-8:
            return aggregated

        ref_norms = [_compute_norm(u) for u in reference_updates]
        median_ref_norm = float(np.median(ref_norms))

        max_allowed = max_multiplier * max(median_ref_norm, 1e-6)
        if agg_norm > max_allowed:
            scale = max_allowed / agg_norm
            return {k: v * scale for k, v in aggregated.items()}
        return aggregated

    def get_stats(self) -> Dict:
        return {
            'total_rounds':    self._total_rounds,
            'fallback_rounds': self._fallback_rounds,
            'fallback_rate':   self._fallback_rounds / max(1, self._total_rounds),
            'avg_filtered':    self._total_filtered / max(1, self._total_rounds),
            'avg_clipped':     self._total_clipped / max(1, self._total_rounds),
            'median_rounds':   self._median_rounds,
        }


# ══════════════════════════════════════════════════════════════════════
# Baseline aggregators (unchanged API, updated signature)
# ══════════════════════════════════════════════════════════════════════

class FedAvgAggregator:
    """Standard FedAvg — uniform average (no defense baseline)."""

    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
        metrics: Optional[List[Dict]] = None,
        round_num: int = 0,
    ) -> Dict[str, torch.Tensor]:
        n = len(updates)
        result = {}
        for name in updates[0].keys():
            result[name] = sum(u[name] for u in updates) / n
        return result

    def get_stats(self) -> Dict:
        return {'fallback_rounds': 0, 'avg_filtered': 0.0, 'fallback_rate': 0.0}


class KrumAggregator:
    """Krum — Byzantine-robust baseline."""

    def __init__(self, num_malicious: int = 0):
        self.num_malicious = num_malicious

    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
        metrics: Optional[List[Dict]] = None,
        round_num: int = 0,
    ) -> Dict[str, torch.Tensor]:
        n = len(updates)
        f = self.num_malicious
        flat = [torch.cat([p.flatten() for p in u.values()]) for u in updates]
        scores = []
        for i in range(n):
            dists = sorted(
                torch.norm(flat[i] - flat[j]).item()
                for j in range(n) if i != j
            )
            scores.append(sum(dists[:max(1, n - f - 2)]))
        best = int(np.argmin(scores))
        return {k: v.clone() for k, v in updates[best].items()}

    def get_stats(self) -> Dict:
        return {'fallback_rounds': 0, 'avg_filtered': 0.0, 'fallback_rate': 0.0}


class TrimmedMeanAggregator:
    """Trimmed Mean — coordinate-wise robust aggregation."""

    def __init__(self, trim_ratio: float = 0.1):
        self.trim_ratio = trim_ratio

    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        client_ids: List[int],
        metrics: Optional[List[Dict]] = None,
        round_num: int = 0,
    ) -> Dict[str, torch.Tensor]:
        n = len(updates)
        trim_count = int(n * self.trim_ratio)
        result = {}
        for name in updates[0].keys():
            stacked = torch.stack([u[name] for u in updates])
            sorted_vals, _ = torch.sort(stacked, dim=0)
            if trim_count > 0 and 2 * trim_count < n:
                trimmed = sorted_vals[trim_count:-trim_count]
            else:
                trimmed = sorted_vals
            result[name] = torch.mean(trimmed, dim=0)
        return result

    def get_stats(self) -> Dict:
        return {'fallback_rounds': 0, 'avg_filtered': 0.0, 'fallback_rate': 0.0}