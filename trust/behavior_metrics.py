"""
Behavior Metrics Module
Computes behavior-based metrics used in trust scoring.

Metrics:
    1. Gradient cosine similarity
    2. Gradient norm ratio
    3. Loss consistency
    4. Update direction consistency
    5. Anomaly score
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


class BehaviorMetrics:
    """
    Computes various behavior metrics for trust evaluation.
    
    These metrics complement the primary cosine similarity signal
    and can be combined to create a richer trust estimate.
    """

    # ------------------------------------------------------------------
    # Gradient-based metrics
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(
        update_a: Dict[str, torch.Tensor],
        update_b: Dict[str, torch.Tensor]
    ) -> float:
        """
        Cosine similarity between two model updates.

        Args:
            update_a: First gradient dict
            update_b: Second gradient dict

        Returns:
            similarity: [-1, 1]
        """
        flat_a = torch.cat([v.flatten() for v in update_a.values()])
        flat_b = torch.cat([v.flatten() for v in update_b.values()])

        norm_a = torch.norm(flat_a)
        norm_b = torch.norm(flat_b)

        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0

        return float(torch.dot(flat_a, flat_b) / (norm_a * norm_b))

    @staticmethod
    def gradient_norm(update: Dict[str, torch.Tensor]) -> float:
        """Compute L2 norm of gradient."""
        total = sum(torch.norm(v).item() ** 2 for v in update.values())
        return float(np.sqrt(total))

    @staticmethod
    def norm_ratio(
        client_update: Dict[str, torch.Tensor],
        reference_update: Dict[str, torch.Tensor]
    ) -> float:
        """
        Ratio of client norm to reference norm.

        Values far from 1.0 indicate suspicious behavior.
        Returns ratio clamped to [0, 10].
        """
        client_norm = BehaviorMetrics.gradient_norm(client_update)
        ref_norm = BehaviorMetrics.gradient_norm(reference_update)

        if ref_norm < 1e-10:
            return 1.0

        ratio = client_norm / ref_norm
        return float(np.clip(ratio, 0.0, 10.0))

    @staticmethod
    def direction_consistency(
        update: Dict[str, torch.Tensor],
        history: List[Dict[str, torch.Tensor]]
    ) -> float:
        """
        How consistent is the update direction compared to recent history?

        Args:
            update:  Current update
            history: List of recent previous updates

        Returns:
            consistency: Mean cosine similarity with history [0, 1]
        """
        if not history:
            return 0.5

        similarities = [
            BehaviorMetrics.cosine_similarity(update, prev)
            for prev in history[-5:]   # use last 5
        ]
        mean_sim = float(np.mean(similarities))
        # Map [-1, 1] → [0, 1]
        return (mean_sim + 1.0) / 2.0

    # ------------------------------------------------------------------
    # Loss-based metrics
    # ------------------------------------------------------------------

    @staticmethod
    def loss_z_score(current_loss: float, loss_history: List[float]) -> float:
        """
        Z-score of current loss relative to client's own history.

        High z-score (|z| >> 0) → unusual loss → suspicious.
        
        Returns:
            z_score: Unsigned z-score (0 = normal, high = suspicious)
        """
        if len(loss_history) < 3:
            return 0.0

        mean = float(np.mean(loss_history))
        std = float(np.std(loss_history))

        if std < 1e-10:
            return 0.0

        return abs((current_loss - mean) / std)

    @staticmethod
    def loss_to_trust_signal(z_score: float, scale: float = 2.0) -> float:
        """
        Convert loss z-score to a trust signal [0, 1].

        z_score = 0 → trust = 1.0 (perfect)
        z_score = scale → trust ≈ 0.5
        z_score >> scale → trust → 0

        Args:
            z_score: Loss anomaly score
            scale:   Z-score that maps to trust=0.5

        Returns:
            trust_signal: [0, 1]
        """
        return float(1.0 / (1.0 + (z_score / scale) ** 2))

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------

    @staticmethod
    def norm_anomaly_score(
        client_update: Dict[str, torch.Tensor],
        all_updates: List[Dict[str, torch.Tensor]]
    ) -> float:
        """
        Detect norm outliers among all client updates.

        Computes how many standard deviations the client's norm is
        from the mean of all norms.

        Returns:
            anomaly_score: 0 = normal, >2 = suspicious
        """
        if not all_updates:
            return 0.0

        all_norms = [BehaviorMetrics.gradient_norm(u) for u in all_updates]
        client_norm = BehaviorMetrics.gradient_norm(client_update)

        mean_norm = float(np.mean(all_norms))
        std_norm = float(np.std(all_norms))

        if std_norm < 1e-10:
            return 0.0

        return abs((client_norm - mean_norm) / std_norm)

    @staticmethod
    def compute_composite_score(
        cosine_sim: float,
        norm_ratio: float,
        loss_signal: float = 0.5,
        weights: Optional[Tuple[float, float, float]] = None
    ) -> float:
        """
        Compute a composite trust observation from multiple signals.

        Args:
            cosine_sim:  Cosine similarity signal [0, 1]
            norm_ratio:  Gradient norm ratio (1.0 = normal)
            loss_signal: Loss-based trust signal [0, 1]
            weights:     (w_cosine, w_norm, w_loss), default (0.6, 0.2, 0.2)

        Returns:
            composite: Combined trust observation [0, 1]
        """
        if weights is None:
            weights = (0.6, 0.2, 0.2)

        w_cos, w_norm, w_loss = weights

        # Norm signal: 1.0 if norm_ratio close to 1, decreases otherwise
        norm_signal = 1.0 / (1.0 + abs(norm_ratio - 1.0))

        composite = w_cos * cosine_sim + w_norm * norm_signal + w_loss * loss_signal
        return float(np.clip(composite, 0.0, 1.0))


class ClientBehaviorTracker:
    """
    Per-client behavior tracker that maintains a window of recent
    metrics and detects behavioral shifts indicative of attacks.
    """

    def __init__(self, client_id: int, window_size: int = 10):
        self.client_id = client_id
        self.window_size = window_size

        self.similarity_history: List[float] = []
        self.norm_history: List[float] = []
        self.loss_history: List[float] = []
        self.trust_history: List[float] = []
        self.update_history: List[Dict[str, torch.Tensor]] = []  # last updates

    def record(
        self,
        update: Dict[str, torch.Tensor],
        similarity: float,
        loss: float,
        trust: float
    ):
        """Record a new round's metrics."""
        norm = BehaviorMetrics.gradient_norm(update)

        self.similarity_history.append(similarity)
        self.norm_history.append(norm)
        self.loss_history.append(loss)
        self.trust_history.append(trust)

        # Keep a small window of actual update tensors
        if len(self.update_history) >= self.window_size:
            self.update_history.pop(0)
        self.update_history.append({k: v.detach().clone() for k, v in update.items()})

        # Trim histories
        for h in [self.similarity_history, self.norm_history,
                  self.loss_history, self.trust_history]:
            if len(h) > self.window_size:
                h.pop(0)

    def get_recent_similarity(self) -> float:
        """Average cosine similarity over recent window."""
        if not self.similarity_history:
            return 0.5
        return float(np.mean(self.similarity_history))

    def detect_behavioral_shift(self, threshold: float = 0.3) -> bool:
        """
        Detect a sudden drop in similarity (possible attack onset).

        Returns True if recent average is much lower than earlier average.
        """
        n = len(self.similarity_history)
        if n < 6:
            return False

        early = float(np.mean(self.similarity_history[:n // 2]))
        late = float(np.mean(self.similarity_history[n // 2:]))

        return (early - late) > threshold

    def get_statistics(self) -> Dict:
        """Return summary statistics."""
        def safe_stats(h):
            return {
                'mean': float(np.mean(h)) if h else 0.0,
                'std': float(np.std(h)) if h else 0.0,
                'last': h[-1] if h else 0.0
            }

        return {
            'client_id': self.client_id,
            'num_observations': len(self.similarity_history),
            'similarity': safe_stats(self.similarity_history),
            'norm': safe_stats(self.norm_history),
            'loss': safe_stats(self.loss_history),
            'trust': safe_stats(self.trust_history),
            'behavioral_shift_detected': self.detect_behavioral_shift()
        }