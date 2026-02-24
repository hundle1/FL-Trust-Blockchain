"""
Trust Score Manager
Core trust mechanism for Trust-Aware Federated Learning

Trust update formula:
    T_i(t+1) = α * T_i(t) + (1 - α) * S_i(t)

Where:
    - α: decay parameter (memory factor)
    - S_i(t): gradient similarity score at round t
    - T_i(t): trust score at round t
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
        alpha: float = 0.9,
        tau: float = 0.3,
        initial_trust: float = 1.0,
        enable_decay: bool = True,
        decay_strategy: str = "exponential",
        window_size: int = 10,
        min_trust: float = 0.0,
        max_trust: float = 1.0
    ):
        """
        Args:
            num_clients:     Total number of clients
            alpha:           EMA decay factor (0~1). Higher = slower trust change
            tau:             Trust threshold for filtering malicious clients
            initial_trust:   Initial trust score for all clients
            enable_decay:    Whether to apply trust decay
            decay_strategy:  Decay strategy (exponential, threshold, adaptive)
            window_size:     History window for behavior tracking
            min_trust:       Minimum allowed trust score
            max_trust:       Maximum allowed trust score
        """
        self.num_clients = num_clients
        self.alpha = alpha
        self.tau = tau
        self.initial_trust = initial_trust
        self.enable_decay = enable_decay
        self.decay_strategy = decay_strategy
        self.min_trust = min_trust
        self.max_trust = max_trust

        # Trust scores: initialized to initial_trust
        self.trust_scores = np.full(num_clients, initial_trust, dtype=np.float64)

        # History tracking
        self.history_manager = ClientHistoryManager(num_clients, window_size)

        # Round participation tracking
        self.last_participated = np.full(num_clients, -1, dtype=int)
        self.participation_count = np.zeros(num_clients, dtype=int)

        # Statistics
        self.update_count = 0
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
        Compute cosine similarity between client update and reference gradient.

        Args:
            client_update:       Client's model update (delta)
            reference_gradient:  Reference gradient (e.g., average of benign clients)

        Returns:
            similarity: Cosine similarity in [−1, 1], mapped to [0, 1]
        """
        client_flat = torch.cat([v.flatten() for v in client_update.values()])
        ref_flat = torch.cat([v.flatten() for v in reference_gradient.values()])

        norm_client = torch.norm(client_flat)
        norm_ref = torch.norm(ref_flat)

        if norm_client < 1e-10 or norm_ref < 1e-10:
            return 0.0

        cosine_sim = torch.dot(client_flat, ref_flat) / (norm_client * norm_ref)
        cosine_sim = cosine_sim.item()

        # Map from [-1, 1] → [0, 1]
        similarity = (cosine_sim + 1.0) / 2.0
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

        Args:
            client_id:          Client identifier
            client_update:      Model update from client
            reference_gradient: Reference gradient for similarity computation
            metrics:            Optional training metrics (loss, accuracy, gradient_norm)
            round_num:          Current training round

        Returns:
            new_trust: Updated trust score
        """
        old_trust = self.trust_scores[client_id]

        # 1. Compute gradient similarity (primary signal)
        similarity = self.compute_gradient_similarity(client_update, reference_gradient)

        # 2. Optionally incorporate loss-based signal
        if metrics is not None:
            loss_signal = self._loss_signal(client_id, metrics.get('loss', None))
            # Blend: 80% gradient similarity, 20% loss signal
            observation = 0.8 * similarity + 0.2 * loss_signal
        else:
            observation = similarity

        # 3. Apply decay strategy
        if self.enable_decay:
            if self.decay_strategy == "threshold":
                new_trust = TrustDecay.threshold_decay(old_trust, observation)
            elif self.decay_strategy == "adaptive":
                variance = np.var(
                    self.history_manager.get_gradient_history(client_id) or [0.0]
                )
                new_trust = TrustDecay.adaptive_decay(old_trust, observation, variance, self.alpha)
            else:  # exponential (default)
                new_trust = TrustDecay.exponential_decay(old_trust, observation, self.alpha)
        else:
            # Simple EMA without named strategy
            new_trust = self.alpha * old_trust + (1 - self.alpha) * observation

        # 4. Clip to valid range
        new_trust = float(np.clip(new_trust, self.min_trust, self.max_trust))

        # 5. Store
        self.trust_scores[client_id] = new_trust
        self.last_participated[client_id] = round_num
        self.participation_count[client_id] += 1
        self.update_count += 1

        # 6. Update history buffers
        self.history_manager.add_gradient_similarity(client_id, similarity)
        self.history_manager.add_trust(client_id, new_trust)
        if metrics:
            self.history_manager.add_loss(client_id, metrics.get('loss', 0.0))
            self.history_manager.add_accuracy(client_id, metrics.get('accuracy', 0.0))

        return new_trust

    def _loss_signal(self, client_id: int, loss: Optional[float]) -> float:
        """
        Convert loss to a trust signal [0, 1].
        Compares client loss to its own historical average.
        """
        if loss is None:
            return 0.5  # neutral

        history = self.history_manager.get_loss_history(client_id)
        if not history:
            return 0.5

        avg_loss = np.mean(history)
        if avg_loss < 1e-10:
            return 0.5

        ratio = loss / avg_loss
        # ratio close to 1 → normal; ratio >> 1 → suspicious
        signal = 1.0 / (1.0 + max(0.0, ratio - 1.0))
        return float(np.clip(signal, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Idle decay (applied each round for non-participating clients)
    # ------------------------------------------------------------------

    def apply_idle_decay(self, active_client_ids: List[int], round_num: int, decay_rate: float = 0.005):
        """
        Apply slight trust decay to clients that did NOT participate this round.
        Prevents attackers from building up trust while idle.

        Args:
            active_client_ids: Clients that participated this round
            round_num:         Current round number
            decay_rate:        Decay rate per missed round
        """
        if not self.enable_decay:
            return

        active_set = set(active_client_ids)
        for cid in range(self.num_clients):
            if cid not in active_set:
                self.trust_scores[cid] = TrustDecay.linear_decay(
                    self.trust_scores[cid], decay_rate
                )

    # ------------------------------------------------------------------
    # Querying trust scores
    # ------------------------------------------------------------------

    def get_trust_score(self, client_id: int) -> float:
        """Get current trust score for a client."""
        return float(self.trust_scores[client_id])

    def get_all_trust_scores(self) -> np.ndarray:
        """Get trust scores for all clients."""
        return self.trust_scores.copy()

    def get_trusted_clients(self, client_ids: List[int]) -> List[int]:
        """
        Filter clients to only those above trust threshold τ.

        Args:
            client_ids: Candidate client IDs

        Returns:
            trusted_ids: Client IDs with trust ≥ τ
        """
        return [cid for cid in client_ids if self.trust_scores[cid] >= self.tau]

    def get_trust_weights(self, client_ids: List[int]) -> List[float]:
        """
        Compute normalized trust weights for aggregation.

        Args:
            client_ids: List of client IDs

        Returns:
            weights: Normalized weights summing to 1
        """
        scores = np.array([self.trust_scores[cid] for cid in client_ids])

        total = scores.sum()
        if total < 1e-10:
            # Uniform fallback
            return [1.0 / len(client_ids)] * len(client_ids)

        return (scores / total).tolist()

    # ------------------------------------------------------------------
    # Statistics & reporting
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict:
        """Get summary statistics of trust scores."""
        scores = self.trust_scores

        trusted_mask = scores >= self.tau
        untrusted_mask = ~trusted_mask

        return {
            'mean_trust': float(np.mean(scores)),
            'std_trust': float(np.std(scores)),
            'min_trust': float(np.min(scores)),
            'max_trust': float(np.max(scores)),
            'num_trusted': int(trusted_mask.sum()),
            'num_untrusted': int(untrusted_mask.sum()),
            'trusted_ratio': float(trusted_mask.sum() / self.num_clients),
            'tau': self.tau,
            'alpha': self.alpha,
            'total_updates': self.update_count
        }

    def get_trust_separation(
        self,
        benign_ids: List[int],
        malicious_ids: List[int]
    ) -> Dict[str, float]:
        """
        Compute separation between benign and malicious trust scores.
        Higher separation = better discrimination.
        """
        benign_scores = [self.trust_scores[i] for i in benign_ids if i < self.num_clients]
        malicious_scores = [self.trust_scores[i] for i in malicious_ids if i < self.num_clients]

        if not benign_scores or not malicious_scores:
            return {'separation': 0.0, 'avg_benign': 0.0, 'avg_malicious': 0.0}

        avg_benign = float(np.mean(benign_scores))
        avg_malicious = float(np.mean(malicious_scores))

        return {
            'separation': avg_benign - avg_malicious,
            'avg_benign': avg_benign,
            'avg_malicious': avg_malicious,
            'benign_std': float(np.std(benign_scores)),
            'malicious_std': float(np.std(malicious_scores))
        }

    def get_client_history(self, client_id: int) -> Dict:
        """Get full history for a specific client."""
        return self.history_manager.get_statistics(client_id)

    def reset_client(self, client_id: int):
        """Reset trust score and history for a client."""
        self.trust_scores[client_id] = self.initial_trust
        self.history_manager.clear_client(client_id)
        self.participation_count[client_id] = 0

    def print_trust_table(self, top_n: int = 20):
        """Print trust scores in a formatted table."""
        print(f"\n{'='*50}")
        print(f"{'Client':<10} {'Trust Score':<15} {'Status':<10} {'Rounds':<10}")
        print(f"{'-'*50}")

        indices = np.argsort(self.trust_scores)[::-1][:top_n]
        for cid in indices:
            score = self.trust_scores[cid]
            status = "TRUSTED" if score >= self.tau else "BLOCKED"
            rounds = self.participation_count[cid]
            print(f"{cid:<10} {score:<15.4f} {status:<10} {rounds:<10}")
        print(f"{'='*50}")