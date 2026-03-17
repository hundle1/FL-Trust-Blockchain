"""
Trust Score Manager — v3 STRONG DEFENSE
=========================================
Phân tích gốc rễ từ kết quả thực nghiệm:

  PROBLEM 1: alpha=0.75 vẫn quá cao
    → Sau 10 rounds bị tấn công, trust malicious vẫn ~0.53
      (0.75^10 * 0.5 + correction ≈ 0.5+) → KHÔNG bị filter
    FIX: alpha=0.5 → trust malicious xuống dưới tau sau ~5 rounds

  PROBLEM 2: initial_trust=0.5 == tau=0.5
    → Client mới NGAY LẬP TỨC được include
    → Attacker round 0 đã có thể poison ngay
    FIX: initial_trust=0.3 < tau=0.45

  PROBLEM 3: tau=0.5 nhưng benign trust converge về ~0.6
    → Margin chỉ 0.1, quá nhỏ → noise dễ gây false positive
    → Đồng thời malicious cần xuống 0.49 mới bị filter
    FIX: tau=0.45 (giảm một chút để benign có margin rộng hơn)
         kết hợp alpha thấp hơn để malicious xuống nhanh hơn

  PROBLEM 4: Similarity weight 0.9 nhưng norm của poisoned gradient
    rất lớn (scale=5 * layer_boost=2.5 → 12.5x benign)
    → cosine sim bị kéo về -1 → observation = 0 → trust giảm
    → Nhưng với alpha=0.75: T_new = 0.75*0.5 + 0.25*0 = 0.375
    → Cần ~3 rounds để xuống dưới tau=0.5
    → Static attacker tấn công mọi round → 3 rounds đầu vẫn poison được
    FIX: alpha=0.5 → T_new = 0.5*0.5 + 0.5*0 = 0.25 → filter ngay round 2

  PROBLEM 5: Không có norm-based anomaly detection
    → Gradient flip với scale=12.5x không bị detect bởi cosine sim alone
    → Thêm norm ratio penalty vào observation

  NEW FEATURE: Norm anomaly penalty
    Nếu client gradient norm >> benign_norm_estimate:
      observation bị nhân thêm penalty factor → trust giảm nhanh hơn

Trust update formula:
    S_i(t) = sim_weight * cosine_sim + loss_weight * loss_signal
    S_i(t) *= norm_penalty(client)   ← NEW
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

    v3 STRONG DEFENSE changes vs v2:
      - alpha: 0.75 → 0.5   (trust thay đổi 2x nhanh hơn)
      - tau: 0.5 → 0.45     (benign có margin rộng hơn, malicious bị filter sớm hơn)
      - initial_trust: 0.5 → 0.3  (không include ngay từ đầu)
      - idle_decay_rate: 0.008 → 0.01
      - NEW: norm_clip_percentile — detect norm outliers
      - NEW: warmup_rounds — trust không được dùng để filter trong N rounds đầu
               (tránh false positive khi model chưa ổn định)
    """

    def __init__(
        self,
        num_clients: int,
        alpha: float = 0.5,              # STRONG: was 0.75 → trust đổi nhanh gấp 2x
        tau: float = 0.45,               # STRONG: was 0.5 → filter với margin tốt hơn
        initial_trust: float = 0.3,      # STRONG: was 0.5 → không include ngay
        enable_decay: bool = True,
        decay_strategy: str = "exponential",
        window_size: int = 5,
        min_trust: float = 0.0,
        max_trust: float = 1.0,
        similarity_weight: float = 0.9,
        idle_decay_rate: float = 0.01,   # STRONG: was 0.008
        # NEW params
        enable_norm_penalty: bool = True,     # Phạt client có norm >> benign norm
        norm_penalty_threshold: float = 2.0,  # norm/benign_norm > 2x → penalty (was 3.0)
        norm_penalty_strength: float = 0.6,   # penalty factor — tăng để trust malicious giảm nhanh hơn
        warmup_rounds: int = 3,               # Không filter trong N rounds đầu
    ):
        self.num_clients        = num_clients
        self.alpha              = alpha
        self.tau                = tau
        self.initial_trust      = initial_trust
        self.enable_decay       = enable_decay
        self.decay_strategy     = decay_strategy
        self.min_trust          = min_trust
        self.max_trust          = max_trust
        self.similarity_weight  = similarity_weight
        self.idle_decay_rate    = idle_decay_rate
        self.enable_norm_penalty    = enable_norm_penalty
        self.norm_penalty_threshold = norm_penalty_threshold
        self.norm_penalty_strength  = norm_penalty_strength
        self.warmup_rounds      = warmup_rounds

        # Trust scores
        self.trust_scores = np.full(num_clients, initial_trust, dtype=np.float64)

        # History tracking
        self.history_manager = ClientHistoryManager(num_clients, window_size)

        # Round participation tracking
        self.last_participated   = np.full(num_clients, -1, dtype=int)
        self.participation_count = np.zeros(num_clients, dtype=int)

        # Norm tracking — running estimate of benign gradient norms
        # Used for norm anomaly detection
        self._norm_history: List[float] = []   # rolling window of recent norms
        self._norm_window_size = 20

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

    def _compute_norm(self, update: Dict[str, torch.Tensor]) -> float:
        """L2 norm của gradient."""
        total = sum(torch.norm(v).item() ** 2 for v in update.values())
        return float(np.sqrt(total))

    def _norm_penalty(self, client_norm: float) -> float:
        """
        NEW: Tính penalty factor dựa trên norm ratio.

        Nếu client gradient norm >> benign estimate:
          → penalty < 1.0 → observation bị giảm → trust giảm nhanh hơn

        Trả về: penalty ∈ [1 - norm_penalty_strength, 1.0]
          = 1.0 nếu norm bình thường
          < 1.0 nếu norm bất thường (quá lớn)
        """
        if not self.enable_norm_penalty or len(self._norm_history) < 3:
            return 1.0   # Chưa đủ data → không penalty (3 samples đủ để ước lượng sớm)

        benign_norm_est = float(np.median(self._norm_history))
        if benign_norm_est < 1e-8:
            return 1.0

        ratio = client_norm / benign_norm_est

        if ratio <= self.norm_penalty_threshold:
            return 1.0   # Bình thường → không penalty

        # Penalty tỷ lệ với mức độ vượt threshold (threshold=2.0)
        # ratio = 2.0 → penalty = 1.0 (no penalty)
        # ratio = 4.0 → penalty = 1 - strength * excess
        # ratio rất lớn → penalty = 1 - norm_penalty_strength (floor)
        excess = (ratio - self.norm_penalty_threshold) / self.norm_penalty_threshold
        penalty = 1.0 - min(self.norm_penalty_strength, self.norm_penalty_strength * excess)
        return float(np.clip(penalty, 1.0 - self.norm_penalty_strength, 1.0))

    def update_trust(
        self,
        client_id: int,
        client_update: Dict[str, torch.Tensor],
        reference_gradient: Dict[str, torch.Tensor],
        metrics: Optional[Dict[str, float]] = None,
        round_num: int = 0
    ) -> float:
        """
        Update trust score cho client sau khi nhận update.

        v3 changes:
          - Thêm norm penalty trước EMA
          - alpha nhỏ hơn → phản ứng nhanh hơn
        """
        old_trust = self.trust_scores[client_id]

        # 1. Cosine similarity (primary signal)
        similarity = self.compute_gradient_similarity(client_update, reference_gradient)

        # 2. Loss-based signal (secondary)
        loss_weight = 1.0 - self.similarity_weight
        if metrics is not None and metrics.get('loss') is not None:
            loss_signal = self._loss_signal(client_id, metrics['loss'])
            observation = self.similarity_weight * similarity + loss_weight * loss_signal
        else:
            observation = similarity

        # 3. NEW: Norm penalty
        client_norm = self._compute_norm(client_update)
        penalty     = self._norm_penalty(client_norm)
        observation = observation * penalty   # giảm trust nếu norm bất thường

        # 4. Update benign norm estimate
        #    Chỉ update nếu observation >= 0.35 (likely benign) — threshold hạ để build estimate sớm
        #    Tránh bị poisoned update kéo lệch estimate
        if observation >= 0.35:
            self._norm_history.append(client_norm)
            if len(self._norm_history) > self._norm_window_size:
                self._norm_history.pop(0)

        # 5. EMA decay — nhanh hơn khi rõ ràng malicious (flipped gradient → observation ~0)
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
                # Fast decay when clearly malicious (obs < 0.2) → trust drop in 1 round
                alpha_eff = self.alpha
                if observation < 0.2:
                    alpha_eff = 0.25  # T_new ≈ 0.25*T_old → filter ngay sau 1 round
                new_trust = TrustDecay.exponential_decay(old_trust, observation, alpha_eff)
        else:
            new_trust = self.alpha * old_trust + (1 - self.alpha) * observation

        # 6. Clip
        new_trust = float(np.clip(new_trust, self.min_trust, self.max_trust))

        # 7. Store
        self.trust_scores[client_id]       = new_trust
        self.last_participated[client_id]  = round_num
        self.participation_count[client_id] += 1
        self.update_count += 1

        # 8. Update history
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
        """
        Apply trust decay to non-participating clients.
        v3: rate tăng 0.008 → 0.01
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
    # Querying — với warmup support
    # ------------------------------------------------------------------

    def get_trust_score(self, client_id: int) -> float:
        return float(self.trust_scores[client_id])

    def get_all_trust_scores(self) -> np.ndarray:
        return self.trust_scores.copy()

    def get_trusted_clients(
        self,
        client_ids: List[int],
        round_num: int = 999   # NEW: warmup bypass
    ) -> List[int]:
        """
        Trả về danh sách trusted clients.

        NEW: Trong warmup_rounds đầu, trả về TẤT CẢ clients
        (tránh false positive khi trust chưa calibrate xong)
        Sau warmup → filter bình thường.
        """
        if round_num < self.warmup_rounds:
            return list(client_ids)   # warmup: include all
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
            'total_updates': self.update_count,
            'benign_norm_estimate': float(np.median(self._norm_history))
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
        self.trust_scores[client_id]       = self.initial_trust
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