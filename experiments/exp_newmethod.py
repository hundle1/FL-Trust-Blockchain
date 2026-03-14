"""
Experiment: Trust-Aware Defense vs Attackers (1v1) — FIXED VERSION
====================================================================
Các fix so với bản cũ:

  FIX 1: alpha 0.9 → 0.75  (trust thay đổi nhanh hơn)
  FIX 2: tau 0.3 → 0.5     (filter sớm hơn)
  FIX 3: initial_trust 1.0 → 0.5  (không tin tuyệt đối từ đầu)
  FIX 4: idle_decay 0.005 → 0.008  (decay nhanh hơn khi vắng mặt)
  FIX 5: clients_per_round 10 → 20  (giảm tỷ lệ malicious/benign mỗi round)
  FIX 6: metrics thực từ client.train() thay vì hardcode loss=0.1
  FIX 7: fallback khi no trusted → skip update (giữ model)
         thay vì dùng all poisoned updates
  FIX 8: tách raw_updates + metrics trong 1 lần train duy nhất

Attacker KHÔNG thay đổi — vẫn dùng layer-wise amplification và norm-matching.

──────────────────────────────────────────────────────────────────────
CÁCH CHẠY:

  Chạy toàn bộ (6 scenarios, sinh đủ 5 ảnh tổng hợp):
    python experiments/exp_newmethod.py
    python experiments/exp_newmethod.py all

  Chạy từng scenario riêng lẻ:
    python experiments/exp_newmethod.py exp1    # no_attack    (baseline)
    python experiments/exp_newmethod.py exp2    # static
    python experiments/exp_newmethod.py exp3    # delayed
    python experiments/exp_newmethod.py exp4    # adaptive
    python experiments/exp_newmethod.py exp5    # intermittent
    python experiments/exp_newmethod.py exp6    # norm_tuned

  Hoặc dùng tên attack trực tiếp:
    python experiments/exp_newmethod.py static
    python experiments/exp_newmethod.py norm_tuned

  Khi chạy riêng lẻ → sinh 2 ảnh riêng cho scenario đó:
    results/figures/exp_newmethod_<attack>_accuracy.png
    results/figures/exp_newmethod_<attack>_trust.png
──────────────────────────────────────────────────────────────────────

Outputs khi chạy all (5 ảnh tổng hợp):
    results/figures/exp_newmethod_accuracy.png
    results/figures/exp_newmethod_loss.png
    results/figures/exp_newmethod_asr.png
    results/figures/exp_newmethod_heatmap.png
    results/figures/exp_newmethod_trust.png
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from models.cnn_mnist import get_model
from fl_core.client import FLClient
from trust.trust_score import TrustScoreManager
from fl_core.aggregation.trust_aware import TrustAwareAggregator
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


# ══════════════════════════════════════════════════════════════════════
# CONFIG — FIXED PARAMS
# ══════════════════════════════════════════════════════════════════════

ATTACK_RATE        = 0.20
POISONING_SCALE    = 5.0
NUM_CLIENTS        = 100
CLIENTS_PER_ROUND  = 20    # FIX 5: was 10 → giảm tỷ lệ malicious mỗi round
NUM_ROUNDS         = 80
SEED               = 42

# FIX 1,2,3,4: defense params
ALPHA         = 0.75    # was 0.9
TAU           = 0.5     # was 0.3
INITIAL_TRUST = 0.5     # was 1.0
IDLE_DECAY    = 0.008   # was 0.005
WINDOW_SIZE   = 5       # was 10
SIM_WEIGHT    = 0.9     # was 0.8

ATTACK_ORDER = ["no_attack", "static", "delayed", "adaptive", "intermittent", "norm_tuned"]

ATTACK_DISPLAY = {
    "no_attack":    "No Attack",
    "static":       "Static",
    "delayed":      "Delayed",
    "adaptive":     "Adaptive",
    "intermittent": "Intermittent",
    "norm_tuned":   "Norm-Tuned",
}

PALETTE = {
    "no_attack":    "#2ecc71",
    "static":       "#e74c3c",
    "delayed":      "#e67e22",
    "adaptive":     "#8e44ad",
    "intermittent": "#2980b9",
    "norm_tuned":   "#c0392b",
}

LINESTYLES = {
    "no_attack":    (0, ()),
    "static":       (0, (6, 2)),
    "delayed":      (0, (2, 2)),
    "adaptive":     (0, (6, 2, 2, 2)),
    "intermittent": (0, (4, 1, 1, 1)),
    "norm_tuned":   (0, (1, 1)),
}

LINEWIDTHS = {
    "no_attack":    2.8,
    "static":       2.2,
    "delayed":      2.0,
    "adaptive":     2.0,
    "intermittent": 1.8,
    "norm_tuned":   2.0,
}


# ══════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════

def load_data(num_clients: int = NUM_CLIENTS):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST('./data/mnist', train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST('./data/mnist', train=False, download=True, transform=transform)
    spc = len(train_ds) // num_clients
    client_datasets = [
        DataLoader(Subset(train_ds, list(range(i * spc, (i + 1) * spc))),
                   batch_size=32, shuffle=True)
        for i in range(num_clients)
    ]
    return client_datasets, DataLoader(test_ds, batch_size=1000, shuffle=False)


# ══════════════════════════════════════════════════════════════════════
# ATTACKER — KHÔNG THAY ĐỔI
# Layer-wise amplification + norm-matching để tấn công mạnh nhất có thể
# ══════════════════════════════════════════════════════════════════════

class Attacker:
    def __init__(
        self,
        client_id: int,
        attack_type: str = "static",
        poisoning_scale: float = POISONING_SCALE,
        delay_rounds: int = 10,
        trust_threshold: float = 0.55,
        dormant_threshold: float = 0.25,
        recovery_rounds: int = 5,
        attack_probability: float = 0.5,
        target_norm_ratio: float = 1.5,
    ):
        self.client_id          = client_id
        self.attack_type        = attack_type
        self.poisoning_scale    = poisoning_scale
        self.delay_rounds       = delay_rounds
        self.trust_threshold    = trust_threshold
        self.dormant_threshold  = dormant_threshold
        self.recovery_rounds    = recovery_rounds
        self.attack_probability = attack_probability
        self.target_norm_ratio  = target_norm_ratio
        self.dormant            = False
        self.dormant_since      = -1
        self._benign_norm       = 1.0

    def should_attack(self, round_num: int, est_trust: float = 1.0) -> bool:
        if self.attack_type == "no_attack":
            return False
        elif self.attack_type == "static":
            return True
        elif self.attack_type == "delayed":
            return round_num >= self.delay_rounds
        elif self.attack_type == "adaptive":
            return self._adaptive(round_num, est_trust)
        elif self.attack_type == "intermittent":
            return np.random.random() < self.attack_probability
        elif self.attack_type == "norm_tuned":
            return True
        return True

    def _adaptive(self, round_num: int, trust: float) -> bool:
        if trust < self.dormant_threshold:
            if not self.dormant:
                self.dormant       = True
                self.dormant_since = round_num
            return False
        if self.dormant:
            if round_num - self.dormant_since < self.recovery_rounds:
                return False
            self.dormant = False
        return trust > self.trust_threshold

    def poison_gradient(
        self,
        clean_gradient: Dict[str, torch.Tensor],
        benign_updates: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Dict[str, torch.Tensor]:
        if self.attack_type == "norm_tuned":
            return self._norm_tuned_poison(clean_gradient, benign_updates)
        return self._flip_poison(clean_gradient)

    def _flip_poison(
        self,
        gradient: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        param_names = list(gradient.keys())
        n = len(param_names)
        poisoned = {}
        for i, name in enumerate(param_names):
            layer_boost = 1.0 + 1.5 * (i / max(1, n - 1))
            poisoned[name] = -self.poisoning_scale * layer_boost * gradient[name]
        return poisoned

    def _norm_tuned_poison(
        self,
        gradient: Dict[str, torch.Tensor],
        benign_updates: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Dict[str, torch.Tensor]:
        poisoned = {name: -p for name, p in gradient.items()}
        if benign_updates:
            norms = [
                float(np.sqrt(sum(torch.norm(p).item() ** 2 for p in u.values())))
                for u in benign_updates
            ]
            self._benign_norm = float(np.mean(norms))
        poison_norm = float(np.sqrt(
            sum(torch.norm(p).item() ** 2 for p in poisoned.values())
        ))
        target_norm = self.target_norm_ratio * self._benign_norm
        if poison_norm > 1e-6:
            poisoned = {k: (target_norm / poison_norm) * v
                        for k, v in poisoned.items()}
        return poisoned


# ══════════════════════════════════════════════════════════════════════
# SCENARIO RUNNER — FIXED
# ══════════════════════════════════════════════════════════════════════

def run_scenario(
    attack_type: str,
    client_datasets: list,
    test_loader,
    malicious_ids: set,
    num_rounds: int = NUM_ROUNDS,
) -> Dict:
    """
    Chạy 1 kịch bản: TrustAware (FIXED) vs 1 loại attacker.

    FIXES APPLIED:
      - initial_trust=0.5, alpha=0.75, tau=0.5, idle_decay=0.008
      - Metrics thực từ client.train() (FIX 6)
      - Aggregation trả về None → skip update (FIX 7)
      - Chỉ gọi client.train() 1 lần mỗi round (FIX 8)
    """
    model         = get_model()
    trust_manager = TrustScoreManager(
        num_clients    = NUM_CLIENTS,
        alpha          = ALPHA,
        tau            = TAU,
        initial_trust  = INITIAL_TRUST,
        enable_decay   = True,
        decay_strategy = "exponential",
        window_size    = WINDOW_SIZE,
        similarity_weight = SIM_WEIGHT,
        idle_decay_rate   = IDLE_DECAY,
    )
    aggregator = TrustAwareAggregator(trust_manager, enable_filtering=True)

    clients = [
        FLClient(i, get_model(), client_datasets[i], is_malicious=(i in malicious_ids))
        for i in range(NUM_CLIENTS)
    ]
    attackers = {
        cid: Attacker(cid, attack_type=attack_type)
        for cid in malicious_ids
    }

    history = {
        'accuracy':           [],
        'loss':               [],
        'trust_benign':       [],
        'trust_malicious':    [],
        'attack_rate_actual': [],
        'clients_filtered':   [],
    }

    for round_num in tqdm(
        range(num_rounds),
        desc=f"TrustAware vs {attack_type:<12}",
        leave=False, ncols=85
    ):
        selected_idx = np.random.choice(NUM_CLIENTS, CLIENTS_PER_ROUND, replace=False)
        selected     = [clients[i] for i in selected_idx]

        global_params = {n: p.data.clone() for n, p in model.named_parameters()}
        for c in selected:
            c.set_parameters(global_params)

        # FIX 8: train 1 lần duy nhất, lấy cả update lẫn metrics thực
        updates, client_ids, metrics_list = [], [], []
        attacks_this_round = 0

        for client in selected:
            cid = client.client_id
            client_ids.append(cid)

            # FIX 6: lấy metrics thực từ training
            raw_update, real_metrics = client.train()

            if client.is_malicious and cid in attackers:
                est_trust = trust_manager.get_trust_score(cid)
                if attackers[cid].should_attack(round_num, est_trust):
                    benign_upds = [
                        updates[j] for j, prev_cid in enumerate(client_ids[:-1])
                        if prev_cid not in malicious_ids
                    ]
                    updates.append(attackers[cid].poison_gradient(
                        raw_update,
                        benign_updates=benign_upds if benign_upds else None
                    ))
                    attacks_this_round += 1
                else:
                    updates.append(raw_update)
            else:
                updates.append(raw_update)

            metrics_list.append(real_metrics)  # FIX 6: metrics thực

        history['attack_rate_actual'].append(attacks_this_round / CLIENTS_PER_ROUND)

        # Reference gradient = avg của benign clients được chọn round này
        benign_sel = [cid for cid in client_ids if cid not in malicious_ids]
        if benign_sel:
            benign_ups = [updates[i] for i, cid in enumerate(client_ids)
                          if cid in benign_sel]
            ref = {k: torch.mean(torch.stack([u[k] for u in benign_ups]), dim=0)
                   for k in benign_ups[0]}
        else:
            # Tất cả clients được chọn đều là malicious → dùng global model gradient
            # (fallback an toàn hơn bản gốc)
            ref = {k: torch.mean(torch.stack([u[k] for u in updates]), dim=0)
                   for k in updates[0]}

        # Trust update với metrics thực
        for i, cid in enumerate(client_ids):
            trust_manager.update_trust(
                cid, updates[i], ref, metrics_list[i], round_num
            )

        trust_manager.apply_idle_decay(client_ids, round_num)

        # Ghi trust stats
        b_trust = [trust_manager.get_trust_score(c)
                   for c in range(NUM_CLIENTS) if c not in malicious_ids]
        m_trust = [trust_manager.get_trust_score(c) for c in malicious_ids]
        history['trust_benign'].append(float(np.mean(b_trust)))
        history['trust_malicious'].append(float(np.mean(m_trust)))

        # FIX 7: Aggregation trả về None → skip update round này
        agg = aggregator.aggregate(updates, client_ids, metrics_list)
        if agg is not None:
            for name, param in model.named_parameters():
                param.data += agg[name]
        # else: giữ nguyên model (không apply poisoned update)

        # Track filtered clients
        agg_stats = aggregator.get_stats()
        history['clients_filtered'].append(
            agg_stats['avg_filtered'] if agg_stats['total_rounds'] > 0 else 0
        )

        # Evaluation
        model.eval()
        correct, total, test_loss = 0, 0, 0.0
        with torch.no_grad():
            for data, target in test_loader:
                out = model(data)
                test_loss += torch.nn.functional.cross_entropy(out, target).item()
                correct   += out.argmax(1).eq(target).sum().item()
                total     += target.size(0)

        history['accuracy'].append(correct / total)
        history['loss'].append(test_loss / len(test_loader))

    return history


# ══════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════

def compute_asr(clean_acc: float, poisoned_acc: float, baseline: float = 0.1) -> float:
    if clean_acc <= baseline:
        return 0.0
    return float(np.clip((clean_acc - poisoned_acc) / (clean_acc - baseline), 0, 1))


# ══════════════════════════════════════════════════════════════════════
# PLOT 1 — ACCURACY
# ══════════════════════════════════════════════════════════════════════

def plot_accuracy(results: dict, clean_acc: float, save_dir: str):
    fig, ax = plt.subplots(figsize=(12, 6))

    for attack in ATTACK_ORDER:
        acc = results[attack]['accuracy']
        ax.plot(
            acc,
            label=f"{ATTACK_DISPLAY[attack]}  (final {acc[-1]*100:.1f}%)",
            color=PALETTE[attack],
            linestyle=LINESTYLES[attack],
            linewidth=LINEWIDTHS[attack],
        )

    ax.axhline(y=TAU, color='gray', linestyle=':', linewidth=1.0,
               alpha=0.5, label=f'τ={TAU}')
    ax.set_xlabel('Round', fontsize=13)
    ax.set_ylabel('Test Accuracy', fontsize=13)
    ax.set_ylim([0, 1.08])
    ax.set_xlim([0, NUM_ROUNDS - 1])
    ax.grid(True, alpha=0.2)
    ax.set_title(
        f"TrustAware Defense (FIXED) — Accuracy per Attack Type\n"
        f"α={ALPHA}, τ={TAU}, init={INITIAL_TRUST}, decay={IDLE_DECAY}, "
        f"cpr={CLIENTS_PER_ROUND}, attack_rate={ATTACK_RATE*100:.0f}%, "
        f"scale={POISONING_SCALE}",
        fontsize=11, fontweight='bold', pad=10,
    )
    ax.legend(fontsize=10, loc='lower right', framealpha=0.92)
    fig.tight_layout()
    path = os.path.join(save_dir, 'exp_newmethod_accuracy.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")


# ══════════════════════════════════════════════════════════════════════
# PLOT 2 — LOSS
# ══════════════════════════════════════════════════════════════════════

def plot_loss(results: dict, save_dir: str):
    fig, ax = plt.subplots(figsize=(12, 6))

    for attack in ATTACK_ORDER:
        loss = results[attack]['loss']
        ax.plot(loss,
                label=ATTACK_DISPLAY[attack],
                color=PALETTE[attack],
                linestyle=LINESTYLES[attack],
                linewidth=LINEWIDTHS[attack])

    ax.set_xlabel('Round', fontsize=13)
    ax.set_ylabel('Test Loss', fontsize=13)
    ax.set_xlim([0, NUM_ROUNDS - 1])
    ax.grid(True, alpha=0.2)
    ax.set_title(
        f"TrustAware Defense (FIXED) — Loss per Attack Type\n"
        f"α={ALPHA}, τ={TAU}, init={INITIAL_TRUST}, cpr={CLIENTS_PER_ROUND}, "
        f"attack_rate={ATTACK_RATE*100:.0f}%, scale={POISONING_SCALE}",
        fontsize=11, fontweight='bold', pad=10,
    )
    ax.legend(fontsize=10, loc='upper right', framealpha=0.92)
    fig.tight_layout()
    path = os.path.join(save_dir, 'exp_newmethod_loss.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")


# ══════════════════════════════════════════════════════════════════════
# PLOT 3 — ASR
# ══════════════════════════════════════════════════════════════════════

def plot_asr(results: dict, clean_acc: float, save_dir: str):
    attacks  = [a for a in ATTACK_ORDER if a != 'no_attack']
    asrs     = [compute_asr(clean_acc, results[a]['accuracy'][-1]) * 100 for a in attacks]
    fin_accs = [results[a]['accuracy'][-1] * 100 for a in attacks]
    colors   = [PALETTE[a] for a in attacks]
    labels   = [ATTACK_DISPLAY[a] for a in attacks]

    x     = np.arange(len(attacks))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width/2, fin_accs, width,
                   label='Final Accuracy (%)', color=colors,
                   alpha=0.85, edgecolor='#333', linewidth=0.8)
    bars2 = ax.bar(x + width/2, asrs, width,
                   label='ASR (%)', color=colors,
                   alpha=0.45, edgecolor='#333', linewidth=0.8, hatch='//')

    for bar, val in zip(bars1, fin_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, asrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#c0392b')

    ax.axhline(y=clean_acc * 100, color='#2ecc71', linestyle='--',
               linewidth=1.5, alpha=0.7,
               label=f'No Attack baseline ({clean_acc*100:.1f}%)')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_ylim([0, 115])
    ax.grid(True, alpha=0.25, axis='y')
    ax.set_title(
        f"TrustAware (FIXED) — ASR & Final Accuracy per Attacker\n"
        f"(Solid = Final Acc,  Hatched = ASR,  lower ASR = better)\n"
        f"α={ALPHA}, τ={TAU}, init={INITIAL_TRUST}, cpr={CLIENTS_PER_ROUND}, "
        f"scale={POISONING_SCALE}",
        fontsize=11, fontweight='bold', pad=10,
    )
    ax.legend(fontsize=10, loc='upper right')
    fig.tight_layout()
    path = os.path.join(save_dir, 'exp_newmethod_asr.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")


# ══════════════════════════════════════════════════════════════════════
# PLOT 4 — HEATMAP: final trust + separation
# ══════════════════════════════════════════════════════════════════════

def plot_heatmap(results: dict, save_dir: str):
    attacks         = ATTACK_ORDER
    final_benign    = [results[a]['trust_benign'][-1]    for a in attacks]
    final_malicious = [results[a]['trust_malicious'][-1] for a in attacks]
    sep             = [b - m for b, m in zip(final_benign, final_malicious)]
    data            = np.array([final_benign, final_malicious])

    fig, axes = plt.subplots(1, 2, figsize=(14, 4),
                             gridspec_kw={'width_ratios': [3, 1]})

    sns.heatmap(
        data, ax=axes[0],
        xticklabels=[ATTACK_DISPLAY[a] for a in attacks],
        yticklabels=['Benign', 'Malicious'],
        annot=True, fmt='.3f', cmap='RdYlGn',
        vmin=0, vmax=1,
        cbar_kws={'label': 'Final Trust Score'},
        linewidths=1.0, annot_kws={'size': 13, 'weight': 'bold'},
    )
    axes[0].set_title(
        f"Final Trust Score: Benign vs Malicious (FIXED)\n"
        f"α={ALPHA}, τ={TAU}, init={INITIAL_TRUST}, decay={IDLE_DECAY}",
        fontsize=11, fontweight='bold',
    )
    axes[0].axhline(y=1, color='white', linewidth=2)

    colors_sep = ['#27ae60' if s > 0.3 else '#e67e22' if s > 0.1 else '#e74c3c'
                  for s in sep]
    bars = axes[1].barh(
        [ATTACK_DISPLAY[a] for a in attacks], sep,
        color=colors_sep, edgecolor='#333', linewidth=0.7, alpha=0.85
    )
    for bar, val in zip(bars, sep):
        axes[1].text(val + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
    axes[1].axvline(x=0.3, color='gray', linestyle='--',
                    linewidth=1.0, alpha=0.6, label='good sep=0.3')
    axes[1].set_xlabel('Trust Separation\n(benign − malicious)', fontsize=10)
    axes[1].set_title('Separation', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=8)
    axes[1].set_xlim([0, 1.0])
    axes[1].grid(True, alpha=0.2, axis='x')

    fig.tight_layout()
    path = os.path.join(save_dir, 'exp_newmethod_heatmap.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")


# ══════════════════════════════════════════════════════════════════════
# PLOT 5 — TRUST EVOLUTION
# ══════════════════════════════════════════════════════════════════════

def plot_trust_evolution(results: dict, save_dir: str):
    attacks_to_plot = [a for a in ATTACK_ORDER if a != 'no_attack']
    ncols = 3
    nrows = (len(attacks_to_plot) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows),
                              sharex=True, sharey=True)
    axes_flat = axes.flatten() if nrows > 1 else list(axes)

    rounds = np.arange(NUM_ROUNDS)

    for i, attack in enumerate(attacks_to_plot):
        ax = axes_flat[i]
        tb = results[attack]['trust_benign']
        tm = results[attack]['trust_malicious']

        ax.plot(rounds, tb, color='#2980b9', linewidth=2.0, label='Benign avg')
        ax.plot(rounds, tm, color='#e74c3c', linewidth=2.0,
                linestyle='--', label='Malicious avg')
        ax.fill_between(rounds, tm, tb, alpha=0.12, color='#27ae60',
                        label='Separation gap')
        ax.axhline(y=TAU, color='gray', linestyle=':', linewidth=1.2,
                   alpha=0.7, label=f'τ={TAU}')

        ax.set_title(ATTACK_DISPLAY[attack], fontsize=12, fontweight='bold',
                     color=PALETTE[attack])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.2)
        if i % ncols == 0:
            ax.set_ylabel('Trust Score', fontsize=10)
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel('Round', fontsize=10)
        if i == 0:
            ax.legend(fontsize=8, loc='center right')

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Trust Evolution — Benign vs Malicious (FIXED: α={ALPHA}, τ={TAU}, "
        f"init={INITIAL_TRUST})\n"
        f"attack_rate={ATTACK_RATE*100:.0f}%, scale={POISONING_SCALE}, "
        f"cpr={CLIENTS_PER_ROUND}",
        fontsize=12, fontweight='bold', y=1.01
    )
    fig.tight_layout()
    path = os.path.join(save_dir, 'exp_newmethod_trust.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")


# ══════════════════════════════════════════════════════════════════════
# CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════

def print_summary(results: dict, clean_acc: float):
    print("\n" + "=" * 75)
    print(f"  TRUST-AWARE (FIXED) — SUMMARY")
    print(f"  α={ALPHA}, τ={TAU}, init={INITIAL_TRUST}, decay={IDLE_DECAY}, "
          f"cpr={CLIENTS_PER_ROUND}")
    print(f"  Clean baseline accuracy: {clean_acc*100:.2f}%")
    print("=" * 75)
    hdr = (f"  {'Attack':<16} {'Final Acc':>11} {'ASR':>9} "
           f"{'TrustSep':>10} {'Benign T':>10} {'Malicious T':>12}")
    print(hdr)
    print("  " + "-" * 73)
    for attack in ATTACK_ORDER:
        acc = results[attack]['accuracy'][-1]
        asr = compute_asr(clean_acc, acc) * 100
        tb  = results[attack]['trust_benign'][-1]
        tm  = results[attack]['trust_malicious'][-1]
        sep = tb - tm
        print(f"  {ATTACK_DISPLAY[attack]:<16} {acc*100:>10.2f}% "
              f"{asr:>8.2f}% {sep:>10.3f} {tb:>10.3f}  {tm:>10.3f}")
    print("=" * 75)


# ══════════════════════════════════════════════════════════════════════
# EXP INDEX MAP
# ══════════════════════════════════════════════════════════════════════

# Mapping exp1..exp6 → attack name (no_attack được tính là exp1/baseline)
EXP_MAP = {
    'exp1': 'no_attack',
    'exp2': 'static',
    'exp3': 'delayed',
    'exp4': 'adaptive',
    'exp5': 'intermittent',
    'exp6': 'norm_tuned',
}


# ══════════════════════════════════════════════════════════════════════
# SINGLE-SCENARIO PLOTS (dùng khi chạy 1 exp riêng lẻ)
# ══════════════════════════════════════════════════════════════════════

def plot_single_accuracy(attack: str, history: dict, save_dir: str):
    """Accuracy + loss curve cho 1 scenario."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    rounds = np.arange(len(history['accuracy']))
    color  = PALETTE[attack]

    # Accuracy
    axes[0].plot(rounds, history['accuracy'],
                 color=color, linewidth=2.4,
                 label=f"{ATTACK_DISPLAY[attack]}  "
                       f"(final {history['accuracy'][-1]*100:.1f}%)")
    axes[0].axhline(y=TAU, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)
    axes[0].set_xlabel('Round', fontsize=12)
    axes[0].set_ylabel('Test Accuracy', fontsize=12)
    axes[0].set_ylim([0, 1.08])
    axes[0].set_title('Accuracy', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.2)
    axes[0].legend(fontsize=10)

    # Loss
    axes[1].plot(rounds, history['loss'],
                 color=color, linewidth=2.4,
                 label=f"Loss  (final {history['loss'][-1]:.3f})")
    axes[1].set_xlabel('Round', fontsize=12)
    axes[1].set_ylabel('Test Loss', fontsize=12)
    axes[1].set_title('Loss', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.2)
    axes[1].legend(fontsize=10)

    fig.suptitle(
        f"TrustAware (FIXED) vs {ATTACK_DISPLAY[attack]}\n"
        f"α={ALPHA}, τ={TAU}, init={INITIAL_TRUST}, decay={IDLE_DECAY}, "
        f"cpr={CLIENTS_PER_ROUND}, scale={POISONING_SCALE}",
        fontsize=11, fontweight='bold',
    )
    fig.tight_layout()
    path = os.path.join(save_dir, f'exp_newmethod_{attack}_accuracy.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_single_trust(attack: str, history: dict, save_dir: str):
    """Trust evolution cho 1 scenario."""
    fig, ax = plt.subplots(figsize=(10, 5))

    rounds = np.arange(len(history['trust_benign']))
    tb     = history['trust_benign']
    tm     = history['trust_malicious']

    ax.plot(rounds, tb, color='#2980b9', linewidth=2.2, label='Benign avg trust')
    ax.plot(rounds, tm, color='#e74c3c', linewidth=2.2,
            linestyle='--', label='Malicious avg trust')
    ax.fill_between(rounds, tm, tb, alpha=0.12, color='#27ae60', label='Separation gap')
    ax.axhline(y=TAU, color='gray', linestyle=':', linewidth=1.2,
               alpha=0.7, label=f'τ={TAU} (filter threshold)')

    sep_final = tb[-1] - tm[-1]
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Trust Score', fontsize=12)
    ax.set_ylim([0, 1.05])
    ax.set_title(
        f"Trust Evolution — TrustAware (FIXED) vs {ATTACK_DISPLAY[attack]}\n"
        f"Final separation: {sep_final:.3f}  "
        f"(Benign={tb[-1]:.3f}, Malicious={tm[-1]:.3f})\n"
        f"α={ALPHA}, τ={TAU}, init={INITIAL_TRUST}, decay={IDLE_DECAY}",
        fontsize=11, fontweight='bold',
    )
    ax.legend(fontsize=10, loc='center right')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    path = os.path.join(save_dir, f'exp_newmethod_{attack}_trust.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")


# ══════════════════════════════════════════════════════════════════════
# SHARED SETUP (dùng chung cho single và all)
# ══════════════════════════════════════════════════════════════════════

def _setup(save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    malicious_ids = set(
        np.random.choice(NUM_CLIENTS, int(NUM_CLIENTS * ATTACK_RATE), replace=False)
    )
    print(f"  Malicious ({len(malicious_ids)}): {sorted(malicious_ids)[:10]} ...")
    print("  Loading MNIST ...")
    client_datasets, test_loader = load_data()
    return malicious_ids, client_datasets, test_loader


# ══════════════════════════════════════════════════════════════════════
# RUN SINGLE EXPERIMENT
# ══════════════════════════════════════════════════════════════════════

def run_single(attack: str, save_dir: str = 'results/figures'):
    """Chạy 1 scenario duy nhất, sinh 2 ảnh riêng."""
    print("\n" + "=" * 75)
    print(f"  TrustAware (FIXED) — Single Run: {attack.upper()}")
    print(f"  α={ALPHA} | τ={TAU} | init={INITIAL_TRUST} | "
          f"decay={IDLE_DECAY} | cpr={CLIENTS_PER_ROUND}")
    print("=" * 75 + "\n")

    malicious_ids, client_datasets, test_loader = _setup(save_dir)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    history = run_scenario(
        attack, client_datasets, test_loader,
        malicious_ids, num_rounds=NUM_ROUNDS
    )

    final = history['accuracy'][-1]
    tb    = history['trust_benign'][-1]
    tm    = history['trust_malicious'][-1]
    print(f"\n  → Accuracy={final*100:.2f}%  "
          f"TrustSep={tb-tm:.3f}  (B={tb:.3f}, M={tm:.3f})")

    print("\n  Generating figures ...")
    plot_single_accuracy(attack, history, save_dir)
    plot_single_trust(attack, history, save_dir)
    print(f"\n  → ./results/figures/exp_newmethod_{attack}_*.png")
    print("=" * 75)


# ══════════════════════════════════════════════════════════════════════
# RUN ALL EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════

def run_all(save_dir: str = 'results/figures'):
    """Chạy toàn bộ 6 scenarios, sinh 5 ảnh tổng hợp."""
    print("\n" + "=" * 75)
    print("  TrustAware (FIXED) — Full Run (all 6 scenarios)")
    print(f"  α={ALPHA} | τ={TAU} | init={INITIAL_TRUST} | "
          f"decay={IDLE_DECAY} | cpr={CLIENTS_PER_ROUND}")
    print(f"  Attacker: unchanged (scale={POISONING_SCALE}, "
          f"attack_rate={ATTACK_RATE*100:.0f}%)")
    print("=" * 75 + "\n")

    malicious_ids, client_datasets, test_loader = _setup(save_dir)

    results = {}
    total   = len(ATTACK_ORDER)
    for idx, attack in enumerate(ATTACK_ORDER, 1):
        print(f"\n  [{idx:02d}/{total}]  TrustAware(FIXED) vs {attack.upper()}")
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        results[attack] = run_scenario(
            attack, client_datasets, test_loader,
            malicious_ids, num_rounds=NUM_ROUNDS
        )
        final = results[attack]['accuracy'][-1]
        tb    = results[attack]['trust_benign'][-1]
        tm    = results[attack]['trust_malicious'][-1]
        print(f"         → Accuracy={final*100:.2f}%  "
              f"TrustSep={tb-tm:.3f} (B={tb:.3f} M={tm:.3f})")

    clean_acc = results['no_attack']['accuracy'][-1]

    print("\n  Generating figures ...")
    plot_accuracy(results, clean_acc, save_dir)
    plot_loss(results, save_dir)
    plot_asr(results, clean_acc, save_dir)
    plot_heatmap(results, save_dir)
    plot_trust_evolution(results, save_dir)

    print_summary(results, clean_acc)
    print(f"\n  → ./results/figures/exp_newmethod_*.png")
    print("=" * 75)


# ══════════════════════════════════════════════════════════════════════
# MAIN — ARGPARSE
# ══════════════════════════════════════════════════════════════════════

def main():
    import argparse

    # Tất cả alias hợp lệ
    valid_expN   = list(EXP_MAP.keys())             # exp1..exp6
    valid_names  = list(EXP_MAP.values())           # no_attack, static, ...
    valid_all    = ['all']
    valid_inputs = valid_all + valid_expN + valid_names

    parser = argparse.ArgumentParser(
        prog='exp_newmethod.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
  TrustAware Defense (FIXED) — 1v1 vs Attackers

  USAGE EXAMPLES:
    python experiments/exp_newmethod.py           # chạy tất cả
    python experiments/exp_newmethod.py all       # chạy tất cả
    python experiments/exp_newmethod.py exp1      # no_attack  (baseline)
    python experiments/exp_newmethod.py exp2      # static
    python experiments/exp_newmethod.py exp3      # delayed
    python experiments/exp_newmethod.py exp4      # adaptive
    python experiments/exp_newmethod.py exp5      # intermittent
    python experiments/exp_newmethod.py exp6      # norm_tuned
    python experiments/exp_newmethod.py static    # alias trực tiếp
        """,
    )
    parser.add_argument(
        'scenario',
        nargs='?',
        default='all',
        choices=valid_inputs,
        metavar='SCENARIO',
        help=(
            f"Scenario muốn chạy. Chọn: all | "
            f"exp1..exp6 | "
            f"{' | '.join(valid_names)}"
        ),
    )

    args = parser.parse_args()
    scenario = args.scenario.lower().strip()

    save_dir = 'results/figures'

    # Resolve expN → attack name
    if scenario in EXP_MAP:
        attack = EXP_MAP[scenario]
        print(f"  {scenario} → {attack}")
        run_single(attack, save_dir)

    elif scenario in valid_names:
        run_single(scenario, save_dir)

    else:  # 'all' hoặc default
        run_all(save_dir)


if __name__ == "__main__":
    main()