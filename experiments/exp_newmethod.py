"""
Experiment: Trust-Aware Defense vs Attackers — v4 FULL DEFENSE
===============================================================
Thay đổi so với bản cũ (v2 FIXED):

  DEFENSE PARAMS:
    alpha:         0.75 → 0.5    (trust đổi 2x nhanh hơn)
    tau:           0.5  → 0.45   (margin rộng hơn cho benign)
    initial_trust: 0.5  → 0.3    (không include client mới ngay)
    idle_decay:    0.008→ 0.01   (decay nhanh hơn khi vắng mặt)
    warmup_rounds: 0    → 3      (không filter 3 rounds đầu)

  AGGREGATION (NEW — 2 lớp bảo vệ):
    Lớp 1 — Norm-clipping: clip gradient về 2.5x median_norm
    Lớp 2 — Coordinate-wise median: thay weighted mean
    → Robust với up to ⌊n/2⌋−1 Byzantine clients sau filter

  CLIENTS_PER_ROUND: 20 (giữ nguyên)
  ATTACKER: KHÔNG thay đổi (scale=5.0, layer_boost tối đa 2.5x)

Tại sao kết hợp norm-clip + median?
  - Norm-clip: giảm variance của median → hội tụ nhanh hơn
  - Median: attacker vượt filter vẫn không dominate được coordinate nào
  - Kết hợp: defense-in-depth, 4 lớp độc lập
──────────────────────────────────────────────────────────────────────
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
# CONFIG — v3 STRONG DEFENSE
# ══════════════════════════════════════════════════════════════════════

ATTACK_RATE        = 0.20
POISONING_SCALE    = 5.0
NUM_CLIENTS        = 100
CLIENTS_PER_ROUND  = 20
NUM_ROUNDS         = 80
SEED               = 42

# v4 defense params — stronger so defense wins vs attacks
ALPHA              = 0.45   # faster trust reaction (was 0.5)
TAU                = 0.45   # filter threshold
INITIAL_TRUST      = 0.3    # new clients start below tau
IDLE_DECAY         = 0.01
WARMUP_ROUNDS      = 1       # was 3 — only 1 round no-filter to limit poison
WINDOW_SIZE        = 5
SIM_WEIGHT         = 0.9

# Norm clipping
ENABLE_NORM_CLIP   = True
CLIP_MULTIPLIER    = 2.0     # stricter: 2.0 * median_norm (was 2.5)

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
# ATTACKER — KHÔNG THAY ĐỔI (giữ nguyên để đánh giá khách quan)
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

    def _flip_poison(self, gradient: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
# SCENARIO RUNNER — v3 STRONG DEFENSE
# ══════════════════════════════════════════════════════════════════════

def run_scenario(
    attack_type: str,
    client_datasets: list,
    test_loader,
    malicious_ids: set,
    num_rounds: int = NUM_ROUNDS,
) -> Dict:
    """
    Chạy 1 kịch bản với TrustAware v3 STRONG DEFENSE.

    Key changes vs v2:
      - TrustScoreManager: alpha=0.5, tau=0.45, initial_trust=0.3
      - TrustAwareAggregator: enable_norm_clip=True, clip_multiplier=2.5
      - aggregator.aggregate() nhận round_num để support warmup
    """
    model = get_model()
    trust_manager = TrustScoreManager(
        num_clients       = NUM_CLIENTS,
        alpha             = ALPHA,
        tau               = TAU,
        initial_trust     = INITIAL_TRUST,
        enable_decay      = True,
        decay_strategy    = "exponential",
        window_size       = WINDOW_SIZE,
        similarity_weight = SIM_WEIGHT,
        idle_decay_rate   = IDLE_DECAY,
        enable_norm_penalty     = True,
        norm_penalty_threshold  = 2.0,   # penalize norm > 2x benign (was 3.0)
        norm_penalty_strength   = 0.6,
        warmup_rounds           = WARMUP_ROUNDS,
    )
    aggregator = TrustAwareAggregator(
        trust_manager,
        enable_filtering      = True,
        enable_norm_clip      = ENABLE_NORM_CLIP,
        clip_multiplier       = CLIP_MULTIPLIER,
        use_median            = True,     # FIX 4: coordinate-wise median
        min_trusted_for_median = 2,       # fallback mean nếu chỉ có 1 trusted
    )

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
        'clients_clipped':    [],
    }

    for round_num in tqdm(
        range(num_rounds),
        desc=f"v3 TrustAware vs {attack_type:<12}",
        leave=False, ncols=90
    ):
        selected_idx = np.random.choice(NUM_CLIENTS, CLIENTS_PER_ROUND, replace=False)
        selected     = [clients[i] for i in selected_idx]

        global_params = {n: p.data.clone() for n, p in model.named_parameters()}
        for c in selected:
            c.set_parameters(global_params)

        updates, client_ids, metrics_list = [], [], []
        attacks_this_round = 0

        for client in selected:
            cid = client.client_id
            client_ids.append(cid)
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

            metrics_list.append(real_metrics)

        history['attack_rate_actual'].append(attacks_this_round / CLIENTS_PER_ROUND)

        # Reference gradient = avg của benign clients được chọn round này
        benign_sel = [cid for cid in client_ids if cid not in malicious_ids]
        if benign_sel:
            benign_ups = [updates[i] for i, cid in enumerate(client_ids)
                          if cid in benign_sel]
            ref = {k: torch.mean(torch.stack([u[k] for u in benign_ups]), dim=0)
                   for k in benign_ups[0]}
        else:
            ref = {k: torch.mean(torch.stack([u[k] for u in updates]), dim=0)
                   for k in updates[0]}

        # Trust update
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

        # Aggregate — pass round_num để support warmup
        agg = aggregator.aggregate(updates, client_ids, metrics_list, round_num=round_num)
        if agg is not None:
            for name, param in model.named_parameters():
                param.data += agg[name]

        # Track stats
        agg_stats = aggregator.get_stats()
        history['clients_filtered'].append(
            agg_stats['avg_filtered'] if agg_stats['total_rounds'] > 0 else 0
        )
        history['clients_clipped'].append(
            agg_stats['avg_clipped'] if agg_stats['total_rounds'] > 0 else 0
        )

        if round_num == 0 and agg_stats['total_rounds'] > 0:
            method = "median" if agg_stats['median_rounds'] > 0 else "weighted_mean"
            print(f"  [Round 0] Aggregation method: {method}")

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
# PLOTS — giống bản cũ, thêm subtitle v3
# ══════════════════════════════════════════════════════════════════════

def _config_str():
    return (f"α={ALPHA}, τ={TAU}, init={INITIAL_TRUST}, decay={IDLE_DECAY}, "
            f"clip={CLIP_MULTIPLIER}x+median, warmup={WARMUP_ROUNDS}r")


def plot_accuracy(results: dict, clean_acc: float, save_dir: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    for attack in ATTACK_ORDER:
        acc = results[attack]['accuracy']
        ax.plot(acc,
                label=f"{ATTACK_DISPLAY[attack]}  (final {acc[-1]*100:.1f}%)",
                color=PALETTE[attack],
                linestyle=LINESTYLES[attack],
                linewidth=LINEWIDTHS[attack])

    ax.axhline(y=TAU, color='gray', linestyle=':', linewidth=1.0,
               alpha=0.5, label=f'τ={TAU} (trust threshold)')
    ax.set_xlabel('Round', fontsize=13)
    ax.set_ylabel('Test Accuracy', fontsize=13)
    ax.set_ylim([0, 1.08])
    ax.set_xlim([0, NUM_ROUNDS - 1])
    ax.grid(True, alpha=0.2)
    ax.set_title(
        f"TrustAware Defense (v3 STRONG) — Accuracy per Attack Type\n"
        f"({_config_str()}, attack_rate={ATTACK_RATE*100:.0f}%, scale={POISONING_SCALE}, "
        f"{NUM_ROUNDS} rounds, {NUM_CLIENTS} clients)",
        fontsize=10, fontweight='bold', pad=10,
    )
    ax.legend(fontsize=10, loc='lower right', framealpha=0.92)
    fig.tight_layout()
    path = os.path.join(save_dir, 'exp_newmethod_accuracy.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")


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
        f"TrustAware (v3 STRONG) — ASR & Final Accuracy per Attacker\n"
        f"(Solid = Final Acc,  Hatched = ASR,  lower ASR = better defence)\n"
        f"α={ALPHA},  τ={TAU},  attack_rate={ATTACK_RATE*100:.0f}%,  scale={POISONING_SCALE}",
        fontsize=11, fontweight='bold', pad=10,
    )
    ax.legend(fontsize=10, loc='upper right')
    fig.tight_layout()
    path = os.path.join(save_dir, 'exp_newmethod_asr.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")


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
        f"Final Trust Score — Benign vs Malicious (TrustAware v3 STRONG)\n"
        f"α={ALPHA},  τ={TAU},  init={INITIAL_TRUST},  decay={IDLE_DECAY},  "
        f"norm_clip={CLIP_MULTIPLIER}x",
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
                    linewidth=1.0, alpha=0.6, label='sep=0.3 (good)')
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
        f"Trust Evolution (v3 STRONG) — Benign vs Malicious\n"
        f"α={ALPHA},  τ={TAU},  init={INITIAL_TRUST},  norm_clip={CLIP_MULTIPLIER}x,  "
        f"warmup={WARMUP_ROUNDS}r\n"
        f"attack_rate={ATTACK_RATE*100:.0f}%,  scale={POISONING_SCALE},  cpr={CLIENTS_PER_ROUND}",
        fontsize=11, fontweight='bold', y=1.01
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
    print("\n" + "=" * 80)
    print(f"  TRUST-AWARE v3 STRONG DEFENSE — SUMMARY")
    print(f"  {_config_str()}")
    print(f"  Clean baseline: {clean_acc*100:.2f}%")
    print("=" * 80)
    hdr = (f"  {'Attack':<16} {'Final Acc':>11} {'ASR':>9} "
           f"{'TrustSep':>10} {'Benign T':>10} {'Malicious T':>12}")
    print(hdr)
    print("  " + "-" * 78)
    for attack in ATTACK_ORDER:
        acc = results[attack]['accuracy'][-1]
        asr = compute_asr(clean_acc, acc) * 100
        tb  = results[attack]['trust_benign'][-1]
        tm  = results[attack]['trust_malicious'][-1]
        sep = tb - tm
        print(f"  {ATTACK_DISPLAY[attack]:<16} {acc*100:>10.2f}% "
              f"{asr:>8.2f}% {sep:>10.3f} {tb:>10.3f}  {tm:>10.3f}")
    print("=" * 80)


# ══════════════════════════════════════════════════════════════════════
# EXP INDEX MAP
# ══════════════════════════════════════════════════════════════════════

EXP_MAP = {
    'exp1': 'no_attack',
    'exp2': 'static',
    'exp3': 'delayed',
    'exp4': 'adaptive',
    'exp5': 'intermittent',
    'exp6': 'norm_tuned',
}


# ══════════════════════════════════════════════════════════════════════
# SHARED SETUP
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
# RUN SINGLE
# ══════════════════════════════════════════════════════════════════════

def run_single_exp(attack: str, save_dir: str = 'results/figures'):
    print("\n" + "=" * 80)
    print(f"  TrustAware v3 STRONG — Single Run: {attack.upper()}")
    print(f"  {_config_str()}")
    print("=" * 80 + "\n")

    malicious_ids, client_datasets, test_loader = _setup(save_dir)

    torch.manual_seed(SEED); np.random.seed(SEED)
    history = run_scenario(attack, client_datasets, test_loader,
                           malicious_ids, num_rounds=NUM_ROUNDS)

    final = history['accuracy'][-1]
    tb    = history['trust_benign'][-1]
    tm    = history['trust_malicious'][-1]
    clean = results_single_acc = final  # single run, no clean reference
    print(f"\n  → Accuracy={final*100:.2f}%  TrustSep={tb-tm:.3f}  (B={tb:.3f}, M={tm:.3f})")
    print("=" * 80)


# ══════════════════════════════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════════════════════════════

def run_all(save_dir: str = 'results/figures'):
    print("\n" + "=" * 80)
    print("  TrustAware v3 STRONG DEFENSE — Full Run (all 6 scenarios)")
    print(f"  {_config_str()}")
    print(f"  Attacker: unchanged (scale={POISONING_SCALE}, "
          f"attack_rate={ATTACK_RATE*100:.0f}%)")
    print("=" * 80 + "\n")

    malicious_ids, client_datasets, test_loader = _setup(save_dir)

    results = {}
    total   = len(ATTACK_ORDER)
    for idx, attack in enumerate(ATTACK_ORDER, 1):
        print(f"\n  [{idx:02d}/{total}]  TrustAware v3 vs {attack.upper()}")
        torch.manual_seed(SEED); np.random.seed(SEED)
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
    plot_asr(results, clean_acc, save_dir)
    plot_heatmap(results, save_dir)
    plot_trust_evolution(results, save_dir)

    print_summary(results, clean_acc)
    print(f"\n  → ./results/figures/exp_newmethod_*.png")
    print("=" * 80)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    import argparse

    valid_expN  = list(EXP_MAP.keys())
    valid_names = list(EXP_MAP.values())
    valid_inputs = ['all'] + valid_expN + valid_names

    parser = argparse.ArgumentParser(
        prog='exp_newmethod.py',
        description='TrustAware Defense v3 STRONG — 1v1 vs Attackers',
    )
    parser.add_argument(
        'scenario', nargs='?', default='all',
        choices=valid_inputs, metavar='SCENARIO',
        help='Scenario: all | exp1..exp6 | no_attack|static|delayed|adaptive|intermittent|norm_tuned',
    )

    args = parser.parse_args()
    scenario = args.scenario.lower().strip()
    save_dir = 'results/figures'

    if scenario in EXP_MAP:
        attack = EXP_MAP[scenario]
        run_single_exp(attack, save_dir)
    elif scenario in valid_names:
        run_single_exp(scenario, save_dir)
    else:
        run_all(save_dir)


if __name__ == "__main__":
    main()