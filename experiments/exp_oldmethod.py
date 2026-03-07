"""
Experiment: Old Methods Comparison
Compare FedAvg vs Krum vs Trimmed-Mean under STRONG gradient poisoning attacks.

Outputs (one chart per file):
    - results/figures/exp_oldmethod_accuracy.png   — 12 accuracy curves
    - results/figures/exp_oldmethod_heatmap.png    — final accuracy heatmap
    - results/figures/exp_oldmethod_asr.png        — ASR bar chart (12 bars)
    - results/figures/exp_oldmethod_loss.png       — 12 loss curves
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from tqdm import tqdm
from typing import Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from models.cnn_mnist import get_model
from fl_core.client import FLClient
from fl_core.aggregation.trust_aware import (
    FedAvgAggregator, KrumAggregator, TrimmedMeanAggregator
)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════

ATTACK_RATE       = 0.20
POISONING_SCALE   = 5.0
NUM_CLIENTS       = 100
CLIENTS_PER_ROUND = 20
NUM_ROUNDS        = 80
SEED              = 42

DEFENSE_ORDER = ["fedavg", "krum", "trimmed_mean"]
ATTACK_ORDER  = ["no_attack", "static", "delayed", "adaptive"]

# ══════════════════════════════════════════════════════════════════════
# COLOR SYSTEM — 12 unique colors (3 families × 4 shades)
# Family:  FedAvg=reds,  Krum=blues,  TrimmedMean=greens
# Shade:   no_attack=darkest → adaptive=lightest
# ══════════════════════════════════════════════════════════════════════

PALETTE = {
    ("fedavg",       "no_attack"): "#c0392b",
    ("fedavg",       "static"):    "#e74c3c",
    ("fedavg",       "delayed"):   "#f1948a",
    ("fedavg",       "adaptive"):  "#f5b7b1",

    ("krum",         "no_attack"): "#1a5276",
    ("krum",         "static"):    "#2980b9",
    ("krum",         "delayed"):   "#7fb3d3",
    ("krum",         "adaptive"):  "#aed6f1",

    ("trimmed_mean", "no_attack"): "#1e8449",
    ("trimmed_mean", "static"):    "#27ae60",
    ("trimmed_mean", "delayed"):   "#82e0aa",
    ("trimmed_mean", "adaptive"):  "#a9dfbf",
}

LINESTYLES = {
    "no_attack": (0, ()),
            # solid   ──────
    "static":    (0, (6, 2)),
            # dashed  ── ── ──
    "delayed":   (0, (2, 2)),
            # dotted  ·· ·· ··
    "adaptive":  (0, (6, 2, 2, 2)),
            # dashdot -·-·-·-
}

LINEWIDTHS = {
    "no_attack": 2.8,
    "static":    2.2,
    "delayed":   2.0,
    "adaptive":  2.0,
}

LABEL_MAP = {
    ("fedavg",       "no_attack"): "FedAvg — No Attack",
    ("fedavg",       "static"):    "FedAvg — Static",
    ("fedavg",       "delayed"):   "FedAvg — Delayed",
    ("fedavg",       "adaptive"):  "FedAvg — Adaptive",
    ("krum",         "no_attack"): "Krum — No Attack",
    ("krum",         "static"):    "Krum — Static",
    ("krum",         "delayed"):   "Krum — Delayed",
    ("krum",         "adaptive"):  "Krum — Adaptive",
    ("trimmed_mean", "no_attack"): "TrimmedMean — No Attack",
    ("trimmed_mean", "static"):    "TrimmedMean — Static",
    ("trimmed_mean", "delayed"):   "TrimmedMean — Delayed",
    ("trimmed_mean", "adaptive"):  "TrimmedMean — Adaptive",
}

ATTACK_DISPLAY  = {"no_attack": "No Attack", "static": "Static",
                   "delayed": "Delayed",     "adaptive": "Adaptive"}
DEFENSE_DISPLAY = {"fedavg": "FedAvg", "krum": "Krum",
                   "trimmed_mean": "Trimmed\nMean"}


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
# ATTACKER
# ══════════════════════════════════════════════════════════════════════

class StrongAttacker:
    def __init__(self, client_id, attack_type="static",
                 poisoning_scale=POISONING_SCALE,
                 delay_rounds=10, trust_threshold=0.55,
                 dormant_threshold=0.25, recovery_rounds=5):
        self.client_id        = client_id
        self.attack_type      = attack_type
        self.poisoning_scale  = poisoning_scale
        self.delay_rounds     = delay_rounds
        self.trust_threshold  = trust_threshold
        self.dormant_threshold = dormant_threshold
        self.recovery_rounds  = recovery_rounds
        self.dormant          = False
        self.dormant_since    = -1
        self.total_attacks    = 0
        self.total_rounds     = 0

    def should_attack(self, round_num: int, estimated_trust: float = 1.0) -> bool:
        self.total_rounds += 1
        if self.attack_type == "no_attack":
            return False
        elif self.attack_type == "static":
            decision = True
        elif self.attack_type == "delayed":
            decision = round_num >= self.delay_rounds
        elif self.attack_type == "intermittent":
            decision = np.random.random() < 0.5
        elif self.attack_type == "adaptive":
            decision = self._adaptive(round_num, estimated_trust)
        else:
            decision = True
        if decision:
            self.total_attacks += 1
        return decision

    def _adaptive(self, round_num, trust):
        if trust < self.dormant_threshold:
            if not self.dormant:
                self.dormant = True
                self.dormant_since = round_num
            return False
        if self.dormant:
            if round_num - self.dormant_since < self.recovery_rounds:
                return False
            self.dormant = False
        return trust > self.trust_threshold

    def poison_gradient(self, clean_gradient: Dict, all_updates=None) -> Dict:
        param_names = list(clean_gradient.keys())
        n = len(param_names)
        if all_updates and len(all_updates) > 1:
            all_norms = [float(np.sqrt(sum(torch.norm(p).item()**2
                         for p in u.values()))) for u in all_updates]
            benign_norm = float(np.mean(all_norms))
        else:
            benign_norm = None
        poisoned = {}
        for i, name in enumerate(param_names):
            layer_boost = 1.0 + 1.5 * (i / max(1, n - 1))
            poisoned[name] = -self.poisoning_scale * layer_boost * clean_gradient[name]
        if benign_norm is not None:
            current_norm = float(np.sqrt(sum(torch.norm(p).item()**2
                                          for p in poisoned.values())))
            if current_norm > 1e-6:
                scale = 3.0 * benign_norm / current_norm
                poisoned = {k: scale * v for k, v in poisoned.items()}
        return poisoned


# ══════════════════════════════════════════════════════════════════════
# SCENARIO RUNNER
# ══════════════════════════════════════════════════════════════════════

def run_scenario(defense_name, attack_type, client_datasets,
                 test_loader, malicious_ids, num_rounds=NUM_ROUNDS) -> Dict:
    model = get_model()
    num_malicious_est = int(NUM_CLIENTS * ATTACK_RATE)

    clients = [
        FLClient(i, get_model(), client_datasets[i], is_malicious=(i in malicious_ids))
        for i in range(NUM_CLIENTS)
    ]
    attackers = {cid: StrongAttacker(cid, attack_type=attack_type)
                 for cid in malicious_ids}

    if defense_name == "krum":
        aggregator = KrumAggregator(num_malicious=num_malicious_est)
    elif defense_name == "trimmed_mean":
        aggregator = TrimmedMeanAggregator(trim_ratio=ATTACK_RATE)
    else:
        aggregator = FedAvgAggregator()

    history = {'accuracy': [], 'loss': []}

    for round_num in tqdm(range(num_rounds),
                          desc=f"{defense_name.upper():12s} | {attack_type:12s}",
                          leave=False, ncols=80):
        selected_idx = np.random.choice(NUM_CLIENTS, CLIENTS_PER_ROUND, replace=False)
        selected     = [clients[i] for i in selected_idx]

        global_params = {n: p.data.clone() for n, p in model.named_parameters()}
        for c in selected:
            c.set_parameters(global_params)

        raw_updates = [client.train()[0] for client in selected]

        updates, client_ids, metrics_list = [], [], []
        for idx, client in enumerate(selected):
            cid = client.client_id
            client_ids.append(cid)
            metrics_list.append({'loss': 0.1, 'accuracy': 0.9, 'gradient_norm': 1.0})
            if client.is_malicious and cid in attackers:
                if attackers[cid].should_attack(round_num):
                    benign_upds = [raw_updates[j] for j, c2 in enumerate(selected)
                                   if c2.client_id not in malicious_ids]
                    updates.append(attackers[cid].poison_gradient(
                        raw_updates[idx],
                        all_updates=benign_upds if benign_upds else None,
                    ))
                else:
                    updates.append(raw_updates[idx])
            else:
                updates.append(raw_updates[idx])

        agg = aggregator.aggregate(updates, client_ids, metrics_list)
        for name, param in model.named_parameters():
            param.data += agg[name]

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

def compute_asr(clean_acc, poisoned_acc, baseline=0.1):
    if clean_acc <= baseline:
        return 0.0
    return float(np.clip((clean_acc - poisoned_acc) / (clean_acc - baseline), 0, 1))


# ══════════════════════════════════════════════════════════════════════
# PLOT 1 — ACCURACY CURVES  (1 chart, 12 lines)
# ══════════════════════════════════════════════════════════════════════

def plot_accuracy_curves(results, save_dir):
    fig, ax = plt.subplots(figsize=(13, 7))

    for defense in DEFENSE_ORDER:
        for attack in ATTACK_ORDER:
            acc = results[defense][attack]['accuracy']
            key = (defense, attack)
            ax.plot(acc,
                    label=LABEL_MAP[key],
                    color=PALETTE[key],
                    linestyle=LINESTYLES[attack],
                    linewidth=LINEWIDTHS[attack])
            # Final-value annotation on the right edge
            ax.annotate(
                f"{acc[-1]*100:.1f}%",
                xy=(len(acc) - 1, acc[-1]),
                xytext=(4, 0), textcoords='offset points',
                fontsize=7, color=PALETTE[key], va='center',
            )

    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_ylim([0, 1.10])
    ax.set_xlim([0, NUM_ROUNDS - 1])
    ax.grid(True, alpha=0.25)
    ax.set_title(
        f"Accuracy Curves — FedAvg vs Krum vs TrimmedMean × 4 Attack Types\n"
        f"(attack_rate={ATTACK_RATE*100:.0f}%,  poisoning_scale={POISONING_SCALE},  "
        f"{NUM_ROUNDS} rounds,  {NUM_CLIENTS} clients)",
        fontsize=12, fontweight='bold', pad=12,
    )
    ax.legend(ncol=3, fontsize=8.5, loc='lower right',
              framealpha=0.92, edgecolor='#bbbbbb')

    fig.tight_layout()
    path = os.path.join(save_dir, 'exp_oldmethod_accuracy.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# PLOT 2 — HEATMAP  (1 chart — accuracy only)
# ══════════════════════════════════════════════════════════════════════

def plot_heatmap(results, save_dir):
    data = np.array([
        [results[d][a]['accuracy'][-1] * 100 for a in ATTACK_ORDER]
        for d in DEFENSE_ORDER
    ])

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(
        data, ax=ax,
        xticklabels=[ATTACK_DISPLAY[a]  for a in ATTACK_ORDER],
        yticklabels=[DEFENSE_DISPLAY[d] for d in DEFENSE_ORDER],
        annot=True, fmt='.1f', cmap='RdYlGn',
        vmin=10, vmax=100,
        cbar_kws={'label': 'Final Accuracy (%)'},
        linewidths=0.6,
        annot_kws={'size': 14, 'weight': 'bold'},
    )
    ax.set_title(
        f"Final Test Accuracy (%) — Defense × Attack\n"
        f"(attack_rate={ATTACK_RATE*100:.0f}%,  poisoning_scale={POISONING_SCALE})",
        fontsize=13, fontweight='bold', pad=10,
    )
    ax.set_xlabel('Attack Type', fontsize=12)
    ax.set_ylabel('Defense',     fontsize=12)

    fig.tight_layout()
    path = os.path.join(save_dir, 'exp_oldmethod_heatmap.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# PLOT 3 — ASR BAR CHART  (1 chart, 12 bars)
# ══════════════════════════════════════════════════════════════════════

def plot_asr_bars(results, clean_accs, save_dir):
    fig, ax = plt.subplots(figsize=(11, 6))

    width = 0.22
    x = np.arange(len(ATTACK_ORDER))

    for i, defense in enumerate(DEFENSE_ORDER):
        clean = clean_accs[defense]
        asrs  = [compute_asr(clean, results[defense][a]['accuracy'][-1]) * 100
                 for a in ATTACK_ORDER]
        bar_colors = [PALETTE[(defense, a)] for a in ATTACK_ORDER]
        bars = ax.bar(
            x + (i - 1) * width, asrs, width,
            color=bar_colors,
            edgecolor='#333333', linewidth=0.6, alpha=0.92,
        )
        for bar, val in zip(bars, asrs):
            if val > 1.5:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.8,
                        f'{val:.1f}%',
                        ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([ATTACK_DISPLAY[a] for a in ATTACK_ORDER], fontsize=12)
    ax.set_ylabel('Attack Success Rate (%)', fontsize=12)
    ax.set_ylim([0, 118])
    ax.axhline(y=100, color='#c0392b', linestyle='--',
               linewidth=1.2, alpha=0.55, label='100% = degraded to random guess')
    ax.grid(True, alpha=0.25, axis='y')
    ax.set_title(
        f"Attack Success Rate — FedAvg vs Krum vs TrimmedMean\n"
        f"(Higher = attack more effective  |  "
        f"attack_rate={ATTACK_RATE*100:.0f}%,  scale={POISONING_SCALE},  {NUM_ROUNDS} rounds)",
        fontsize=12, fontweight='bold', pad=10,
    )

    # Legend: 3 defense families + linestyle guide
    legend_handles = [
        mpatches.Patch(color=PALETTE[("fedavg",       "static")], label='FedAvg'),
        mpatches.Patch(color=PALETTE[("krum",         "static")], label='Krum'),
        mpatches.Patch(color=PALETTE[("trimmed_mean", "static")], label='TrimmedMean'),
    ]
    ax.legend(handles=legend_handles, fontsize=11, loc='upper left')

    fig.tight_layout()
    path = os.path.join(save_dir, 'exp_oldmethod_asr.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# PLOT 4 — LOSS CURVES  (1 chart, 12 lines)
# ══════════════════════════════════════════════════════════════════════

def plot_loss_curves(results, save_dir):
    fig, ax = plt.subplots(figsize=(13, 7))

    for defense in DEFENSE_ORDER:
        for attack in ATTACK_ORDER:
            loss = results[defense][attack]['loss']
            key  = (defense, attack)
            ax.plot(loss,
                    label=LABEL_MAP[key],
                    color=PALETTE[key],
                    linestyle=LINESTYLES[attack],
                    linewidth=LINEWIDTHS[attack])

    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Test Loss', fontsize=12)
    ax.set_xlim([0, NUM_ROUNDS - 1])
    ax.grid(True, alpha=0.25)
    ax.set_title(
        f"Loss Curves — FedAvg vs Krum vs TrimmedMean × 4 Attack Types\n"
        f"(attack_rate={ATTACK_RATE*100:.0f}%,  poisoning_scale={POISONING_SCALE},  "
        f"{NUM_ROUNDS} rounds)",
        fontsize=12, fontweight='bold', pad=12,
    )
    ax.legend(ncol=3, fontsize=8.5, loc='upper right',
              framealpha=0.92, edgecolor='#bbbbbb')

    fig.tight_layout()
    path = os.path.join(save_dir, 'exp_oldmethod_loss.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 72)
    print("  OLD METHODS COMPARISON: FedAvg vs Krum vs Trimmed-Mean")
    print(f"  Attack Rate: {ATTACK_RATE*100:.0f}%  |  Poisoning Scale: {POISONING_SCALE}")
    print(f"  Clients: {NUM_CLIENTS}  |  Per Round: {CLIENTS_PER_ROUND}  |  Rounds: {NUM_ROUNDS}")
    print("=" * 72)

    save_dir = 'results/figures'
    os.makedirs(save_dir, exist_ok=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    malicious_ids = set(np.random.choice(
        NUM_CLIENTS, int(NUM_CLIENTS * ATTACK_RATE), replace=False))
    print(f"\n  Malicious IDs (first 10): {sorted(malicious_ids)[:10]} ...")

    print("\n  Loading MNIST data ...")
    client_datasets, test_loader = load_data()

    # ── Run all 12 scenarios ──────────────────────────────────────────
    results    = {d: {} for d in DEFENSE_ORDER}
    clean_accs = {}
    total      = len(DEFENSE_ORDER) * len(ATTACK_ORDER)
    idx        = 0

    for defense in DEFENSE_ORDER:
        for attack in ATTACK_ORDER:
            idx += 1
            print(f"\n  [{idx:02d}/{total}] {defense.upper()} × {attack.upper()}")
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            history = run_scenario(defense, attack, client_datasets,
                                   test_loader, malicious_ids)
            results[defense][attack] = history
            print(f"         → Final Accuracy: {history['accuracy'][-1]*100:.2f}%")
        clean_accs[defense] = results[defense]['no_attack']['accuracy'][-1]

    # ── Plots ─────────────────────────────────────────────────────────
    print("\n  Generating figures ...")
    plot_accuracy_curves(results, save_dir)
    plot_heatmap(results, save_dir)
    plot_asr_bars(results, clean_accs, save_dir)
    plot_loss_curves(results, save_dir)

    # ── Console tables ────────────────────────────────────────────────
    header = f"  {'Defense':<16}" + "".join(f"{ATTACK_DISPLAY[a]:>13}" for a in ATTACK_ORDER)

    print("\n" + "=" * 72)
    print("  FINAL ACCURACY (%)")
    print("=" * 72)
    print(header)
    print("  " + "-" * 68)
    for d in DEFENSE_ORDER:
        row = f"  {d.replace('_',' ').title():<16}"
        for a in ATTACK_ORDER:
            row += f"{results[d][a]['accuracy'][-1]*100:>12.2f}%"
        print(row)

    print("\n" + "=" * 72)
    print("  ATTACK SUCCESS RATE (%)")
    print("=" * 72)
    print(header)
    print("  " + "-" * 68)
    for d in DEFENSE_ORDER:
        row = f"  {d.replace('_',' ').title():<16}"
        for a in ATTACK_ORDER:
            asr = compute_asr(clean_accs[d], results[d][a]['accuracy'][-1]) * 100
            row += f"{asr:>12.2f}%"
        print(row)

    print("\n" + "=" * 72)
    print("  KEY OBSERVATIONS")
    print("=" * 72)
    for attack in [a for a in ATTACK_ORDER if a != 'no_attack']:
        accs  = {d: results[d][attack]['accuracy'][-1] for d in DEFENSE_ORDER}
        best  = max(accs, key=accs.get)
        worst = min(accs, key=accs.get)
        print(f"  {attack:12s}: BEST={best.upper()} ({accs[best]*100:.1f}%)"
              f"  WORST={worst.upper()} ({accs[worst]*100:.1f}%)")

    print(f"\n  Figures → ./results/figures/exp_oldmethod_*.png")
    print("=" * 72)


if __name__ == "__main__":
    main()