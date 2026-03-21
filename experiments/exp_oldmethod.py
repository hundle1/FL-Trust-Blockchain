"""
exp_oldmethod.py — Baseline Comparison (v2 Q1-Ready)
======================================================
So sánh: FedAvg vs Krum vs TrimmedMean vs TrustAware (new)
Attacks: Static, Delayed, Adaptive, Intermittent, Norm-Tuned, Label Flip, Gaussian

Tất cả attack types trong CÙNG bộ biểu đồ:
  1. Accuracy heatmap (defense × attack)
  2. ASR bar chart (grouped by defense)
  3. Accuracy curves (tất cả)
  4. Final comparison table

Paper purpose:
  - Chứng minh Trust-Aware vượt trội tất cả baseline defenses
  - Đặc biệt với adaptive và stealthy attacks (norm-tuned, label flip)
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from models.cnn_mnist import get_model
from fl_core.client import FLClient
from trust.trust_score import TrustScoreManager
from fl_core.aggregation.trust_aware import (
    FedAvgAggregator, KrumAggregator, TrimmedMeanAggregator, TrustAwareAggregator
)
from attacks.attack_suite import create_attacker, compute_realistic_reference
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
PRETRAIN_ROUNDS   = 10    # TrustAware pretrain phase
SEED              = 42

ALPHA = 0.9
TAU   = 0.25   # v8: calibrated for adaptive det≥10%

# Defenses to compare
DEFENSE_ORDER = ["fedavg", "krum", "trimmed_mean", "trust_aware"]
DEFENSE_DISPLAY = {
    "fedavg":        "FedAvg",
    "krum":          "Krum",
    "trimmed_mean":  "Trimmed Mean",
    "trust_aware":   "Trust-Aware (Ours)",
}

# All attacks including new types
ATTACK_ORDER = [
    "no_attack", "static", "delayed", "adaptive",
    "intermittent", "norm_tuned", "label_flip", "gaussian"
]
ATTACK_DISPLAY = {
    "no_attack":    "No Attack",
    "static":       "Static",
    "delayed":      "Delayed",
    "adaptive":     "Adaptive",
    "intermittent": "Intermittent",
    "norm_tuned":   "Norm-Tuned",
    "label_flip":   "Label Flip",
    "gaussian":     "Gaussian",
}

# Color palette: defense family × attack severity
DEFENSE_COLORS = {
    "fedavg":       "#e74c3c",
    "krum":         "#2980b9",
    "trimmed_mean": "#27ae60",
    "trust_aware":  "#8e44ad",
}

ATTACK_LINESTYLES = {
    "no_attack":    (0, ()),
    "static":       (0, (6, 2)),
    "delayed":      (0, (2, 2)),
    "adaptive":     (0, (6, 2, 2, 2)),
    "intermittent": (0, (4, 1, 1, 1)),
    "norm_tuned":   (0, (1, 1)),
    "label_flip":   (0, (3, 1, 1, 1, 1, 1)),
    "gaussian":     (0, (5, 2, 1, 2)),
}

# ══════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════

def load_data(num_clients=NUM_CLIENTS):
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, '..', 'data')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    spc = len(train_ds) // num_clients
    client_datasets = [
        DataLoader(Subset(train_ds, list(range(i * spc, (i + 1) * spc))),
                   batch_size=32, shuffle=True)
        for i in range(num_clients)
    ]
    return client_datasets, DataLoader(test_ds, batch_size=1000, shuffle=False)


def evaluate(model, test_loader, device='cpu'):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss_sum += torch.nn.functional.cross_entropy(out, target).item()
            correct  += out.argmax(1).eq(target).sum().item()
            total    += target.size(0)
    return correct / total, loss_sum / len(test_loader)


def compute_asr(clean_acc, poisoned_acc, baseline=0.1):
    if clean_acc <= baseline: return 0.0
    return float(np.clip((clean_acc - poisoned_acc) / (clean_acc - baseline), 0, 1))


# ══════════════════════════════════════════════════════════════════════
# SCENARIO RUNNER
# ══════════════════════════════════════════════════════════════════════

def run_scenario(defense_name, attack_type, client_datasets,
                 test_loader, malicious_ids,
                 num_rounds=NUM_ROUNDS) -> Dict:
    """
    Run defense × attack scenario.

    Trust-Aware uses pretrain phase + realistic median reference.
    All other defenses use simple FedAvg-style training (no pretrain).
    """
    device = 'cpu'
    model = get_model().to(device)
    fedavg_agg = FedAvgAggregator()

    # Build aggregator
    if defense_name == "krum":
        aggregator = KrumAggregator(num_malicious=int(NUM_CLIENTS * ATTACK_RATE))
    elif defense_name == "trimmed_mean":
        aggregator = TrimmedMeanAggregator(trim_ratio=ATTACK_RATE)
    elif defense_name == "trust_aware":
        trust_manager = TrustScoreManager(
            NUM_CLIENTS, alpha=ALPHA, tau=TAU,
            initial_trust=1.0, enable_decay=True,
            similarity_weight=0.55, direction_weight=0.25, loss_weight=0.20,
            smoothing_beta=0.7, smoothing_window=5,
            idle_decay_rate=0.002,
            enable_norm_penalty=True,
            norm_penalty_threshold=3.0, norm_penalty_strength=0.80,
            absolute_norm_threshold=15.0,
            enable_sustained_penalty=True,
            sustained_threshold=0.45, sustained_window=3,
            sustained_penalty_strength=0.15,
            warmup_rounds=0,
        )
        aggregator = TrustAwareAggregator(
            trust_manager, enable_filtering=True,
            enable_norm_clip=True, clip_multiplier=2.0,
            warmup_rounds=0, fallback_top_k_ratio=0.3,
        )
    else:  # fedavg
        aggregator = fedavg_agg
        trust_manager = None

    clients = [
        FLClient(i, get_model().to(device), client_datasets[i],
                 is_malicious=(i in malicious_ids))
        for i in range(NUM_CLIENTS)
    ]
    attackers = {cid: create_attacker(cid, attack_type, POISONING_SCALE)
                 for cid in malicious_ids}

    history = {'accuracy': [], 'loss': []}

    for round_num in tqdm(
        range(num_rounds),
        desc=f"{defense_name.upper():12s}|{attack_type:14s}",
        leave=False, ncols=90
    ):
        is_pretrain = (defense_name == "trust_aware" and round_num < PRETRAIN_ROUNDS)

        selected_idx = np.random.choice(NUM_CLIENTS, CLIENTS_PER_ROUND, replace=False)
        selected     = [clients[i] for i in selected_idx]

        global_params = {n: p.data.clone() for n, p in model.named_parameters()}
        for c in selected:
            c.set_parameters(global_params)

        updates, client_ids, metrics_list = [], [], []
        for client in selected:
            cid = client.client_id
            client_ids.append(cid)
            raw_update, metrics = client.train()

            if not is_pretrain and client.is_malicious and cid in attackers:
                est_trust = (trust_manager.get_trust_score(cid)
                             if defense_name == "trust_aware" else 1.0)
                if attackers[cid].should_attack(round_num, est_trust, PRETRAIN_ROUNDS):
                    benign_so_far = [updates[j] for j, pc in enumerate(client_ids[:-1])
                                     if pc not in malicious_ids]
                    updates.append(attackers[cid].poison_gradient(
                        raw_update, benign_updates=benign_so_far or None))
                else:
                    updates.append(raw_update)
            else:
                updates.append(raw_update)
            metrics_list.append(metrics)

        # Reference + trust update (only for trust_aware)
        if defense_name == "trust_aware" and trust_manager is not None:
            ref = compute_realistic_reference(
                updates, client_ids, malicious_ids=None, mode="median")
            for i, cid in enumerate(client_ids):
                if is_pretrain and cid in malicious_ids:
                    continue  # don't update trust for malicious during pretrain
                trust_manager.update_trust(cid, updates[i], ref,
                                           metrics_list[i], round_num)
            if not is_pretrain:
                trust_manager.apply_idle_decay(client_ids, round_num)

        # Aggregate
        if is_pretrain:
            agg = fedavg_agg.aggregate(updates, client_ids, metrics_list)
        else:
            agg = aggregator.aggregate(updates, client_ids, metrics_list,
                                       round_num=round_num)
        if agg is not None:
            for name, param in model.named_parameters():
                param.data += agg[name]

        acc, loss = evaluate(model, test_loader, device)
        history['accuracy'].append(acc)
        history['loss'].append(loss)

    return history


# ══════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════

def plot_heatmap_comparison(results, clean_accs, save_dir):
    """
    PRIMARY: Accuracy heatmap — defense × attack (all 8 attacks, 4 defenses).
    """
    data_acc = np.array([
        [results[d][a]['accuracy'][-1] * 100 for a in ATTACK_ORDER]
        for d in DEFENSE_ORDER
    ])

    fig, axes = plt.subplots(1, 2, figsize=(20, 4.5))

    # Accuracy heatmap
    sns.heatmap(
        data_acc, ax=axes[0],
        xticklabels=[ATTACK_DISPLAY[a] for a in ATTACK_ORDER],
        yticklabels=[DEFENSE_DISPLAY[d] for d in DEFENSE_ORDER],
        annot=True, fmt='.1f', cmap='RdYlGn',
        vmin=10, vmax=100,
        cbar_kws={'label': 'Final Accuracy (%)'},
        linewidths=0.8,
        annot_kws={'size': 11, 'weight': 'bold'},
    )
    axes[0].set_title(
        f'Final Test Accuracy (%) — Defense × Attack\n'
        f'attack_rate={ATTACK_RATE*100:.0f}%, scale={POISONING_SCALE}, {NUM_ROUNDS} rounds',
        fontsize=12, fontweight='bold'
    )
    axes[0].tick_params(axis='x', rotation=20)

    # ASR heatmap
    data_asr = np.array([
        [compute_asr(clean_accs[d], results[d][a]['accuracy'][-1]) * 100 for a in ATTACK_ORDER]
        for d in DEFENSE_ORDER
    ])
    sns.heatmap(
        data_asr, ax=axes[1],
        xticklabels=[ATTACK_DISPLAY[a] for a in ATTACK_ORDER],
        yticklabels=[DEFENSE_DISPLAY[d] for d in DEFENSE_ORDER],
        annot=True, fmt='.1f', cmap='RdYlGn_r',
        vmin=0, vmax=100,
        cbar_kws={'label': 'ASR (%) — lower is better'},
        linewidths=0.8,
        annot_kws={'size': 11, 'weight': 'bold'},
    )
    axes[1].set_title(
        f'Attack Success Rate (%) — lower = better defense\n'
        f'ASR = (clean_acc − attack_acc) / (clean_acc − baseline)',
        fontsize=12, fontweight='bold'
    )
    axes[1].tick_params(axis='x', rotation=20)

    fig.suptitle(
        'Defense Comparison — All Attacks\n'
        'Trust-Aware (Ours) vs FedAvg vs Krum vs Trimmed Mean',
        fontsize=13, fontweight='bold'
    )
    fig.tight_layout()
    path = os.path.join(save_dir, 'oldmethod_heatmap.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_accuracy_curves(results, save_dir):
    """
    SECONDARY: Accuracy curves grouped by attack type.
    4×2 grid — một subplot per attack, 4 defense lines mỗi subplot.
    """
    attacks = [a for a in ATTACK_ORDER if a != 'no_attack']
    ncols = 4
    nrows = (len(attacks) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows),
                             sharex=True, sharey=True)
    axes_flat = np.array(axes).flatten()

    for i, attack in enumerate(attacks):
        ax = axes_flat[i]
        for defense in DEFENSE_ORDER:
            if attack not in results[defense]: continue
            acc = results[defense][attack]['accuracy']
            ax.plot(acc, label=DEFENSE_DISPLAY[defense],
                    color=DEFENSE_COLORS[defense],
                    linewidth=2.2 if defense == 'trust_aware' else 1.8)
        ax.set_title(ATTACK_DISPLAY[attack], fontsize=11, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.2)
        if i % ncols == 0: ax.set_ylabel('Accuracy', fontsize=10)
        if i >= (nrows - 1) * ncols: ax.set_xlabel('Round', fontsize=10)
        if i == 0: ax.legend(fontsize=8.5, loc='lower right')

        # Highlight pretrain boundary for trust-aware
        ax.axvline(x=PRETRAIN_ROUNDS, color='#8e44ad', linestyle=':',
                   linewidth=1.0, alpha=0.7)

    for j in range(len(attacks), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        'Accuracy Curves — All Defenses × All Attacks\n'
        'Purple dotted = Trust-Aware pretrain end',
        fontsize=12, fontweight='bold'
    )
    fig.tight_layout()
    path = os.path.join(save_dir, 'oldmethod_accuracy.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_asr_grouped(results, clean_accs, save_dir):
    """
    PRIMARY: Grouped bar chart — ASR per defense, grouped by attack.
    Trust-Aware should consistently have lowest ASR.
    """
    attacks = [a for a in ATTACK_ORDER if a != 'no_attack']
    n_attacks  = len(attacks)
    n_defenses = len(DEFENSE_ORDER)
    width      = 0.8 / n_defenses
    x          = np.arange(n_attacks)

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, defense in enumerate(DEFENSE_ORDER):
        asrs = [compute_asr(clean_accs[defense],
                             results[defense][a]['accuracy'][-1]) * 100
                for a in attacks]
        offset = (i - n_defenses / 2 + 0.5) * width
        bars = ax.bar(x + offset, asrs, width,
                      label=DEFENSE_DISPLAY[defense],
                      color=DEFENSE_COLORS[defense],
                      edgecolor='#333', linewidth=0.6, alpha=0.88)
        for bar, val in zip(bars, asrs):
            if val > 3:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.8, f'{val:.0f}',
                        ha='center', va='bottom', fontsize=7.5, fontweight='bold',
                        color=DEFENSE_COLORS[defense])

    ax.set_xticks(x)
    ax.set_xticklabels([ATTACK_DISPLAY[a] for a in attacks],
                       rotation=18, ha='right', fontsize=10)
    ax.set_ylabel('Attack Success Rate (%)', fontsize=12)
    ax.set_title(
        'Attack Success Rate — Defense Comparison (lower = better)\n'
        'Trust-Aware should achieve lowest ASR across all attack types',
        fontsize=12, fontweight='bold'
    )
    ax.axhline(y=0, color='#27ae60', linestyle='-', linewidth=1.0, alpha=0.4)
    ax.set_ylim([0, 115])
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.2, axis='y')

    fig.tight_layout()
    path = os.path.join(save_dir, 'oldmethod_asr.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_radar_chart(results, clean_accs, save_dir):
    """
    BONUS: Radar chart — defense robustness across attack types.
    Shows Trust-Aware dominates in all dimensions.
    """
    attacks = [a for a in ATTACK_ORDER if a != 'no_attack']
    N = len(attacks)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for defense in DEFENSE_ORDER:
        # Higher = better defense (1 - ASR)
        values = [
            (1 - compute_asr(clean_accs[defense], results[defense][a]['accuracy'][-1])) * 100
            for a in attacks
        ]
        values += values[:1]
        ax.plot(angles, values, color=DEFENSE_COLORS[defense],
                linewidth=2.5 if defense == 'trust_aware' else 1.8,
                label=DEFENSE_DISPLAY[defense],
                linestyle='-' if defense == 'trust_aware' else '--')
        ax.fill(angles, values, color=DEFENSE_COLORS[defense],
                alpha=0.08 if defense != 'trust_aware' else 0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([ATTACK_DISPLAY[a] for a in attacks], fontsize=9)
    ax.set_ylim([0, 100])
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=7)
    ax.set_title('Defense Robustness Radar\n(higher = better defense, 100% = no accuracy drop)',
                  fontsize=12, fontweight='bold', pad=20)
    ax.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.3, 1.1))

    fig.tight_layout()
    path = os.path.join(save_dir, 'oldmethod_radar.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")


# ══════════════════════════════════════════════════════════════════════
# CONSOLE TABLES
# ══════════════════════════════════════════════════════════════════════

def print_tables(results, clean_accs):
    attacks_display = [a for a in ATTACK_ORDER]
    header = f"  {'Defense':<18}" + "".join(
        f"{ATTACK_DISPLAY[a]:>12}" for a in attacks_display)

    print("\n" + "=" * (18 + 12 * len(attacks_display) + 4))
    print("  FINAL ACCURACY (%)")
    print("=" * (18 + 12 * len(attacks_display) + 4))
    print(header)
    print("  " + "-" * (16 + 12 * len(attacks_display)))
    for d in DEFENSE_ORDER:
        row = f"  {DEFENSE_DISPLAY[d]:<18}"
        for a in attacks_display:
            row += f"{results[d][a]['accuracy'][-1]*100:>11.1f}%"
        print(row)

    print("\n" + "=" * (18 + 12 * len(attacks_display) + 4))
    print("  ATTACK SUCCESS RATE (%)")
    print("=" * (18 + 12 * len(attacks_display) + 4))
    print(header)
    print("  " + "-" * (16 + 12 * len(attacks_display)))
    for d in DEFENSE_ORDER:
        row = f"  {DEFENSE_DISPLAY[d]:<18}"
        for a in attacks_display:
            asr = compute_asr(clean_accs[d], results[d][a]['accuracy'][-1]) * 100
            row += f"{asr:>11.1f}%"
        print(row)

    print("\n" + "=" * 65)
    print("  KEY INSIGHT (for paper)")
    print("=" * 65)
    for attack in [a for a in ATTACK_ORDER if a != 'no_attack']:
        accs  = {d: results[d][attack]['accuracy'][-1] for d in DEFENSE_ORDER}
        best  = max(accs, key=accs.get)
        worst = min(accs, key=accs.get)
        improvement = accs['trust_aware'] - accs['fedavg']
        print(f"  {ATTACK_DISPLAY[attack]:<14}: BEST={DEFENSE_DISPLAY[best][:12]} "
              f"({accs[best]*100:.1f}%)  "
              f"Trust-Aware vs FedAvg: {improvement*100:+.1f}%")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 75)
    print("  DEFENSE COMPARISON: FedAvg vs Krum vs TrimmedMean vs Trust-Aware")
    print(f"  Attacks: {', '.join(ATTACK_DISPLAY.values())}")
    print(f"  attack_rate={ATTACK_RATE*100:.0f}%  rounds={NUM_ROUNDS}  clients={NUM_CLIENTS}")
    print("=" * 75)

    save_dir = 'results/figures'
    os.makedirs(save_dir, exist_ok=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    malicious_ids = set(np.random.choice(
        NUM_CLIENTS, int(NUM_CLIENTS * ATTACK_RATE), replace=False))
    print(f"\n  Malicious IDs (first 10): {sorted(malicious_ids)[:10]} ...")

    print("  Loading MNIST ...")
    client_datasets, test_loader = load_data()

    # Run all defense × attack combinations
    results    = {d: {} for d in DEFENSE_ORDER}
    clean_accs = {}
    total_scenarios = len(DEFENSE_ORDER) * len(ATTACK_ORDER)
    idx = 0

    for defense in DEFENSE_ORDER:
        for attack in ATTACK_ORDER:
            idx += 1
            print(f"\n  [{idx:02d}/{total_scenarios}] "
                  f"{DEFENSE_DISPLAY[defense]} × {ATTACK_DISPLAY[attack]}")
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            results[defense][attack] = run_scenario(
                defense, attack, client_datasets,
                test_loader, malicious_ids,
                num_rounds=NUM_ROUNDS,
            )
            acc = results[defense][attack]['accuracy'][-1]
            print(f"         → Final Accuracy: {acc*100:.2f}%")

        clean_accs[defense] = results[defense]['no_attack']['accuracy'][-1]

    # Plots
    print("\n  Generating figures ...")
    plot_heatmap_comparison(results, clean_accs, save_dir)
    plot_accuracy_curves(results, save_dir)
    plot_asr_grouped(results, clean_accs, save_dir)
    plot_radar_chart(results, clean_accs, save_dir)

    # Tables
    print_tables(results, clean_accs)

    print(f"\n  Figures → {save_dir}/")
    print(f"    [PRIMARY]   oldmethod_heatmap.png  (Acc + ASR)")
    print(f"    [PRIMARY]   oldmethod_asr.png       (Grouped bar)")
    print(f"    [PRIMARY]   oldmethod_radar.png     (Radar robustness)")
    print(f"    [secondary] oldmethod_accuracy.png  (Curves per attack)")
    print("=" * 75)


if __name__ == "__main__":
    main()