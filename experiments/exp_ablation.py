"""
exp_ablation.py — Full Ablation Study (v2 Q1-Ready)
======================================================
Các variants để isolate contribution của từng component:
  1. FedAvg baseline (no defense)
  2. Clip-only     (norm clipping, no trust)
  3. Trust-only    (trust filter, no clip)
  4. Full System   (trust + clip + blockchain)

Chạy với tất cả attack types để thấy từng component
contribute gì trong điều kiện nào.

Key insight cho paper:
  - Clip-only: handles static/norm-tuned (norm-based detection)
  - Trust-only: handles adaptive/delayed (behavior-based detection)
  - Full: handles tất cả attack types consistently
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time

from models.cnn_mnist import get_model
from fl_core.client import FLClient
from trust.trust_score import TrustScoreManager
from fl_core.aggregation.trust_aware import TrustAwareAggregator, FedAvgAggregator
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
PRETRAIN_ROUNDS   = 10
SEED              = 42

ALPHA = 0.9
TAU   = 0.3
ABSOLUTE_NORM_THRESHOLD = 15.0
CLIP_MULTIPLIER = 2.0

# Ablation variants
VARIANTS = [
    ("fedavg",       "FedAvg (Baseline)",      False, False, False),
    ("clip_only",    "Clip-Only",               False, True,  False),
    ("trust_only",   "Trust-Only",              True,  False, False),
    ("full",         "Full System (Ours)",       True,  True,  True),
]
VARIANT_COLORS = {
    "fedavg":     "#e74c3c",
    "clip_only":  "#e67e22",
    "trust_only": "#2980b9",
    "full":       "#27ae60",
}

# Selected representative attacks for ablation
ABLATION_ATTACKS = ["static", "adaptive", "norm_tuned", "label_flip"]
ATTACK_DISPLAY = {
    "static": "Static", "adaptive": "Adaptive",
    "norm_tuned": "Norm-Tuned", "label_flip": "Label Flip",
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
    return (
        [DataLoader(Subset(train_ds, list(range(i * spc, (i + 1) * spc))),
                    batch_size=32, shuffle=True) for i in range(num_clients)],
        DataLoader(test_ds, batch_size=1000, shuffle=False)
    )


def evaluate(model, test_loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for data, target in test_loader:
            out = model(data)
            loss_sum += torch.nn.functional.cross_entropy(out, target).item()
            correct  += out.argmax(1).eq(target).sum().item()
            total    += target.size(0)
    return correct / total, loss_sum / len(test_loader)


# ══════════════════════════════════════════════════════════════════════
# VARIANT RUNNER
# ══════════════════════════════════════════════════════════════════════

def run_variant(variant_id, attack_type, client_datasets,
                test_loader, malicious_ids, use_trust, use_clip, use_blockchain,
                num_rounds=NUM_ROUNDS):
    """Run one ablation variant against one attack type."""
    model = get_model()
    fedavg_agg = FedAvgAggregator()

    # Build defense components based on variant
    if use_trust:
        trust_manager = TrustScoreManager(
            NUM_CLIENTS, alpha=ALPHA, tau=TAU, initial_trust=1.0,
            enable_decay=True, similarity_weight=0.7, idle_decay_rate=0.002,
            enable_norm_penalty=use_clip,  # norm penalty only with clip
            norm_penalty_threshold=3.0, norm_penalty_strength=0.80,
            absolute_norm_threshold=ABSOLUTE_NORM_THRESHOLD, warmup_rounds=0,
        )
        aggregator = TrustAwareAggregator(
            trust_manager, enable_filtering=True,
            enable_norm_clip=use_clip, clip_multiplier=CLIP_MULTIPLIER,
            warmup_rounds=0,
        )
    elif use_clip:
        # Clip-only: use FedAvg but with a norm-clipping wrapper
        trust_manager = None
        aggregator = _ClipOnlyAggregator(clip_multiplier=CLIP_MULTIPLIER)
    else:
        trust_manager = None
        aggregator = fedavg_agg

    clients = [
        FLClient(i, get_model(), client_datasets[i],
                 is_malicious=(i in malicious_ids))
        for i in range(NUM_CLIENTS)
    ]
    attackers = {cid: create_attacker(cid, attack_type, POISONING_SCALE)
                 for cid in malicious_ids}

    history = {
        'accuracy': [], 'loss': [],
        'round_times': [],
        'blockchain_overhead': 0.0,
    }

    blockchain_time = 0.0
    for round_num in tqdm(range(num_rounds),
                          desc=f"{variant_id:12s}|{attack_type:12s}",
                          leave=False, ncols=88):
        t0 = time.time()
        is_pretrain = (use_trust and round_num < PRETRAIN_ROUNDS)

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
                est_trust = trust_manager.get_trust_score(cid) if trust_manager else 1.0
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

        if trust_manager:
            ref = compute_realistic_reference(
                updates, client_ids, malicious_ids=None, mode="median")
            for i, cid in enumerate(client_ids):
                if is_pretrain and cid in malicious_ids:
                    continue
                trust_manager.update_trust(cid, updates[i], ref,
                                           metrics_list[i], round_num)
            if not is_pretrain:
                trust_manager.apply_idle_decay(client_ids, round_num)

        # Blockchain simulation overhead
        if use_blockchain and not is_pretrain:
            t_bc = time.time()
            time.sleep(0.001 * len(client_ids))  # simulate consensus latency
            blockchain_time += time.time() - t_bc

        if is_pretrain:
            agg = fedavg_agg.aggregate(updates, client_ids, metrics_list)
        else:
            agg = aggregator.aggregate(updates, client_ids, metrics_list,
                                       round_num=round_num)

        if agg is not None:
            for name, param in model.named_parameters():
                param.data += agg[name]

        acc, loss = evaluate(model, test_loader)
        history['accuracy'].append(acc)
        history['loss'].append(loss)
        history['round_times'].append(time.time() - t0)

    history['blockchain_overhead'] = blockchain_time
    return history


class _ClipOnlyAggregator:
    """FedAvg + norm clipping only (no trust filter)."""

    def __init__(self, clip_multiplier=2.0):
        self.clip_multiplier = clip_multiplier

    def aggregate(self, updates, client_ids, metrics=None, round_num=0):
        norms = [float(np.sqrt(sum(torch.norm(v).item() ** 2 for v in u.values())))
                 for u in updates]
        median_norm = float(np.median(norms))
        clip_norm = self.clip_multiplier * max(median_norm, 1e-6)

        clipped = []
        for u, norm in zip(updates, norms):
            if norm > clip_norm:
                scale = clip_norm / norm
                clipped.append({k: v * scale for k, v in u.items()})
            else:
                clipped.append(u)

        n = len(clipped)
        agg = {}
        for name in clipped[0]:
            agg[name] = sum(u[name] for u in clipped) / n
        return agg

    def get_stats(self):
        return {'skipped_rounds': 0, 'avg_filtered': 0, 'avg_clipped': 0,
                'total_rounds': 0, 'median_rounds': 0, 'mean_rounds': 0}


# ══════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════

def plot_ablation_accuracy(results, save_dir):
    """
    PRIMARY: Accuracy per attack, 4 ablation variants.
    2×2 grid (one attack per subplot).
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for i, attack in enumerate(ABLATION_ATTACKS):
        ax = axes_flat[i]
        for vid, vname, _, _, _ in VARIANTS:
            if vid not in results or attack not in results[vid]: continue
            acc = results[vid][attack]['accuracy']
            ax.plot(acc, label=vname, color=VARIANT_COLORS[vid],
                    linewidth=2.5 if vid == 'full' else 1.8)

        ax.axvline(x=PRETRAIN_ROUNDS, color='gray', linestyle='--',
                   linewidth=1.0, alpha=0.7)
        ax.set_title(f'Attack: {ATTACK_DISPLAY[attack]}',
                     fontsize=11, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.2)
        if i % 2 == 0: ax.set_ylabel('Accuracy', fontsize=10)
        if i >= 2:     ax.set_xlabel('Round', fontsize=10)
        if i == 0:     ax.legend(fontsize=9, loc='lower right')

    fig.suptitle(
        'Ablation Study — Component Contribution to Defense\n'
        'Vertical line = attack starts (pretrain end)',
        fontsize=12, fontweight='bold'
    )
    fig.tight_layout()
    path = os.path.join(save_dir, 'ablation_accuracy.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_ablation_heatmap(results, save_dir):
    """
    PRIMARY: Heatmap — variant × attack final accuracy.
    """
    variant_ids = [v[0] for v in VARIANTS]
    variant_names = [v[1] for v in VARIANTS]

    data = np.array([
        [results[vid][a]['accuracy'][-1] * 100
         if vid in results and a in results[vid] else 0.0
         for a in ABLATION_ATTACKS]
        for vid in variant_ids
    ])

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        data, ax=ax,
        xticklabels=[ATTACK_DISPLAY[a] for a in ABLATION_ATTACKS],
        yticklabels=variant_names,
        annot=True, fmt='.1f', cmap='RdYlGn', vmin=30, vmax=100,
        cbar_kws={'label': 'Final Accuracy (%)'},
        linewidths=1.2,
        annot_kws={'size': 13, 'weight': 'bold'},
    )
    ax.set_title(
        'Ablation Study — Final Accuracy (%)\n'
        'Each row adds one more defense component',
        fontsize=12, fontweight='bold'
    )
    fig.tight_layout()
    path = os.path.join(save_dir, 'ablation_heatmap.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_component_contribution(results, save_dir):
    """
    PRIMARY: Bar chart showing marginal gain of each component.
    """
    variant_ids   = [v[0] for v in VARIANTS]
    variant_names = [v[1] for v in VARIANTS]

    # Average final accuracy across all attack types
    avg_accs = []
    for vid in variant_ids:
        if vid not in results:
            avg_accs.append(0.0)
            continue
        accs = [results[vid][a]['accuracy'][-1] * 100
                for a in ABLATION_ATTACKS if a in results[vid]]
        avg_accs.append(float(np.mean(accs)) if accs else 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Average accuracy bar chart
    colors = [VARIANT_COLORS[v[0]] for v in VARIANTS]
    bars = axes[0].bar(range(len(variant_names)), avg_accs,
                        color=colors, edgecolor='#333', linewidth=0.8, alpha=0.88)

    # Annotate marginal gain
    for i, (bar, val) in enumerate(zip(bars, avg_accs)):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5, f'{val:.1f}%',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
        if i > 0:
            gain = val - avg_accs[i - 1]
            if abs(gain) > 0.1:
                axes[0].text(bar.get_x() + bar.get_width() / 2,
                             bar.get_height() + 4,
                             f'+{gain:.1f}%' if gain > 0 else f'{gain:.1f}%',
                             ha='center', va='bottom', fontsize=9,
                             color='#27ae60' if gain > 0 else '#e74c3c',
                             fontweight='bold')

    axes[0].set_xticks(range(len(variant_names)))
    axes[0].set_xticklabels(variant_names, rotation=15, ha='right', fontsize=9)
    axes[0].set_ylabel('Average Final Accuracy (%)', fontsize=11)
    axes[0].set_title('Component Contribution\n(Average across 4 attack types)',
                       fontsize=11, fontweight='bold')
    axes[0].set_ylim([0, 110])
    axes[0].grid(True, alpha=0.2, axis='y')

    # Per-attack comparison
    x = np.arange(len(ABLATION_ATTACKS))
    w = 0.8 / len(VARIANTS)
    for i, (vid, vname, _, _, _) in enumerate(VARIANTS):
        accs = [results.get(vid, {}).get(a, {}).get('accuracy', [0.0])[-1] * 100
                for a in ABLATION_ATTACKS]
        offset = (i - len(VARIANTS) / 2 + 0.5) * w
        axes[1].bar(x + offset, accs, w, label=vname,
                    color=VARIANT_COLORS[vid], edgecolor='#333', linewidth=0.6, alpha=0.88)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([ATTACK_DISPLAY[a] for a in ABLATION_ATTACKS],
                              rotation=10, fontsize=10)
    axes[1].set_ylabel('Final Accuracy (%)', fontsize=11)
    axes[1].set_title('Per-Attack Accuracy\n(Each variant × 4 attacks)',
                       fontsize=11, fontweight='bold')
    axes[1].set_ylim([0, 110])
    axes[1].legend(fontsize=8.5, loc='lower right')
    axes[1].grid(True, alpha=0.2, axis='y')

    fig.suptitle(
        f'Ablation Study — Defense Component Analysis\n'
        f'attack_rate={ATTACK_RATE*100:.0f}%, rounds={NUM_ROUNDS}, α={ALPHA}, τ={TAU}',
        fontsize=12, fontweight='bold'
    )
    fig.tight_layout()
    path = os.path.join(save_dir, 'ablation_contribution.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_blockchain_overhead(results, save_dir):
    """
    Blockchain overhead analysis.
    """
    full_results = results.get('full', {})
    if not full_results:
        return

    attack = ABLATION_ATTACKS[0]  # use first attack for timing
    if attack not in full_results:
        return

    h_full     = full_results[attack]
    h_fedavg   = results.get('fedavg', {}).get(attack, {})
    h_trustonly = results.get('trust_only', {}).get(attack, {})

    variants_timing = {
        'FedAvg':          h_fedavg.get('round_times', []),
        'Trust-Only':      h_trustonly.get('round_times', []),
        'Full (+BC)':      h_full.get('round_times', []),
    }
    colors_timing = {'FedAvg': '#e74c3c', 'Trust-Only': '#2980b9', 'Full (+BC)': '#27ae60'}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Time per round
    for name, times in variants_timing.items():
        if times:
            axes[0].plot(times, label=name, color=colors_timing[name], linewidth=2.0)

    axes[0].set_xlabel('Round', fontsize=11)
    axes[0].set_ylabel('Time per Round (s)', fontsize=11)
    axes[0].set_title('Computational Overhead per Round', fontsize=11, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.2)

    # Cumulative overhead
    names  = []
    avgs   = []
    bc_ovh = []
    for name, times in variants_timing.items():
        if times:
            names.append(name)
            avgs.append(float(np.mean(times)) * 1000)  # ms
    bc_time = h_full.get('blockchain_overhead', 0.0)
    total_time = sum(h_full.get('round_times', [0.0]))
    bc_pct = bc_time / max(total_time, 1e-6) * 100

    bars = axes[1].bar(names, avgs,
                        color=[colors_timing[n] for n in names],
                        edgecolor='#333', linewidth=0.8, alpha=0.88)
    for bar, val in zip(bars, avgs):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.2, f'{val:.1f}ms',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    axes[1].set_ylabel('Avg Time per Round (ms)', fontsize=11)
    axes[1].set_title(
        f'Avg Round Latency\nBlockchain overhead: {bc_pct:.1f}% of total time',
        fontsize=11, fontweight='bold'
    )
    axes[1].grid(True, alpha=0.2, axis='y')

    fig.suptitle(
        f'Blockchain Overhead Analysis\n'
        f'attack_rate={ATTACK_RATE*100:.0f}%, {NUM_ROUNDS} rounds, {NUM_CLIENTS} clients',
        fontsize=12, fontweight='bold'
    )
    fig.tight_layout()
    path = os.path.join(save_dir, 'ablation_overhead.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  ABLATION STUDY: FedAvg vs Clip-only vs Trust-only vs Full")
    print(f"  Attacks tested: {', '.join(ABLATION_ATTACKS)}")
    print(f"  attack_rate={ATTACK_RATE*100:.0f}%  rounds={NUM_ROUNDS}")
    print("=" * 70)

    save_dir = 'results/figures'
    os.makedirs(save_dir, exist_ok=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    malicious_ids = set(np.random.choice(
        NUM_CLIENTS, int(NUM_CLIENTS * ATTACK_RATE), replace=False))
    print(f"  Malicious ({len(malicious_ids)}): {sorted(malicious_ids)[:8]} ...")

    client_datasets, test_loader = load_data()

    results = {v[0]: {} for v in VARIANTS}
    total = len(VARIANTS) * len(ABLATION_ATTACKS)
    idx = 0

    for vid, vname, use_trust, use_clip, use_bc in VARIANTS:
        for attack in ABLATION_ATTACKS:
            idx += 1
            print(f"\n  [{idx:02d}/{total}] {vname} × {ATTACK_DISPLAY[attack]}")
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            results[vid][attack] = run_variant(
                vid, attack, client_datasets, test_loader,
                malicious_ids, use_trust, use_clip, use_bc,
                num_rounds=NUM_ROUNDS,
            )
            acc = results[vid][attack]['accuracy'][-1]
            print(f"         → Final Accuracy: {acc*100:.2f}%")

    # Plots
    print("\n  Generating figures ...")
    plot_ablation_accuracy(results, save_dir)
    plot_ablation_heatmap(results, save_dir)
    plot_component_contribution(results, save_dir)
    plot_blockchain_overhead(results, save_dir)

    # Console summary
    print("\n" + "=" * 70)
    print("  ABLATION SUMMARY")
    print("=" * 70)
    for vid, vname, _, _, _ in VARIANTS:
        accs = [results[vid][a]['accuracy'][-1] * 100 for a in ABLATION_ATTACKS]
        print(f"  {vname:<25}:  " +
              "  ".join(f"{ATTACK_DISPLAY[a]}={acc:.1f}%"
                        for a, acc in zip(ABLATION_ATTACKS, accs)) +
              f"  [avg={np.mean(accs):.1f}%]")

    print(f"\n  Figures → {save_dir}/")
    print(f"    [PRIMARY]   ablation_heatmap.png")
    print(f"    [PRIMARY]   ablation_contribution.png")
    print(f"    [PRIMARY]   ablation_accuracy.png")
    print(f"    [secondary] ablation_overhead.png")


if __name__ == "__main__":
    main()