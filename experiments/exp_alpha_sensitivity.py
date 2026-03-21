"""
exp_alpha_sensitivity.py — Sensitivity Analysis (v2 Q1-Ready)
==============================================================
Sweep α và τ với cả realistic reference gradient (median).
Thêm: joint heatmap accuracy × trust_separation.

Key insight cho paper:
  "Tại sao chọn α=0.9 và τ=0.3?"
  → Sensitivity analysis chứng minh configuration là robust,
    không chỉ optimal tại một điểm.
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

ATTACK_RATE        = 0.20
POISONING_SCALE    = 5.0
NUM_CLIENTS        = 100
CLIENTS_PER_ROUND  = 20
NUM_ROUNDS         = 50     # Shorter for grid sweep
PRETRAIN_ROUNDS    = 10
SEED               = 42
ABSOLUTE_NORM_THRESHOLD = 15.0
CLIP_MULTIPLIER    = 2.0

ALPHA_RANGE = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
TAU_RANGE   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

# Attack types to test in sensitivity (representative)
SENSITIVITY_ATTACKS = ["static", "adaptive"]

# ══════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════

def load_data(num_clients=NUM_CLIENTS):
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, '..', 'data')
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    spc = len(train_ds) // num_clients
    return (
        [DataLoader(Subset(train_ds, list(range(i * spc, (i + 1) * spc))),
                    batch_size=32, shuffle=True) for i in range(num_clients)],
        DataLoader(test_ds, batch_size=1000, shuffle=False)
    )


# ══════════════════════════════════════════════════════════════════════
# SINGLE RUN
# ══════════════════════════════════════════════════════════════════════

def run_single(alpha, tau, attack_type, client_datasets, test_loader,
               malicious_ids, num_rounds=NUM_ROUNDS):
    """Run one (alpha, tau, attack) combination."""
    model = get_model()
    fedavg = FedAvgAggregator()
    benign_ids = [c for c in range(NUM_CLIENTS) if c not in malicious_ids]

    trust_manager = TrustScoreManager(
        NUM_CLIENTS, alpha=alpha, tau=tau, initial_trust=1.0,
        enable_decay=True,
        similarity_weight=0.55, direction_weight=0.25, loss_weight=0.20,
        smoothing_beta=0.7, smoothing_window=5,
        idle_decay_rate=0.002,
        enable_norm_penalty=True, norm_penalty_threshold=3.0,
        norm_penalty_strength=0.80, absolute_norm_threshold=ABSOLUTE_NORM_THRESHOLD,
        enable_sustained_penalty=True,
        sustained_threshold=0.45, sustained_window=3,
        sustained_penalty_strength=0.15,
        warmup_rounds=0,
    )
    aggregator = TrustAwareAggregator(
        trust_manager, enable_filtering=True,
        enable_norm_clip=True, clip_multiplier=CLIP_MULTIPLIER,
        warmup_rounds=0, fallback_top_k_ratio=0.3,
    )

    clients = [
        FLClient(i, get_model(), client_datasets[i], is_malicious=(i in malicious_ids))
        for i in range(NUM_CLIENTS)
    ]
    attackers = {cid: create_attacker(cid, attack_type, POISONING_SCALE)
                 for cid in malicious_ids}

    accuracy_history = []
    trust_sep_history = []

    for round_num in range(num_rounds):
        is_pretrain = (round_num < PRETRAIN_ROUNDS)
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
                est_trust = trust_manager.get_trust_score(cid)
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

        ref = compute_realistic_reference(updates, client_ids, malicious_ids=None, mode="median")
        for i, cid in enumerate(client_ids):
            if is_pretrain and cid in malicious_ids:
                continue
            trust_manager.update_trust(cid, updates[i], ref, metrics_list[i], round_num)
        if not is_pretrain:
            trust_manager.apply_idle_decay(client_ids, round_num)

        agg = (fedavg.aggregate(updates, client_ids, metrics_list) if is_pretrain
               else aggregator.aggregate(updates, client_ids, metrics_list, round_num))
        if agg is not None:
            for name, param in model.named_parameters():
                param.data += agg[name]

        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                out = model(data)
                correct += out.argmax(1).eq(target).sum().item()
                total   += target.size(0)
        accuracy_history.append(correct / total)

        # Trust separation
        b_scores = [trust_manager.get_trust_score(c) for c in benign_ids]
        m_scores = [trust_manager.get_trust_score(c) for c in malicious_ids]
        trust_sep_history.append(float(np.mean(b_scores)) - float(np.mean(m_scores)))

    return {
        'final_acc':     accuracy_history[-1],
        'min_acc':       min(accuracy_history),
        'acc_history':   accuracy_history,
        'final_sep':     trust_sep_history[-1],
        'mean_sep':      float(np.mean(trust_sep_history[-20:])),  # last 20 rounds
        'sep_history':   trust_sep_history,
    }


# ══════════════════════════════════════════════════════════════════════
# SENSITIVITY SWEEPS
# ══════════════════════════════════════════════════════════════════════

def sweep_alpha(alpha_range, tau=0.3, attack_type="adaptive",
                client_datasets=None, test_loader=None, malicious_ids=None):
    print(f"\n  Alpha sweep (τ={tau}, attack={attack_type})")
    results = {}
    for alpha in alpha_range:
        torch.manual_seed(SEED); np.random.seed(SEED)
        print(f"    α={alpha:.2f} ... ", end='', flush=True)
        r = run_single(alpha, tau, attack_type, client_datasets, test_loader, malicious_ids)
        results[alpha] = r
        print(f"acc={r['final_acc']*100:.2f}%  sep={r['final_sep']:.3f}")
    return results


def sweep_tau(tau_range, alpha=0.9, attack_type="adaptive",
              client_datasets=None, test_loader=None, malicious_ids=None):
    print(f"\n  Tau sweep (α={alpha}, attack={attack_type})")
    results = {}
    for tau in tau_range:
        torch.manual_seed(SEED); np.random.seed(SEED)
        print(f"    τ={tau:.2f} ... ", end='', flush=True)
        r = run_single(alpha, tau, attack_type, client_datasets, test_loader, malicious_ids)
        results[tau] = r
        print(f"acc={r['final_acc']*100:.2f}%  sep={r['final_sep']:.3f}")
    return results


def joint_grid_sweep(alpha_range, tau_range, attack_type="adaptive",
                     client_datasets=None, test_loader=None, malicious_ids=None):
    """Full α × τ grid."""
    print(f"\n  Joint α×τ grid ({len(alpha_range)}×{len(tau_range)}, attack={attack_type})")
    acc_grid = np.zeros((len(alpha_range), len(tau_range)))
    sep_grid = np.zeros_like(acc_grid)
    det_grid = np.zeros_like(acc_grid)

    for i, alpha in enumerate(alpha_range):
        for j, tau in enumerate(tau_range):
            torch.manual_seed(SEED); np.random.seed(SEED)
            r = run_single(alpha, tau, attack_type, client_datasets, test_loader, malicious_ids)
            acc_grid[i, j] = r['final_acc']
            sep_grid[i, j] = r['final_sep']
            print(f"    α={alpha:.2f} τ={tau:.2f} → acc={r['final_acc']*100:.1f}% sep={r['final_sep']:.3f}")

    return acc_grid, sep_grid, alpha_range, tau_range


# ══════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════

def plot_alpha_sweep(alpha_results, tau_results, save_dir):
    """α and τ sensitivity side by side."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Alpha sweep - accuracy
    for alpha, r in sorted(alpha_results.items()):
        axes[0, 0].plot(r['acc_history'], label=f'α={alpha}', linewidth=2.0)
    axes[0, 0].axvline(x=PRETRAIN_ROUNDS, color='gray', linestyle='--',
                        linewidth=1.0, alpha=0.7)
    axes[0, 0].set(xlabel='Round', ylabel='Accuracy',
                   title=f'Accuracy vs α (τ=0.3, attack=adaptive)')
    axes[0, 0].legend(fontsize=8.5); axes[0, 0].grid(True, alpha=0.2)

    # Alpha sweep - final acc + sep
    alphas  = sorted(alpha_results.keys())
    accs_a  = [alpha_results[a]['final_acc'] * 100 for a in alphas]
    seps_a  = [alpha_results[a]['final_sep'] for a in alphas]
    ax_a = axes[0, 1]
    color1, color2 = '#2980b9', '#e74c3c'
    l1, = ax_a.plot(alphas, accs_a, 'o-', color=color1, linewidth=2.2, label='Final Acc (%)')
    ax_a.set_ylabel('Final Accuracy (%)', fontsize=11, color=color1)
    ax_a.tick_params(axis='y', labelcolor=color1)
    ax2 = ax_a.twinx()
    l2, = ax2.plot(alphas, seps_a, 's--', color=color2, linewidth=2.2, label='Trust Sep')
    ax2.set_ylabel('Trust Separation', fontsize=11, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax_a.axvline(x=0.9, color='gray', linestyle=':', linewidth=1.5,
                  alpha=0.8, label='Chosen α=0.9')
    ax_a.set(xlabel='α (EMA decay)', title='Final Accuracy + Trust Sep vs α')
    ax_a.legend(handles=[l1, l2], fontsize=8.5, loc='lower left')
    ax_a.grid(True, alpha=0.2)

    # Tau sweep - accuracy
    for tau, r in sorted(tau_results.items()):
        axes[1, 0].plot(r['acc_history'], label=f'τ={tau}', linewidth=2.0)
    axes[1, 0].axvline(x=PRETRAIN_ROUNDS, color='gray', linestyle='--',
                        linewidth=1.0, alpha=0.7)
    axes[1, 0].set(xlabel='Round', ylabel='Accuracy',
                   title='Accuracy vs τ (α=0.9, attack=adaptive)')
    axes[1, 0].legend(fontsize=8.5); axes[1, 0].grid(True, alpha=0.2)

    # Tau sweep - final acc + sep
    taus   = sorted(tau_results.keys())
    accs_t = [tau_results[t]['final_acc'] * 100 for t in taus]
    seps_t = [tau_results[t]['final_sep'] for t in taus]
    ax_t = axes[1, 1]
    l3, = ax_t.plot(taus, accs_t, 'o-', color=color1, linewidth=2.2, label='Final Acc (%)')
    ax_t.set_ylabel('Final Accuracy (%)', fontsize=11, color=color1)
    ax_t.tick_params(axis='y', labelcolor=color1)
    ax3 = ax_t.twinx()
    l4, = ax3.plot(taus, seps_t, 's--', color=color2, linewidth=2.2, label='Trust Sep')
    ax3.set_ylabel('Trust Separation', fontsize=11, color=color2)
    ax3.tick_params(axis='y', labelcolor=color2)
    ax_t.axvline(x=0.3, color='gray', linestyle=':', linewidth=1.5,
                  alpha=0.8, label='Chosen τ=0.3')
    ax_t.set(xlabel='τ (trust threshold)', title='Final Accuracy + Trust Sep vs τ')
    ax_t.legend(handles=[l3, l4], fontsize=8.5, loc='lower right')
    ax_t.grid(True, alpha=0.2)

    fig.suptitle(
        f'Hyperparameter Sensitivity Analysis — α and τ\n'
        f'attack_rate={ATTACK_RATE*100:.0f}%, {NUM_ROUNDS} rounds, '
        f'ref=median (realistic)',
        fontsize=12, fontweight='bold'
    )
    fig.tight_layout()
    path = os.path.join(save_dir, 'sensitivity_alpha_tau.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_joint_heatmap(acc_grid, sep_grid, alpha_range, tau_range, save_dir,
                       attack_type="adaptive"):
    """Joint α×τ heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(
        acc_grid * 100, ax=axes[0],
        xticklabels=[f'{t:.1f}' for t in tau_range],
        yticklabels=[f'{a:.2f}' for a in alpha_range],
        annot=True, fmt='.1f', cmap='YlGn',
        vmin=50, vmax=100,
        cbar_kws={'label': 'Final Accuracy (%)'},
        linewidths=0.6,
        annot_kws={'size': 12, 'weight': 'bold'},
    )
    # Highlight chosen config
    if 0.9 in alpha_range and 0.3 in tau_range:
        ai = alpha_range.index(0.9)
        ti = tau_range.index(0.3)
        axes[0].add_patch(plt.Rectangle((ti, ai), 1, 1,
                                         fill=False, edgecolor='blue',
                                         linewidth=3, zorder=5))
    axes[0].set(xlabel='τ (trust threshold)', ylabel='α (EMA decay)',
                title=f'Final Accuracy (%) — α×τ Grid\n(attack={attack_type}, blue=chosen config)')

    sns.heatmap(
        sep_grid, ax=axes[1],
        xticklabels=[f'{t:.1f}' for t in tau_range],
        yticklabels=[f'{a:.2f}' for a in alpha_range],
        annot=True, fmt='.3f', cmap='Blues',
        vmin=0, vmax=0.8,
        cbar_kws={'label': 'Trust Separation'},
        linewidths=0.6,
        annot_kws={'size': 12, 'weight': 'bold'},
    )
    if 0.9 in alpha_range and 0.3 in tau_range:
        axes[1].add_patch(plt.Rectangle((ti, ai), 1, 1,
                                         fill=False, edgecolor='blue',
                                         linewidth=3, zorder=5))
    axes[1].set(xlabel='τ (trust threshold)', ylabel='α (EMA decay)',
                title=f'Trust Separation — α×τ Grid\n(higher = better separation)')

    fig.suptitle(
        f'Joint Sensitivity: α × τ\n'
        f'attack_rate={ATTACK_RATE*100:.0f}%, {NUM_ROUNDS} rounds, '
        f'attack={attack_type}, ref=median',
        fontsize=12, fontweight='bold'
    )
    fig.tight_layout()
    path = os.path.join(save_dir, f'sensitivity_joint_{attack_type}.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 65)
    print("  SENSITIVITY ANALYSIS: α and τ")
    print(f"  α range: {ALPHA_RANGE}")
    print(f"  τ range: {TAU_RANGE}")
    print(f"  Attacks: {SENSITIVITY_ATTACKS}")
    print(f"  Rounds: {NUM_ROUNDS}  (pretrain={PRETRAIN_ROUNDS})")
    print("=" * 65)

    save_dir = 'results/figures'
    os.makedirs(save_dir, exist_ok=True)

    np.random.seed(SEED); torch.manual_seed(SEED)
    malicious_ids = set(np.random.choice(
        NUM_CLIENTS, int(NUM_CLIENTS * ATTACK_RATE), replace=False))
    print(f"  Malicious ({len(malicious_ids)}): {sorted(malicious_ids)[:8]} ...")

    client_datasets, test_loader = load_data()
    kwargs = dict(client_datasets=client_datasets, test_loader=test_loader,
                  malicious_ids=malicious_ids)

    # 1. Alpha sweep (vs adaptive attack)
    alpha_results = sweep_alpha(ALPHA_RANGE, tau=0.3, attack_type="adaptive", **kwargs)

    # 2. Tau sweep (vs adaptive attack)
    tau_results = sweep_tau(TAU_RANGE, alpha=0.9, attack_type="adaptive", **kwargs)

    # 3. Plot alpha + tau together
    plot_alpha_sweep(alpha_results, tau_results, save_dir)

    # 4. Joint grid for each attack type
    # Smaller grid for speed
    small_alpha = [0.7, 0.8, 0.9, 0.95]
    small_tau   = [0.1, 0.2, 0.3, 0.4, 0.5]

    for attack in SENSITIVITY_ATTACKS:
        print(f"\n  Joint grid for attack={attack} ...")
        acc_grid, sep_grid, ar, tr = joint_grid_sweep(
            small_alpha, small_tau, attack_type=attack, **kwargs)
        plot_joint_heatmap(acc_grid, sep_grid, ar, tr, save_dir, attack_type=attack)

    # Summary table
    print("\n" + "=" * 65)
    print("  ALPHA SENSITIVITY SUMMARY (τ=0.3, attack=adaptive)")
    print("=" * 65)
    print(f"  {'α':<8} {'Final Acc':<14} {'Trust Sep':<12} {'Assessment'}")
    print("  " + "-" * 55)
    for alpha in ALPHA_RANGE:
        r = alpha_results[alpha]
        acc, sep = r['final_acc'] * 100, r['final_sep']
        note = "← CHOSEN" if abs(alpha - 0.9) < 0.01 else \
               ("✓ good" if acc > 90 else "✗ poor")
        print(f"  {alpha:<8.2f} {acc:<14.2f}% {sep:<12.3f} {note}")

    print("\n" + "=" * 65)
    print("  TAU SENSITIVITY SUMMARY (α=0.9, attack=adaptive)")
    print("=" * 65)
    print(f"  {'τ':<8} {'Final Acc':<14} {'Trust Sep':<12} {'Assessment'}")
    print("  " + "-" * 55)
    for tau in TAU_RANGE:
        r = tau_results[tau]
        acc, sep = r['final_acc'] * 100, r['final_sep']
        note = "← CHOSEN" if abs(tau - 0.3) < 0.01 else \
               ("✓ good" if acc > 90 else "✗ poor")
        print(f"  {tau:<8.2f} {acc:<14.2f}% {sep:<12.3f} {note}")

    print(f"\n  Figures → {save_dir}/")
    print(f"    sensitivity_alpha_tau.png        [PRIMARY]")
    print(f"    sensitivity_joint_adaptive.png   [PRIMARY]")
    print(f"    sensitivity_joint_static.png     [secondary]")


if __name__ == "__main__":
    main()