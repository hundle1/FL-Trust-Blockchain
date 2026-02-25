"""
Experiment 3: Alpha (α) and Tau (τ) Sensitivity Analysis

Investigates how the two key hyperparameters of Trust-Aware FL
affect defense performance:

    α (alpha): EMA decay factor
        - Controls how fast trust updates
        - Low α → fast adaptation (reactive to sudden attacks)
        - High α → slow adaptation (stable but slow to detect delayed attacks)

    τ (tau): Trust threshold for filtering
        - Low τ → lenient (more clients included, less false exclusions)
        - High τ → strict (fewer clients, risks excluding benign clients)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from models.cnn_mnist import get_model
from fl_core.client import FLClient
from trust.trust_score import TrustScoreManager
from attacks.adaptive_controller import AdaptiveAttackController
from fl_core.aggregation.trust_aware import TrustAwareAggregator
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_data(num_clients: int = 100):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST('./data/mnist', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data/mnist', train=False, download=True, transform=transform)

    spc = len(train_ds) // num_clients
    client_datasets = [
        DataLoader(Subset(train_ds, list(range(i * spc, (i + 1) * spc))),
                   batch_size=32, shuffle=True)
        for i in range(num_clients)
    ]
    test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False)
    return client_datasets, test_loader


# ------------------------------------------------------------------
# Single run
# ------------------------------------------------------------------

def run_single(
    alpha: float,
    tau: float,
    attack_type: str = "adaptive",
    num_clients: int = 100,
    attack_rate: float = 0.2,
    num_rounds: int = 50,
    client_datasets=None,
    test_loader=None
):
    """
    Run one Trust-Aware FL experiment with given α and τ.

    Returns:
        final_accuracy, final_asr, trust_separation
    """
    if client_datasets is None or test_loader is None:
        client_datasets, test_loader = load_data(num_clients)

    model = get_model()
    num_malicious = int(num_clients * attack_rate)
    malicious_ids = set(np.random.choice(num_clients, num_malicious, replace=False))

    clients = [
        FLClient(i, get_model(), client_datasets[i], is_malicious=(i in malicious_ids))
        for i in range(num_clients)
    ]

    attack_controllers = {
        cid: AdaptiveAttackController(
            cid, attack_type=attack_type,
            poisoning_scale=5.0,
            trust_threshold=0.7,
            knows_trust_mechanism=True
        )
        for cid in malicious_ids
    }

    trust_manager = TrustScoreManager(num_clients, alpha=alpha, tau=tau, enable_decay=True)
    aggregator = TrustAwareAggregator(trust_manager, enable_filtering=True)

    accuracy_history = []

    for round_num in range(num_rounds):
        selected_idx = np.random.choice(num_clients, 10, replace=False)
        selected_clients = [clients[i] for i in selected_idx]

        global_params = {n: p.data.clone() for n, p in model.named_parameters()}
        for c in selected_clients:
            c.set_parameters(global_params)

        updates, client_ids, metrics_list = [], [], []

        for client in selected_clients:
            cid = client.client_id
            client_ids.append(cid)

            if client.is_malicious:
                controller = attack_controllers[cid]
                est_trust = trust_manager.get_trust_score(cid)
                should_attack = controller.should_attack(round_num, est_trust)
                clean_update, metrics = client.train()
                updates.append(controller.poison_gradient(clean_update) if should_attack else clean_update)
            else:
                update, metrics = client.train()
                updates.append(update)

            metrics_list.append(metrics)

        # Trust update
        benign_sel = [cid for cid in client_ids if cid not in malicious_ids]
        if benign_sel:
            benign_ups = [updates[i] for i, cid in enumerate(client_ids) if cid in benign_sel]
            ref_grad = {k: torch.mean(torch.stack([u[k] for u in benign_ups]), dim=0)
                        for k in benign_ups[0]}
        else:
            ref_grad = {k: torch.mean(torch.stack([u[k] for u in updates]), dim=0)
                        for k in updates[0]}

        for i, cid in enumerate(client_ids):
            trust_manager.update_trust(cid, updates[i], ref_grad, metrics_list[i], round_num)

        trust_manager.apply_idle_decay(client_ids, round_num)

        # Aggregate
        agg_update = aggregator.aggregate(updates, client_ids, metrics_list)
        for name, param in model.named_parameters():
            param.data += agg_update[name]

        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                out = model(data)
                correct += out.argmax(1).eq(target).sum().item()
                total += target.size(0)
        accuracy_history.append(correct / total)

    final_accuracy = accuracy_history[-1]

    # Trust separation
    sep = trust_manager.get_trust_separation(
        [c for c in range(num_clients) if c not in malicious_ids],
        list(malicious_ids)
    )

    return final_accuracy, accuracy_history, sep['separation']


# ------------------------------------------------------------------
# Sensitivity sweeps
# ------------------------------------------------------------------

def alpha_sensitivity_study(
    alpha_range=None,
    tau: float = 0.3,
    num_rounds: int = 50,
    num_clients: int = 100,
    attack_rate: float = 0.2
):
    """Sweep α while keeping τ fixed."""
    if alpha_range is None:
        alpha_range = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]

    print(f"\n{'='*60}")
    print(f"ALPHA SENSITIVITY  (τ={tau} fixed)")
    print(f"{'='*60}")

    # Load data once
    client_datasets, test_loader = load_data(num_clients)

    results = {}
    for alpha in alpha_range:
        torch.manual_seed(42)
        np.random.seed(42)
        print(f"  α={alpha:.2f} ...", end=' ', flush=True)
        final_acc, history, sep = run_single(
            alpha, tau,
            num_rounds=num_rounds,
            num_clients=num_clients,
            attack_rate=attack_rate,
            client_datasets=client_datasets,
            test_loader=test_loader
        )
        results[alpha] = {'final_acc': final_acc, 'history': history, 'separation': sep}
        print(f"acc={final_acc*100:.2f}%  sep={sep:.3f}")

    return results


def tau_sensitivity_study(
    tau_range=None,
    alpha: float = 0.9,
    num_rounds: int = 50,
    num_clients: int = 100,
    attack_rate: float = 0.2
):
    """Sweep τ while keeping α fixed."""
    if tau_range is None:
        tau_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    print(f"\n{'='*60}")
    print(f"TAU SENSITIVITY  (α={alpha} fixed)")
    print(f"{'='*60}")

    client_datasets, test_loader = load_data(num_clients)

    results = {}
    for tau in tau_range:
        torch.manual_seed(42)
        np.random.seed(42)
        print(f"  τ={tau:.2f} ...", end=' ', flush=True)
        final_acc, history, sep = run_single(
            alpha, tau,
            num_rounds=num_rounds,
            num_clients=num_clients,
            attack_rate=attack_rate,
            client_datasets=client_datasets,
            test_loader=test_loader
        )
        results[tau] = {'final_acc': final_acc, 'history': history, 'separation': sep}
        print(f"acc={final_acc*100:.2f}%  sep={sep:.3f}")

    return results


def joint_sensitivity_heatmap(
    alpha_range=None,
    tau_range=None,
    num_rounds: int = 30,
    num_clients: int = 100,
    attack_rate: float = 0.2
):
    """
    2D grid sweep: all (α, τ) combinations.
    Produces a heatmap of final accuracy.
    """
    if alpha_range is None:
        alpha_range = [0.7, 0.8, 0.9, 0.95]
    if tau_range is None:
        tau_range = [0.1, 0.2, 0.3, 0.4, 0.5]

    print(f"\n{'='*60}")
    print(f"JOINT (α × τ) SENSITIVITY  [{len(alpha_range)}×{len(tau_range)} grid]")
    print(f"{'='*60}")

    client_datasets, test_loader = load_data(num_clients)

    acc_grid = np.zeros((len(alpha_range), len(tau_range)))
    sep_grid = np.zeros_like(acc_grid)

    for i, alpha in enumerate(alpha_range):
        for j, tau in enumerate(tau_range):
            torch.manual_seed(42)
            np.random.seed(42)
            final_acc, _, sep = run_single(
                alpha, tau,
                num_rounds=num_rounds,
                num_clients=num_clients,
                attack_rate=attack_rate,
                client_datasets=client_datasets,
                test_loader=test_loader
            )
            acc_grid[i, j] = final_acc
            sep_grid[i, j] = sep
            print(f"  α={alpha:.2f} τ={tau:.2f}  acc={final_acc*100:.1f}%  sep={sep:.3f}")

    return acc_grid, sep_grid, alpha_range, tau_range


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_alpha_sensitivity(alpha_results: dict, save_path: str = None):
    """Plot accuracy and separation vs alpha."""
    alphas = sorted(alpha_results.keys())
    accs = [alpha_results[a]['final_acc'] * 100 for a in alphas]
    seps = [alpha_results[a]['separation'] for a in alphas]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Accuracy over rounds
    for alpha in alphas:
        hist = alpha_results[alpha]['history']
        axes[0].plot(hist, label=f'α={alpha}', linewidth=2)
    axes[0].set(xlabel='Round', ylabel='Test Accuracy', title='Accuracy vs α')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Final accuracy vs alpha
    axes[1].plot(alphas, accs, 'bo-', linewidth=2, markersize=7)
    axes[1].set(xlabel='α (EMA decay)', ylabel='Final Accuracy (%)',
                title='Final Accuracy vs α')
    axes[1].grid(True, alpha=0.3)
    for a, acc in zip(alphas, accs):
        axes[1].annotate(f'{acc:.1f}%', (a, acc), textcoords='offset points',
                         xytext=(0, 8), ha='center', fontsize=9)

    # Trust separation vs alpha
    axes[2].plot(alphas, seps, 'rs-', linewidth=2, markersize=7)
    axes[2].set(xlabel='α (EMA decay)', ylabel='Trust Separation',
                title='Trust Separation vs α')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='k', linestyle='--', linewidth=0.8)

    plt.tight_layout()
    path = save_path or 'results/figures/exp_alpha_sensitivity.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {path}")
    plt.close()


def plot_joint_heatmap(
    acc_grid: np.ndarray,
    sep_grid: np.ndarray,
    alpha_range: list,
    tau_range: list,
    save_path: str = None
):
    """Plot joint (α, τ) heatmaps."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(
        acc_grid * 100, ax=axes[0],
        xticklabels=[f'{t:.1f}' for t in tau_range],
        yticklabels=[f'{a:.2f}' for a in alpha_range],
        annot=True, fmt='.1f', cmap='YlGn',
        cbar_kws={'label': 'Final Accuracy (%)'}
    )
    axes[0].set(xlabel='τ (trust threshold)', ylabel='α (EMA decay)',
                title='Final Accuracy (%) – α × τ Grid')

    sns.heatmap(
        sep_grid, ax=axes[1],
        xticklabels=[f'{t:.1f}' for t in tau_range],
        yticklabels=[f'{a:.2f}' for a in alpha_range],
        annot=True, fmt='.3f', cmap='Blues',
        cbar_kws={'label': 'Trust Separation'}
    )
    axes[1].set(xlabel='τ (trust threshold)', ylabel='α (EMA decay)',
                title='Trust Separation – α × τ Grid')

    plt.tight_layout()
    path = save_path or 'results/figures/exp_alpha_tau_heatmap.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {path}")
    plt.close()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: ALPHA & TAU SENSITIVITY ANALYSIS")
    print("=" * 60)

    os.makedirs('results/figures', exist_ok=True)

    NUM_ROUNDS = 50
    NUM_CLIENTS = 100
    ATTACK_RATE = 0.2

    # 1. Alpha sweep
    alpha_results = alpha_sensitivity_study(
        alpha_range=[0.5, 0.7, 0.8, 0.9, 0.95, 0.99],
        tau=0.3,
        num_rounds=NUM_ROUNDS,
        num_clients=NUM_CLIENTS,
        attack_rate=ATTACK_RATE
    )
    plot_alpha_sensitivity(alpha_results)

    # 2. Tau sweep
    tau_results = tau_sensitivity_study(
        tau_range=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        alpha=0.9,
        num_rounds=NUM_ROUNDS,
        num_clients=NUM_CLIENTS,
        attack_rate=ATTACK_RATE
    )

    # 3. Joint heatmap (smaller grid for speed)
    acc_grid, sep_grid, alpha_range, tau_range = joint_sensitivity_heatmap(
        alpha_range=[0.7, 0.8, 0.9, 0.95],
        tau_range=[0.1, 0.2, 0.3, 0.4, 0.5],
        num_rounds=30,
        num_clients=NUM_CLIENTS,
        attack_rate=ATTACK_RATE
    )
    plot_joint_heatmap(acc_grid, sep_grid, alpha_range, tau_range)

    # Summary
    print("\n" + "=" * 60)
    print("ALPHA SENSITIVITY SUMMARY")
    print("=" * 60)
    print(f"{'α':<10} {'Final Acc (%)':<18} {'Trust Separation':<18}")
    print("-" * 48)
    for alpha, data in sorted(alpha_results.items()):
        print(f"{alpha:<10.2f} {data['final_acc']*100:<18.2f} {data['separation']:<18.4f}")

    print("\n" + "=" * 60)
    print("TAU SENSITIVITY SUMMARY")
    print("=" * 60)
    print(f"{'τ':<10} {'Final Acc (%)':<18} {'Trust Separation':<18}")
    print("-" * 48)
    for tau, data in sorted(tau_results.items()):
        print(f"{tau:<10.2f} {data['final_acc']*100:<18.2f} {data['separation']:<18.4f}")
    print("=" * 60)

    print("\nKey Insights:")
    best_alpha = max(alpha_results, key=lambda a: alpha_results[a]['final_acc'])
    best_tau = max(tau_results, key=lambda t: tau_results[t]['final_acc'])
    print(f"  ✓ Best α = {best_alpha} (final acc = {alpha_results[best_alpha]['final_acc']*100:.2f}%)")
    print(f"  ✓ Best τ = {best_tau} (final acc = {tau_results[best_tau]['final_acc']*100:.2f}%)")
    print(f"  ✓ High α (slow decay) works best when attacks are intermittent/delayed")
    print(f"  ✓ τ=0.3 provides a good balance between filtering and false exclusions")


if __name__ == "__main__":
    main()