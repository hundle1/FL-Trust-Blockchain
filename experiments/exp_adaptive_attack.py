"""
Experiment 2: Adaptive Attack vs Trust-Aware Defense
Tests how adaptive attackers (who know about the trust mechanism)
behave against Trust-Aware FL compared to static attackers.

Scenarios:
    1. FedAvg vs Static Attack         (baseline)
    2. Trust-Aware vs Static Attack    (defense baseline)
    3. Trust-Aware vs Adaptive Attack  (adaptive threat)
    4. Trust-Aware vs Delayed Attack   (delayed threat)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.cnn_mnist import get_model
from fl_core.client import FLClient
from trust.trust_score import TrustScoreManager
from attacks.adaptive_controller import AdaptiveAttackController
from fl_core.aggregation.trust_aware import TrustAwareAggregator, FedAvgAggregator
from evaluation.metrics import AttackSuccessRate
from evaluation.trust_evolution import TrustEvolutionAnalyzer
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

    client_datasets = []
    spc = len(train_ds) // num_clients

    for i in range(num_clients):
        subset = Subset(train_ds, list(range(i * spc, (i + 1) * spc)))
        client_datasets.append(DataLoader(subset, batch_size=32, shuffle=True))

    test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False)
    return client_datasets, test_loader


# ------------------------------------------------------------------
# Single scenario runner
# ------------------------------------------------------------------

def run_scenario(
    scenario_name: str,
    attack_type: str,
    use_trust_defense: bool,
    num_clients: int = 100,
    attack_rate: float = 0.2,
    num_rounds: int = 60,
    alpha: float = 0.9,
    tau: float = 0.3,
    poisoning_scale: float = 5.0
):
    """
    Run one experimental scenario.

    Args:
        scenario_name:     Human-readable name
        attack_type:       'static' | 'delayed' | 'adaptive'
        use_trust_defense: True = TrustAware, False = FedAvg
        num_clients:       Total number of FL clients
        attack_rate:       Fraction of malicious clients
        num_rounds:        Number of FL training rounds
        alpha:             Trust EMA decay factor
        tau:               Trust threshold
        poisoning_scale:   Gradient flip scale

    Returns:
        history: {'accuracy', 'loss', 'trust_benign', 'trust_malicious'}
    """
    print(f"\n{'='*65}")
    print(f"  Scenario: {scenario_name}")
    print(f"  Attack: {attack_type}, Defense: {'Trust-Aware' if use_trust_defense else 'FedAvg'}")
    print(f"{'='*65}")

    client_datasets, test_loader = load_data(num_clients)
    model = get_model()

    # Create clients
    num_malicious = int(num_clients * attack_rate)
    malicious_ids = set(np.random.choice(num_clients, num_malicious, replace=False))

    clients = [
        FLClient(i, get_model(), client_datasets[i], is_malicious=(i in malicious_ids))
        for i in range(num_clients)
    ]

    # Attack controllers
    attack_kwargs = dict(
        poisoning_scale=poisoning_scale,
        trust_threshold=0.7,
        knows_trust_mechanism=use_trust_defense,  # adaptive attacker is aware
        dormant_threshold=0.3
    )
    attack_controllers = {
        cid: AdaptiveAttackController(cid, attack_type=attack_type, **attack_kwargs)
        for cid in malicious_ids
    }

    # Defense
    if use_trust_defense:
        trust_manager = TrustScoreManager(num_clients, alpha=alpha, tau=tau, enable_decay=True)
        aggregator = TrustAwareAggregator(trust_manager, enable_filtering=True)
    else:
        trust_manager = None
        aggregator = FedAvgAggregator()

    history = {
        'accuracy': [],
        'loss': [],
        'trust_benign': [],
        'trust_malicious': [],
        'attack_rate_actual': []
    }

    for round_num in tqdm(range(num_rounds), desc=scenario_name, leave=False):
        selected_idx = np.random.choice(num_clients, 10, replace=False)
        selected_clients = [clients[i] for i in selected_idx]

        # Distribute model
        global_params = {n: p.data.clone() for n, p in model.named_parameters()}
        for c in selected_clients:
            c.set_parameters(global_params)

        updates, client_ids, metrics_list = [], [], []
        attacks_this_round = 0

        for client in selected_clients:
            cid = client.client_id
            client_ids.append(cid)

            if client.is_malicious:
                controller = attack_controllers[cid]
                est_trust = trust_manager.get_trust_score(cid) if trust_manager else 1.0
                should_attack = controller.should_attack(round_num, est_trust)
                clean_update, metrics = client.train()

                if should_attack:
                    updates.append(controller.poison_gradient(clean_update))
                    attacks_this_round += 1
                else:
                    updates.append(clean_update)
            else:
                update, metrics = client.train()
                updates.append(update)

            metrics_list.append(metrics)

        history['attack_rate_actual'].append(attacks_this_round / len(selected_clients))

        # Update trust
        if trust_manager:
            benign_selected = [cid for cid in client_ids if cid not in malicious_ids]
            if benign_selected:
                benign_ups = [updates[i] for i, cid in enumerate(client_ids)
                              if cid in benign_selected]
                ref_grad = {k: torch.mean(torch.stack([u[k] for u in benign_ups]), dim=0)
                            for k in benign_ups[0]}
            else:
                # Fallback: use mean of all updates
                ref_grad = {k: torch.mean(torch.stack([u[k] for u in updates]), dim=0)
                            for k in updates[0]}

            for i, cid in enumerate(client_ids):
                trust_manager.update_trust(cid, updates[i], ref_grad, metrics_list[i], round_num)

            trust_manager.apply_idle_decay(client_ids, round_num)

            # Record trust stats
            b_trust = [trust_manager.get_trust_score(c) for c in range(num_clients)
                       if c not in malicious_ids]
            m_trust = [trust_manager.get_trust_score(c) for c in malicious_ids]
            history['trust_benign'].append(float(np.mean(b_trust)))
            history['trust_malicious'].append(float(np.mean(m_trust)))

        # Aggregate
        agg_update = aggregator.aggregate(updates, client_ids, metrics_list)
        for name, param in model.named_parameters():
            param.data += agg_update[name]

        # Evaluate
        model.eval()
        correct, total, test_loss = 0, 0, 0.0
        with torch.no_grad():
            for data, target in test_loader:
                out = model(data)
                test_loss += torch.nn.functional.cross_entropy(out, target).item()
                correct += out.argmax(1).eq(target).sum().item()
                total += target.size(0)

        history['accuracy'].append(correct / total)
        history['loss'].append(test_loss / len(test_loader))

    final_acc = history['accuracy'][-1]
    print(f"  → Final Accuracy: {final_acc * 100:.2f}%")

    if trust_manager:
        sep = trust_manager.get_trust_separation(
            [c for c in range(num_clients) if c not in malicious_ids],
            list(malicious_ids)
        )
        print(f"  → Trust Separation: {sep['separation']:.3f} "
              f"(benign={sep['avg_benign']:.3f}, malicious={sep['avg_malicious']:.3f})")

    return history


# ------------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------------

def main():
    print("\n" + "=" * 65)
    print("EXPERIMENT 2: ADAPTIVE ATTACK vs TRUST-AWARE FL")
    print("=" * 65)

    NUM_ROUNDS = 60
    NUM_CLIENTS = 100
    ATTACK_RATE = 0.2

    scenarios = [
        ("FedAvg + Static",      "static",   False),
        ("TrustAware + Static",  "static",   True),
        ("TrustAware + Delayed", "delayed",  True),
        ("TrustAware + Adaptive","adaptive", True),
    ]

    results = {}
    for name, attack_type, use_defense in scenarios:
        torch.manual_seed(42)
        np.random.seed(42)
        history = run_scenario(
            name, attack_type, use_defense,
            num_clients=NUM_CLIENTS,
            attack_rate=ATTACK_RATE,
            num_rounds=NUM_ROUNDS
        )
        results[name] = history

    # ---- Plotting ----
    os.makedirs('results/figures', exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Accuracy
    for name, data in results.items():
        axes[0, 0].plot(data['accuracy'], label=name, linewidth=2)
    axes[0, 0].set(xlabel='Round', ylabel='Test Accuracy',
                   title='Accuracy: Adaptive vs Static Attack')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    # Loss
    for name, data in results.items():
        axes[0, 1].plot(data['loss'], label=name, linewidth=2)
    axes[0, 1].set(xlabel='Round', ylabel='Test Loss', title='Loss Evolution')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    # Trust evolution (only trust-aware scenarios)
    for name, data in results.items():
        if data['trust_benign']:
            axes[1, 0].plot(data['trust_benign'], label=f'{name} – benign', linewidth=2)
            axes[1, 0].plot(data['trust_malicious'], linestyle='--',
                            label=f'{name} – malicious', linewidth=1.5)
    axes[1, 0].axhline(y=0.3, color='k', linestyle=':', label='τ=0.3')
    axes[1, 0].set(xlabel='Round', ylabel='Trust Score', title='Trust Evolution')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1.05])

    # Actual attack rate per round
    for name, data in results.items():
        if data['attack_rate_actual']:
            axes[1, 1].plot(data['attack_rate_actual'], label=name, linewidth=1.5)
    axes[1, 1].set(xlabel='Round', ylabel='Attack Rate in Selected Clients',
                   title='Actual Attack Rate per Round')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/figures/exp_adaptive_attack.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure saved: results/figures/exp_adaptive_attack.png")

    # Summary table
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"{'Scenario':<30} {'Final Acc':<14} {'Min Acc':<12}")
    print("-" * 65)
    for name, data in results.items():
        acc = data['accuracy']
        print(f"{name:<30} {acc[-1]*100:<14.2f}% {min(acc)*100:<12.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()