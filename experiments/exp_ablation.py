"""
Experiment 4: Ablation Study
Isolates contribution of each system component.
Variants: 1) FedAvg 2) Trust Only 3) Trust+Decay 4) Full (Trust+Decay+Blockchain)
"""
import sys, os
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
from blockchain.ledger import MockBlockchain
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def load_data(num_clients=100):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST('./data/mnist', train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST('./data/mnist', train=False, download=True, transform=transform)
    spc = len(train_ds) // num_clients
    client_datasets = [
        DataLoader(Subset(train_ds, list(range(i*spc, (i+1)*spc))), batch_size=32, shuffle=True)
        for i in range(num_clients)
    ]
    return client_datasets, DataLoader(test_ds, batch_size=1000, shuffle=False)


def run_ablation(variant_name, use_trust, use_decay, use_blockchain, num_rounds=80):
    print(f"\n{'='*65}\nRunning: {variant_name}\n  Trust={use_trust} Decay={use_decay} Blockchain={use_blockchain}\n{'='*65}")

    num_clients, attack_rate = 100, 0.2
    client_datasets, test_loader = load_data(num_clients)
    model = get_model()

    malicious_ids = set(np.random.choice(num_clients, int(num_clients * attack_rate), replace=False))
    clients = [FLClient(i, get_model(), client_datasets[i], is_malicious=(i in malicious_ids))
               for i in range(num_clients)]

    attack_controllers = {
        cid: AdaptiveAttackController(cid, attack_type="adaptive", poisoning_scale=5.0,
                                      trust_threshold=0.7, knows_trust_mechanism=use_trust)
        for cid in malicious_ids
    }

    trust_manager = TrustScoreManager(num_clients, alpha=0.9, tau=0.3, enable_decay=use_decay) if use_trust else None
    aggregator = TrustAwareAggregator(trust_manager, enable_filtering=True) if use_trust else FedAvgAggregator()
    blockchain = MockBlockchain(consensus_latency=0.001, block_time=0.05) if use_blockchain else None

    history = {'accuracy': [], 'loss': []}

    for round_num in tqdm(range(num_rounds), desc=variant_name, leave=False):
        selected_idx = np.random.choice(num_clients, 10, replace=False)
        selected = [clients[i] for i in selected_idx]

        global_params = {n: p.data.clone() for n, p in model.named_parameters()}
        for c in selected:
            c.set_parameters(global_params)

        updates, client_ids, metrics_list = [], [], []
        for client in selected:
            est_trust = trust_manager.get_trust_score(client.client_id) if trust_manager else 1.0
            if client.is_malicious:
                ctrl = attack_controllers[client.client_id]
                if ctrl.should_attack(round_num, est_trust):
                    upd, met = client.train()
                    updates.append(ctrl.poison_gradient(upd))
                else:
                    upd, met = client.train()
                    updates.append(upd)
            else:
                upd, met = client.train()
                updates.append(upd)
            client_ids.append(client.client_id)
            metrics_list.append(met)
            if blockchain and trust_manager:
                blockchain.log_client_update(round_num, client.client_id, est_trust, met)

        if trust_manager:
            benign_upds = [updates[i] for i, cid in enumerate(client_ids) if cid not in malicious_ids]
            if benign_upds:
                ref = {k: torch.mean(torch.stack([u[k] for u in benign_upds]), dim=0) for k in benign_upds[0]}
                for i, cid in enumerate(client_ids):
                    trust_manager.update_trust(cid, updates[i], ref, metrics_list[i])

        agg = aggregator.aggregate(updates, client_ids, metrics_list)
        for name, param in model.named_parameters():
            param.data += agg[name]

        if blockchain:
            blockchain.log_aggregation(round_num, client_ids,
                                       "trust_aware" if use_trust else "fedavg",
                                       {})
            if round_num % 5 == 0:
                blockchain.create_block()

        model.eval()
        correct, test_loss = 0, 0.0
        with torch.no_grad():
            for data, target in test_loader:
                out = model(data)
                test_loss += torch.nn.functional.cross_entropy(out, target).item()
                correct += out.argmax(1).eq(target).sum().item()
        history['accuracy'].append(correct / len(test_loader.dataset))
        history['loss'].append(test_loss / len(test_loader))

    final_acc = history['accuracy'][-1] * 100
    overhead = blockchain.get_overhead_metrics() if blockchain else {}
    print(f"  Final Accuracy: {final_acc:.2f}%")
    return history, final_acc, overhead


def main():
    print("\n" + "="*65 + "\nABLATION STUDY\n" + "="*65)
    os.makedirs('results/figures', exist_ok=True)

    variants = [
        ("1. Baseline (FedAvg)",          False, False, False),
        ("2. Trust Only (no decay)",       True,  False, False),
        ("3. Trust + Decay",               True,  True,  False),
        ("4. Full System (+ Blockchain)",  True,  True,  True),
    ]

    results = {}
    for name, use_trust, use_decay, use_bc in variants:
        torch.manual_seed(42); np.random.seed(42)
        history, final_acc, overhead = run_ablation(name, use_trust, use_decay, use_bc, 80)
        results[name] = {'history': history, 'final_acc': final_acc, 'overhead': overhead}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for name in results:
        axes[0, 0].plot(results[name]['history']['accuracy'], label=name, linewidth=2)
    axes[0, 0].set(xlabel='Round', ylabel='Accuracy', title='Accuracy: Component Contribution')
    axes[0, 0].legend(fontsize=9); axes[0, 0].grid(True, alpha=0.3)

    for name in results:
        axes[0, 1].plot(results[name]['history']['loss'], label=name, linewidth=2)
    axes[0, 1].set(xlabel='Round', ylabel='Loss', title='Loss Evolution')
    axes[0, 1].legend(fontsize=9); axes[0, 1].grid(True, alpha=0.3)

    names = [n.split('.')[1].strip() for n in results]
    accs  = [results[n]['final_acc'] for n in results]
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
    bars = axes[1, 0].bar(range(len(names)), accs, color=colors, alpha=0.8, edgecolor='black')
    axes[1, 0].set_xticks(range(len(names)))
    axes[1, 0].set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    axes[1, 0].set(ylabel='Final Accuracy (%)', title='Component Impact')
    axes[1, 0].set_ylim([0, 100]); axes[1, 0].grid(True, alpha=0.3, axis='y')
    for bar in bars:
        h = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, h, f'{h:.1f}%',
                        ha='center', va='bottom', fontweight='bold', fontsize=9)

    full = "4. Full System (+ Blockchain)"
    if results[full]['overhead']:
        ov = results[full]['overhead']
        axes[1, 1].bar(['Write Time (s)', 'Storage (KB)'],
                       [ov['total_write_time'], ov['total_storage_kb']],
                       color='coral', alpha=0.8)
        axes[1, 1].set(title='Blockchain Overhead')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('results/figures/exp_ablation.png', dpi=300, bbox_inches='tight')
    print("\n✓ Figure saved: results/figures/exp_ablation.png")

    print("\n" + "="*65 + "\nABLATION SUMMARY\n" + "="*65)
    baseline = results["1. Baseline (FedAvg)"]['final_acc']
    for name in results:
        acc = results[name]['final_acc']
        print(f"  {name:<35}  {acc:.2f}%  ({acc-baseline:+.2f}% vs baseline)")


if __name__ == "__main__":
    main()