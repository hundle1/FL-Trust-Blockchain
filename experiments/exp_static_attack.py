"""
Experiment 1: Static Attack Baseline
Compare defenses against static poisoning attacks
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import yaml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.cnn_mnist import get_model
from fl_core.client import FLClient
from fl_core.server import FLServer
from trust.trust_score import TrustScoreManager
from attacks.adaptive_controller import AdaptiveAttackController
from fl_core.aggregation.trust_aware import (
    FedAvgAggregator, KrumAggregator, TrimmedMeanAggregator, TrustAwareAggregator
)
from evaluation.metrics import AttackSuccessRate, ConvergenceMetrics
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def load_mnist_data(num_clients=100, samples_per_client=600):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data/mnist', train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST('./data/mnist', train=False, download=True, transform=transform)

    indices = np.random.permutation(len(train_dataset))
    client_datasets = []
    for i in range(num_clients):
        s, e = i * samples_per_client, (i+1) * samples_per_client
        client_datasets.append(DataLoader(Subset(train_dataset, indices[s:e]),
                                          batch_size=32, shuffle=True))
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return client_datasets, test_loader


def run_experiment(defense_name="fedavg", attack_rate=0.2, num_rounds=50):
    print(f"\n{'='*60}\nRunning: {defense_name.upper()} vs Static Attack\n{'='*60}")

    num_clients = 100
    client_datasets, test_loader = load_mnist_data(num_clients)
    model = get_model()

    num_malicious = int(num_clients * attack_rate)
    malicious_ids = np.random.choice(num_clients, num_malicious, replace=False)

    clients = [
        FLClient(i, get_model(), client_datasets[i], device="cpu",
                 is_malicious=(i in malicious_ids))
        for i in range(num_clients)
    ]

    attack_controllers = {
        cid: AdaptiveAttackController(cid, attack_type="static", poisoning_scale=5.0)
        for cid in malicious_ids
    }

    if defense_name == "trust_aware":
        trust_manager = TrustScoreManager(num_clients, alpha=0.9, tau=0.3)
        aggregator = TrustAwareAggregator(trust_manager, enable_filtering=True)
    else:
        trust_manager = None
        aggregator = (KrumAggregator(num_malicious=num_malicious) if defense_name == "krum"
                      else TrimmedMeanAggregator(trim_ratio=0.1) if defense_name == "trimmed_mean"
                      else FedAvgAggregator())

    history = {'accuracy': [], 'loss': []}

    for round_num in tqdm(range(num_rounds), desc=defense_name):
        selected_idx = np.random.choice(num_clients, 10, replace=False)
        selected = [clients[i] for i in selected_idx]

        global_params = {n: p.data.clone() for n, p in model.named_parameters()}
        for c in selected:
            c.set_parameters(global_params)

        updates, client_ids, metrics_list = [], [], []
        for client in selected:
            if client.is_malicious and client.client_id in attack_controllers:
                ctrl = attack_controllers[client.client_id]
                if ctrl.should_attack(round_num):
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

        if trust_manager is not None:
            benign_upds = [updates[i] for i, cid in enumerate(client_ids)
                           if cid not in set(malicious_ids)]
            if benign_upds:
                ref = {k: torch.mean(torch.stack([u[k] for u in benign_upds]), dim=0)
                       for k in benign_upds[0]}
                for i, cid in enumerate(client_ids):
                    trust_manager.update_trust(cid, updates[i], ref, metrics_list[i])

        agg = aggregator.aggregate(updates, client_ids, metrics_list)
        for name, param in model.named_parameters():
            param.data += agg[name]

        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                out = model(data)
                test_loss += torch.nn.functional.cross_entropy(out, target).item()
                correct += out.argmax(1).eq(target).sum().item()
                total += target.size(0)

        history['accuracy'].append(correct / total)
        history['loss'].append(test_loss / len(test_loader))

    final_acc = history['accuracy'][-1]
    print(f"  Final Accuracy: {final_acc*100:.2f}%")
    return history, final_acc, 0.0


def main():
    os.makedirs('results/figures', exist_ok=True)
    
    # Load config if available, else use defaults
    try:
        with open('config/fl_config.yaml') as f:
            import yaml
            fl_config = yaml.safe_load(f)
        num_rounds = fl_config.get('fl', {}).get('num_rounds', 50)
    except Exception:
        num_rounds = 50

    attack_rate = 0.2
    defenses = ['fedavg', 'krum', 'trimmed_mean', 'trust_aware']
    results = {}

    for defense in defenses:
        torch.manual_seed(42); np.random.seed(42)
        history, final_acc, asr = run_experiment(defense, attack_rate, num_rounds)
        results[defense] = {'history': history, 'final_accuracy': final_acc, 'attack_success_rate': asr}

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for d in defenses:
        plt.plot(results[d]['history']['accuracy'], label=d.upper())
    plt.xlabel('Round'); plt.ylabel('Test Accuracy')
    plt.title('Accuracy vs Static Attack'); plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    accs = [results[d]['final_accuracy'] * 100 for d in defenses]
    plt.bar([d.upper() for d in defenses], accs)
    plt.ylabel('Final Accuracy (%)'); plt.title('Defense Comparison')
    plt.ylim([0, 100]); plt.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('results/figures/exp_static_attack.png', dpi=300)
    print("\nFigure saved to: results/figures/exp_static_attack.png")

    print("\n" + "="*60 + "\nSUMMARY\n" + "="*60)
    print(f"{'Defense':<15} {'Final Acc (%)':<15} {'ASR (%)':<15}")
    print("-" * 45)
    for d in defenses:
        print(f"{d.upper():<15} {results[d]['final_accuracy']*100:<15.2f} {results[d]['attack_success_rate']*100:<15.2f}")


if __name__ == "__main__":
    torch.manual_seed(42); np.random.seed(42)
    main()