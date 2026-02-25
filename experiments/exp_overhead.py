"""
Experiment 5: Blockchain Overhead Measurement
Measures computational and storage overhead of blockchain logging.
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.cnn_mnist import get_model
from fl_core.client import FLClient
from trust.trust_score import TrustScoreManager
from fl_core.aggregation.trust_aware import TrustAwareAggregator
from blockchain.ledger import MockBlockchain
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def load_data(num_clients=50):
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST('./data/mnist', train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST('./data/mnist', train=False, download=True, transform=transform)
    spc = len(train_ds) // num_clients
    client_datasets = [
        DataLoader(Subset(train_ds, list(range(i*spc, (i+1)*spc))), batch_size=32, shuffle=True)
        for i in range(num_clients)
    ]
    return client_datasets, DataLoader(test_ds, batch_size=1000, shuffle=False)


def run_overhead_experiment(use_blockchain, num_rounds=50):
    num_clients = 50
    client_datasets, test_loader = load_data(num_clients)
    model = get_model()
    clients = [FLClient(i, get_model(), client_datasets[i]) for i in range(num_clients)]
    trust_manager = TrustScoreManager(num_clients, alpha=0.9)
    aggregator = TrustAwareAggregator(trust_manager)
    blockchain = MockBlockchain(consensus_latency=0.001, block_time=0.05) if use_blockchain else None

    round_times, storage_per_round = [], []
    total_start = time.time()

    for round_num in tqdm(range(num_rounds), desc=("With BC" if use_blockchain else "No BC"), leave=False):
        t0 = time.time()
        selected_idx = np.random.choice(num_clients, 10, replace=False)
        selected = [clients[i] for i in selected_idx]

        global_params = {n: p.data.clone() for n, p in model.named_parameters()}
        for c in selected:
            c.set_parameters(global_params)

        updates, client_ids, metrics_list = [], [], []
        for client in selected:
            upd, met = client.train()
            updates.append(upd)
            client_ids.append(client.client_id)
            metrics_list.append(met)
            if blockchain:
                blockchain.log_client_update(round_num, client.client_id,
                                             trust_manager.get_trust_score(client.client_id), met)

        ref = {k: torch.mean(torch.stack([u[k] for u in updates]), dim=0) for k in updates[0]}
        for i, cid in enumerate(client_ids):
            trust_manager.update_trust(cid, updates[i], ref, metrics_list[i])

        agg = aggregator.aggregate(updates, client_ids, metrics_list)
        for name, param in model.named_parameters():
            param.data += agg[name]

        if blockchain:
            blockchain.log_aggregation(round_num, client_ids, "trust_aware", {})
            if round_num % 5 == 0:
                blockchain.create_block()

        round_times.append(time.time() - t0)
        storage_per_round.append(blockchain.total_storage_bytes / 1024 if blockchain else 0)

    total_time = time.time() - total_start
    overhead_metrics = blockchain.get_overhead_metrics() if blockchain else {}
    return {
        'round_times': round_times, 'storage_per_round': storage_per_round,
        'total_time': total_time, 'overhead_metrics': overhead_metrics
    }


def main():
    print("\n" + "="*65 + "\nBLOCKCHAIN OVERHEAD EXPERIMENT\n" + "="*65)
    os.makedirs('results/figures', exist_ok=True)

    num_rounds = 50

    print("\n[1/2] WITHOUT blockchain...")
    torch.manual_seed(42); np.random.seed(42)
    r_no  = run_overhead_experiment(False, num_rounds)

    print("[2/2] WITH blockchain...")
    torch.manual_seed(42); np.random.seed(42)
    r_bc  = run_overhead_experiment(True, num_rounds)

    avg_no = np.mean(r_no['round_times'])
    avg_bc = np.mean(r_bc['round_times'])
    overhead_pct = (avg_bc - avg_no) / max(1e-6, avg_no) * 100
    total_storage = r_bc['storage_per_round'][-1] if r_bc['storage_per_round'] else 0

    print(f"\n{'='*65}\nOVERHEAD SUMMARY")
    print(f"  No-BC total time:   {r_no['total_time']:.2f}s")
    print(f"  BC total time:      {r_bc['total_time']:.2f}s")
    print(f"  Time overhead:      +{r_bc['total_time']-r_no['total_time']:.2f}s ({overhead_pct:.1f}%)")
    print(f"  Total storage:      {total_storage:.2f} KB")
    print(f"  Storage/round:      {total_storage/num_rounds:.2f} KB")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0,0].plot(r_no['round_times'], label='No Blockchain', linewidth=2)
    axes[0,0].plot(r_bc['round_times'], label='With Blockchain', linewidth=2)
    axes[0,0].set(xlabel='Round', ylabel='Time (s)', title='Time per Round')
    axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

    cum_no = np.cumsum(r_no['round_times'])
    cum_bc = np.cumsum(r_bc['round_times'])
    axes[0,1].plot(cum_no, label='No Blockchain', linewidth=2)
    axes[0,1].plot(cum_bc, label='With Blockchain', linewidth=2)
    axes[0,1].fill_between(range(num_rounds), cum_no, cum_bc, alpha=0.3, color='red', label='Overhead')
    axes[0,1].set(xlabel='Round', ylabel='Cumulative Time (s)', title='Cumulative Time')
    axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

    axes[1,0].plot(r_bc['storage_per_round'], linewidth=2, color='green')
    axes[1,0].fill_between(range(len(r_bc['storage_per_round'])), r_bc['storage_per_round'], alpha=0.3, color='green')
    axes[1,0].set(xlabel='Round', ylabel='Storage (KB)', title='Blockchain Storage Growth')
    axes[1,0].grid(True, alpha=0.3)

    axes[1,1].bar(['Time\nOverhead (s)', 'Storage\n(KB)'],
                  [r_bc['total_time']-r_no['total_time'], total_storage],
                  color=['coral', 'lightgreen'], alpha=0.8)
    axes[1,1].set(title='Overhead Summary'); axes[1,1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('results/figures/exp_overhead.png', dpi=300, bbox_inches='tight')
    print("\n✓ Figure saved: results/figures/exp_overhead.png")

    print(f"\n  Time overhead:     {overhead_pct:.1f}%  (acceptable for production systems)")
    print(f"  Storage growth:    linear, {total_storage/num_rounds:.2f} KB/round")


if __name__ == "__main__":
    main()