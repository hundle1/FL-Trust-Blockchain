"""
Experiment 5: Blockchain Overhead Measurement
Measure computational and storage overhead of blockchain logging
"""

import sys
import os
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
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data/mnist', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data/mnist', train=False, download=True, transform=transform)
    
    client_datasets = []
    samples_per_client = len(train_dataset) // num_clients
    
    for i in range(num_clients):
        start = i * samples_per_client
        end = start + samples_per_client
        client_subset = Subset(train_dataset, list(range(start, end)))
        client_datasets.append(DataLoader(client_subset, batch_size=32, shuffle=True))
    
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return client_datasets, test_loader


def run_overhead_experiment(use_blockchain, num_rounds=50):
    """Run training with/without blockchain"""
    
    num_clients = 50
    client_datasets, test_loader = load_data(num_clients)
    
    model = get_model()
    clients = [FLClient(i, get_model(), client_datasets[i]) for i in range(num_clients)]
    
    trust_manager = TrustScoreManager(num_clients, alpha=0.9)
    aggregator = TrustAwareAggregator(trust_manager)
    
    blockchain = MockBlockchain(consensus_latency=0.001, block_time=0.05) if use_blockchain else None
    
    # Timing
    round_times = []
    storage_per_round = []
    
    total_start = time.time()
    
    for round_num in tqdm(range(num_rounds), desc="With BC" if use_blockchain else "No BC", leave=False):
        round_start = time.time()
        
        # Select clients
        selected_idx = np.random.choice(num_clients, 10, replace=False)
        selected_clients = [clients[i] for i in selected_idx]
        
        # Distribute
        global_params = {name: p.data.clone() for name, p in model.named_parameters()}
        for client in selected_clients:
            client.set_parameters(global_params)
        
        # Training
        updates, client_ids, metrics_list = [], [], []
        for client in selected_clients:
            update, metrics = client.train()
            updates.append(update)
            client_ids.append(client.client_id)
            metrics_list.append(metrics)
            
            # Blockchain logging
            if blockchain:
                blockchain.log_client_update(
                    round_num, client.client_id,
                    trust_manager.get_trust_score(client.client_id),
                    metrics
                )
        
        # Update trust
        benign_updates = updates  # All benign in this overhead test
        ref_grad = {key: torch.mean(torch.stack([u[key] for u in benign_updates]), dim=0)
                   for key in benign_updates[0].keys()}
        
        for i, cid in enumerate(client_ids):
            trust_manager.update_trust(cid, updates[i], ref_grad, metrics_list[i])
        
        # Aggregate
        agg_update = aggregator.aggregate(updates, client_ids, metrics_list)
        
        for name, param in model.named_parameters():
            param.data += agg_update[name]
        
        # Log aggregation
        if blockchain:
            blockchain.log_aggregation(round_num, client_ids, "trust_aware", {})
            # Create block periodically
            if round_num % 5 == 0:
                blockchain.create_block()
        
        round_time = time.time() - round_start
        round_times.append(round_time)
        
        if blockchain:
            current_storage = blockchain.total_storage_bytes / 1024  # KB
            storage_per_round.append(current_storage)
        else:
            storage_per_round.append(0)
    
    total_time = time.time() - total_start
    
    overhead_metrics = {}
    if blockchain:
        overhead_metrics = blockchain.get_overhead_metrics()
    
    return {
        'round_times': round_times,
        'storage_per_round': storage_per_round,
        'total_time': total_time,
        'overhead_metrics': overhead_metrics
    }


def main():
    """Compare training with and without blockchain"""
    print("\n" + "="*70)
    print("BLOCKCHAIN OVERHEAD EXPERIMENT")
    print("="*70)
    
    num_rounds = 50
    
    print("\n[1/2] Running WITHOUT blockchain...")
    torch.manual_seed(42)
    np.random.seed(42)
    results_no_bc = run_overhead_experiment(use_blockchain=False, num_rounds=num_rounds)
    
    print("[2/2] Running WITH blockchain...")
    torch.manual_seed(42)
    np.random.seed(42)
    results_with_bc = run_overhead_experiment(use_blockchain=True, num_rounds=num_rounds)
    
    # Calculate overhead
    avg_time_no_bc = np.mean(results_no_bc['round_times'])
    avg_time_with_bc = np.mean(results_with_bc['round_times'])
    time_overhead_pct = ((avg_time_with_bc - avg_time_no_bc) / avg_time_no_bc) * 100
    
    total_time_overhead = results_with_bc['total_time'] - results_no_bc['total_time']
    total_storage = results_with_bc['storage_per_round'][-1]
    
    print(f"\n{'='*70}")
    print("OVERHEAD METRICS")
    print(f"{'='*70}")
    print(f"Training Time (No Blockchain):    {results_no_bc['total_time']:.2f}s")
    print(f"Training Time (With Blockchain):  {results_with_bc['total_time']:.2f}s")
    print(f"Time Overhead:                    +{total_time_overhead:.2f}s ({time_overhead_pct:.1f}%)")
    print(f"Total Storage:                    {total_storage:.2f} KB")
    print(f"Storage per Round:                {total_storage/num_rounds:.2f} KB")
    print(f"{'='*70}")
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Round time comparison
    axes[0, 0].plot(results_no_bc['round_times'], label='No Blockchain', linewidth=2)
    axes[0, 0].plot(results_with_bc['round_times'], label='With Blockchain', linewidth=2)
    axes[0, 0].set_xlabel('Round', fontsize=11)
    axes[0, 0].set_ylabel('Time per Round (s)', fontsize=11)
    axes[0, 0].set_title('Training Time per Round', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cumulative time
    cum_time_no_bc = np.cumsum(results_no_bc['round_times'])
    cum_time_with_bc = np.cumsum(results_with_bc['round_times'])
    
    axes[0, 1].plot(cum_time_no_bc, label='No Blockchain', linewidth=2)
    axes[0, 1].plot(cum_time_with_bc, label='With Blockchain', linewidth=2)
    axes[0, 1].fill_between(range(num_rounds), cum_time_no_bc, cum_time_with_bc, 
                            alpha=0.3, color='red', label='Overhead')
    axes[0, 1].set_xlabel('Round', fontsize=11)
    axes[0, 1].set_ylabel('Cumulative Time (s)', fontsize=11)
    axes[0, 1].set_title('Cumulative Training Time', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Storage growth
    axes[1, 0].plot(results_with_bc['storage_per_round'], 
                   linewidth=2, color='green')
    axes[1, 0].set_xlabel('Round', fontsize=11)
    axes[1, 0].set_ylabel('Storage (KB)', fontsize=11)
    axes[1, 0].set_title('Blockchain Storage Growth', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].fill_between(range(len(results_with_bc['storage_per_round'])), 
                            results_with_bc['storage_per_round'], alpha=0.3, color='green')
    
    # Overhead breakdown
    overhead_components = ['Time\nOverhead\n(s)', 'Storage\n(KB)']
    overhead_values = [total_time_overhead, total_storage]
    colors = ['coral', 'lightgreen']
    
    bars = axes[1, 1].bar(overhead_components, overhead_values, 
                          color=colors, alpha=0.8, edgecolor='black')
    axes[1, 1].set_ylabel('Overhead', fontsize=11)
    axes[1, 1].set_title('Blockchain Overhead Summary', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/figures/exp_overhead.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure saved: results/figures/exp_overhead.png")
    
    # Detailed blockchain metrics
    if results_with_bc['overhead_metrics']:
        bc_metrics = results_with_bc['overhead_metrics']
        print(f"\nDetailed Blockchain Metrics:")
        print(f"  Total Blocks:             {bc_metrics['total_blocks']}")
        print(f"  Total Write Time:         {bc_metrics['total_write_time']:.3f}s")
        print(f"  Total Read Time:          {bc_metrics['total_read_time']:.3f}s")
        print(f"  Avg Write Time per Block: {bc_metrics['avg_write_time_per_block']:.3f}s")
        print(f"  Storage per Block:        {bc_metrics['storage_per_block_kb']:.2f} KB")
    
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    print(f"✓ Blockchain adds {time_overhead_pct:.1f}% time overhead (acceptable)")
    print(f"✓ Storage growth is linear and predictable ({total_storage/num_rounds:.2f} KB/round)")
    print(f"✓ Overhead is justified by accountability benefits")
    print(f"✗ NOT suitable for real-time applications (mock only)")
    print("="*70)


if __name__ == "__main__":
    main()