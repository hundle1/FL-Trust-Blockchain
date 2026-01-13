"""
Experiment 1: Static Attack Baseline
Compare defenses against static poisoning attacks
"""

import sys
import os
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
    """Load and partition MNIST data"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data/mnist', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data/mnist', train=False, download=True, transform=transform)
    
    # Partition data to clients
    client_datasets = []
    indices = np.random.permutation(len(train_dataset))
    
    for i in range(num_clients):
        start = i * samples_per_client
        end = start + samples_per_client
        client_indices = indices[start:end]
        client_subset = Subset(train_dataset, client_indices)
        client_datasets.append(DataLoader(client_subset, batch_size=32, shuffle=True))
    
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return client_datasets, test_loader


def run_experiment(defense_name="fedavg", attack_rate=0.2, num_rounds=50):
    """Run single experiment"""
    print(f"\n{'='*60}")
    print(f"Running: {defense_name.upper()} vs Static Attack")
    print(f"Attack Rate: {attack_rate*100}%")
    print(f"{'='*60}\n")
    
    # Load data
    num_clients = 100
    client_datasets, test_loader = load_mnist_data(num_clients=num_clients)
    
    # Create model
    model = get_model()
    
    # Create clients
    clients = []
    num_malicious = int(num_clients * attack_rate)
    malicious_ids = np.random.choice(num_clients, num_malicious, replace=False)
    
    for i in range(num_clients):
        is_malicious = i in malicious_ids
        client = FLClient(
            client_id=i,
            model=get_model(),
            train_data=client_datasets[i],
            device="cpu",
            is_malicious=is_malicious
        )
        clients.append(client)
    
    # Create attack controllers for malicious clients
    attack_controllers = {}
    for cid in malicious_ids:
        controller = AdaptiveAttackController(
            client_id=cid,
            attack_type="static",  # Always attack
            poisoning_scale=5.0
        )
        attack_controllers[cid] = controller
    
    # Setup defense
    if defense_name == "trust_aware":
        trust_manager = TrustScoreManager(num_clients=num_clients, alpha=0.9, tau=0.3)
        aggregator = TrustAwareAggregator(trust_manager, enable_filtering=True)
    else:
        trust_manager = None
        if defense_name == "krum":
            aggregator = KrumAggregator(num_malicious=num_malicious)
        elif defense_name == "trimmed_mean":
            aggregator = TrimmedMeanAggregator(trim_ratio=0.1)
        else:  # fedavg
            aggregator = FedAvgAggregator()
    
    # Training
    history = {
        'accuracy': [],
        'loss': [],
        'attack_success': []
    }
    
    clients_per_round = 10
    
    for round_num in tqdm(range(num_rounds), desc=f"{defense_name}"):
        # Select clients
        selected_indices = np.random.choice(num_clients, clients_per_round, replace=False)
        selected_clients = [clients[i] for i in selected_indices]
        
        # Distribute global model
        global_params = {name: param.data.clone() for name, param in model.named_parameters()}
        for client in selected_clients:
            client.set_parameters(global_params)
        
        # Collect updates
        updates = []
        client_ids = []
        metrics_list = []
        
        for client in selected_clients:
            if client.is_malicious and client.client_id in attack_controllers:
                # Attacker: poison gradient
                controller = attack_controllers[client.client_id]
                should_attack = controller.should_attack(round_num)
                
                if should_attack:
                    # Get clean update first
                    clean_update, metrics = client.train()
                    # Poison it
                    poisoned_update = controller.poison_gradient(clean_update)
                    updates.append(poisoned_update)
                else:
                    # Behave normally
                    update, metrics = client.train()
                    updates.append(update)
            else:
                # Benign client
                update, metrics = client.train()
                updates.append(update)
            
            client_ids.append(client.client_id)
            metrics_list.append(metrics)
        
        # Update trust (if using trust-aware)
        if trust_manager is not None:
            # Compute reference gradient (from benign clients)
            benign_updates = [updates[i] for i, cid in enumerate(client_ids) 
                            if cid not in malicious_ids]
            if benign_updates:
                reference_gradient = {}
                for key in benign_updates[0].keys():
                    reference_gradient[key] = torch.mean(
                        torch.stack([u[key] for u in benign_updates]), dim=0
                    )
                
                # Update trust for all clients
                for i, cid in enumerate(client_ids):
                    trust_manager.update_trust(
                        cid, updates[i], reference_gradient, metrics_list[i]
                    )
        
        # Aggregate
        aggregated_update = aggregator.aggregate(updates, client_ids, metrics_list)
        
        # Apply update
        for name, param in model.named_parameters():
            param.data += aggregated_update[name]
        
        # Evaluate
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += torch.nn.functional.cross_entropy(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = correct / total
        avg_loss = test_loss / len(test_loader)
        
        history['accuracy'].append(accuracy)
        history['loss'].append(avg_loss)
        
        # Calculate attack success (accuracy drop)
        if round_num > 0:
            acc_drop = history['accuracy'][0] - accuracy
            history['attack_success'].append(max(0, acc_drop))
    
    # Calculate final metrics
    final_accuracy = history['accuracy'][-1]
    avg_attack_success = np.mean(history['attack_success']) if history['attack_success'] else 0
    
    print(f"\nResults for {defense_name}:")
    print(f"  Final Accuracy: {final_accuracy*100:.2f}%")
    print(f"  Avg Attack Success: {avg_attack_success*100:.2f}%")
    
    return history, final_accuracy, avg_attack_success


def main():
    """Main experiment"""
    # Load config
    with open('config/fl_config.yaml', 'r') as f:
        fl_config = yaml.safe_load(f)
    
    num_rounds = 50
    attack_rate = 0.2
    
    # Run experiments for different defenses
    defenses = ['fedavg', 'krum', 'trimmed_mean', 'trust_aware']
    
    results = {}
    for defense in defenses:
        history, final_acc, asr = run_experiment(defense, attack_rate, num_rounds)
        results[defense] = {
            'history': history,
            'final_accuracy': final_acc,
            'attack_success_rate': asr
        }
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Accuracy over rounds
    plt.subplot(1, 2, 1)
    for defense in defenses:
        plt.plot(results[defense]['history']['accuracy'], label=defense.upper())
    plt.xlabel('Round')
    plt.ylabel('Test Accuracy')
    plt.title('Accuracy vs Static Attack')
    plt.legend()
    plt.grid(True)
    
    # Final metrics comparison
    plt.subplot(1, 2, 2)
    defense_names = [d.upper() for d in defenses]
    accuracies = [results[d]['final_accuracy'] * 100 for d in defenses]
    
    plt.bar(defense_names, accuracies)
    plt.ylabel('Final Accuracy (%)')
    plt.title('Defense Comparison')
    plt.ylim([0, 100])
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/figures/exp_static_attack.png', dpi=300)
    print(f"\nFigure saved to: results/figures/exp_static_attack.png")
    
    # Print summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"{'Defense':<15} {'Final Acc (%)':<15} {'ASR (%)':<15}")
    print("-"*60)
    for defense in defenses:
        acc = results[defense]['final_accuracy'] * 100
        asr = results[defense]['attack_success_rate'] * 100
        print(f"{defense.upper():<15} {acc:<15.2f} {asr:<15.2f}")
    print("="*60)


if __name__ == "__main__":
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()