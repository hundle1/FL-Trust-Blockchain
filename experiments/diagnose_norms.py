"""
Diagnostic: đo actual gradient norm của benign clients trên MNIST
Chạy script này trước để biết threshold nên set bao nhiêu.

Usage: python experiments/diagnose_norms.py
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from models.cnn_mnist import get_model
from fl_core.client import FLClient

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLIENTS    = 100
POISONING_SCALE = 5.0

def load_data(num_clients=NUM_CLIENTS):
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, '..', 'data')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    spc = len(train_ds) // num_clients
    return [
        DataLoader(Subset(train_ds, list(range(i*spc, (i+1)*spc))),
                   batch_size=32, shuffle=True)
        for i in range(num_clients)
    ]

def compute_norm(update):
    return float(np.sqrt(sum(torch.norm(v).item()**2 for v in update.values())))

def main():
    print("=" * 60)
    print("DIAGNOSTIC: Measuring actual gradient norms")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    client_datasets = load_data()
    model = get_model().to(DEVICE)

    # Sample 30 clients, train 1 round, measure norms
    benign_norms = []
    sampled_ids = np.random.choice(NUM_CLIENTS, 30, replace=False)

    print(f"\nMeasuring benign norms ({len(sampled_ids)} clients)...")
    global_params = {n: p.data.clone() for n, p in model.named_parameters()}

    for cid in sampled_ids:
        client = FLClient(cid, get_model().to(DEVICE), client_datasets[cid], device=DEVICE)
        client.set_parameters(global_params)
        update, _ = client.train()
        norm = compute_norm(update)
        benign_norms.append(norm)

    benign_norms = np.array(benign_norms)

    print(f"\nBENIGN gradient norms:")
    print(f"  min    = {benign_norms.min():.4f}")
    print(f"  max    = {benign_norms.max():.4f}")
    print(f"  mean   = {benign_norms.mean():.4f}")
    print(f"  median = {np.median(benign_norms):.4f}")
    print(f"  p75    = {np.percentile(benign_norms, 75):.4f}")
    print(f"  p90    = {np.percentile(benign_norms, 90):.4f}")
    print(f"  p95    = {np.percentile(benign_norms, 95):.4f}")
    print(f"  std    = {benign_norms.std():.4f}")

    # Simulate static attacker norms
    print(f"\nSimulating STATIC ATTACKER norms (scale={POISONING_SCALE})...")
    attacker_norms = []
    for cid in sampled_ids[:10]:
        client = FLClient(cid, get_model().to(DEVICE), client_datasets[cid], device=DEVICE)
        client.set_parameters(global_params)
        update, _ = client.train()
        # Apply static attack: flip + layer boost
        param_names = list(update.keys())
        n = len(param_names)
        poisoned = {}
        for i, name in enumerate(param_names):
            layer_boost = 1.0 + 1.5 * (i / max(1, n-1))
            poisoned[name] = -POISONING_SCALE * layer_boost * update[name]
        attacker_norms.append(compute_norm(poisoned))

    attacker_norms = np.array(attacker_norms)
    print(f"\nATTACKER gradient norms:")
    print(f"  min    = {attacker_norms.min():.4f}")
    print(f"  max    = {attacker_norms.max():.4f}")
    print(f"  mean   = {attacker_norms.mean():.4f}")
    print(f"  median = {np.median(attacker_norms):.4f}")

    ratio = attacker_norms.mean() / benign_norms.mean()
    print(f"\nAttacker/Benign norm ratio: {ratio:.2f}x")
    print(f"\nRECOMMENDED CONFIG:")
    safe_threshold = np.percentile(benign_norms, 97) * 1.5
    good_clip      = np.median(benign_norms) * 2.0
    print(f"  absolute_norm_threshold = {safe_threshold:.1f}  "
          f"(benign p97={np.percentile(benign_norms,97):.2f} × 1.5)")
    print(f"  clip_multiplier         = {good_clip/np.median(benign_norms):.1f}  "
          f"(median={np.median(benign_norms):.2f} × 2.0)")
    print(f"  norm_penalty_threshold  = {ratio*0.4:.1f}  "
          f"(attacker ratio {ratio:.1f}x → penalty starts at {ratio*0.4:.1f}x benign)")
    print("=" * 60)

if __name__ == "__main__":
    main()