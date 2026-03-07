"""
Experiment: Old Methods Comparison
Compare FedAvg vs Krum vs Trimmed-Mean under STRONG gradient poisoning attacks.

Attack Setup (deliberately strong to expose weaknesses):
    - attack_rate:      30% malicious clients  (was 20%)
    - poisoning_scale:  15.0                   (was 5.0)
    - poison_method:    'lie' + 'gradient_flip' combo
    - clients_per_round: 10, so ~3 attackers/round — this BREAKS Krum's n>2f+2 assumption
    - 4 attack types tested: static, delayed, intermittent, adaptive

Scenarios:
    For each defense ∈ {FedAvg, Krum, TrimmedMean}:
        - No attack (clean baseline)
        - Static attack
        - Delayed attack
        - Adaptive attack

Outputs:
    - results/figures/exp_oldmethod_accuracy.png   — accuracy curves per scenario
    - results/figures/exp_oldmethod_heatmap.png    — final accuracy heatmap (defense × attack)
    - results/figures/exp_oldmethod_asr.png        — ASR bar chart
    - Console summary table
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from models.cnn_mnist import get_model
from fl_core.client import FLClient
from fl_core.aggregation.trust_aware import (
    FedAvgAggregator, KrumAggregator, TrimmedMeanAggregator
)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset



ATTACK_RATE       = 0.20      
POISONING_SCALE   = 5.0    
NUM_CLIENTS       = 100
CLIENTS_PER_ROUND = 20     
NUM_ROUNDS        = 80
SEED              = 42


ATTACK_TYPES = ["no_attack", "static", "delayed", "adaptive"]


DEFENSES = ["fedavg", "krum", "trimmed_mean"]

COLORS = {
    "fedavg":       "#e74c3c",
    "krum":         "#3498db",
    "trimmed_mean": "#2ecc71",
}

ATTACK_STYLES = {
    "no_attack": ("solid",   "○"),
    "static":    ("dashed",  "▲"),
    "delayed":   ("dotted",  "◆"),
    "adaptive":  ("dashdot", "●"),
}

def load_data(num_clients: int = NUM_CLIENTS):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST('./data/mnist', train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST('./data/mnist', train=False, download=True, transform=transform)

    spc = len(train_ds) // num_clients
    client_datasets = [
        DataLoader(Subset(train_ds, list(range(i * spc, (i + 1) * spc))),
                   batch_size=32, shuffle=True)
        for i in range(num_clients)
    ]
    test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False)
    return client_datasets, test_loader

class StrongAttacker:
    """
    Unified strong attacker supporting multiple strategies.
    Uses boosted gradient flip + optional LIE-style perturbation.
    """

    def __init__(
        self,
        client_id: int,
        attack_type: str = "static",
        poisoning_scale: float = POISONING_SCALE,
        delay_rounds: int = 10,
        trust_threshold: float = 0.55,
        dormant_threshold: float = 0.25,
        recovery_rounds: int = 5,
    ):
        self.client_id = client_id
        self.attack_type = attack_type
        self.poisoning_scale = poisoning_scale
        self.delay_rounds = delay_rounds
        self.trust_threshold = trust_threshold
        self.dormant_threshold = dormant_threshold
        self.recovery_rounds = recovery_rounds

        self.dormant = False
        self.dormant_since = -1
        self.total_attacks = 0
        self.total_rounds = 0

    def should_attack(self, round_num: int, estimated_trust: float = 1.0) -> bool:
        self.total_rounds += 1

        if self.attack_type == "no_attack":
            return False
        elif self.attack_type == "static":
            decision = True
        elif self.attack_type == "delayed":
            decision = round_num >= self.delay_rounds
        elif self.attack_type == "intermittent":
            decision = np.random.random() < 0.5
        elif self.attack_type == "adaptive":
            decision = self._adaptive(round_num, estimated_trust)
        else:
            decision = True

        if decision:
            self.total_attacks += 1
        return decision

    def _adaptive(self, round_num: int, trust: float) -> bool:
        if trust < self.dormant_threshold:
            if not self.dormant:
                self.dormant = True
                self.dormant_since = round_num
            return False
        if self.dormant:
            if round_num - self.dormant_since < self.recovery_rounds:
                return False
            self.dormant = False
        return trust > self.trust_threshold

    def poison_gradient(
        self,
        clean_gradient: Dict,
        all_updates=None,
    ) -> Dict:
        """
        Strong poisoning: boosted layer-wise flip.
        For Krum bypass: scale to match benign norm magnitude closely,
        then add adversarial direction.
        """
        param_names = list(clean_gradient.keys())
        n = len(param_names)

    
        if all_updates and len(all_updates) > 1:
            all_norms = []
            for u in all_updates:
                norm = float(np.sqrt(sum(torch.norm(p).item()**2 for p in u.values())))
                all_norms.append(norm)
            benign_norm = float(np.mean(all_norms))
        else:
            benign_norm = None

        poisoned = {}
        for i, name in enumerate(param_names):
           layer_boost = 1.0 + 1.5 * (i / max(1, n - 1))
           poisoned[name] = -self.poisoning_scale * layer_boost * clean_gradient[name]

        if benign_norm is not None:
            current_norm = float(np.sqrt(sum(torch.norm(p).item()**2 for p in poisoned.values())))
            if current_norm > 1e-6:
                target_norm = 3.0 * benign_norm
                scale = target_norm / current_norm
                poisoned = {name: scale * p for name, p in poisoned.items()}

        return poisoned


def run_scenario(
    defense_name: str,
    attack_type: str,
    client_datasets,
    test_loader,
    malicious_ids: set,
    num_rounds: int = NUM_ROUNDS,
) -> Dict:
    """
    Run one (defense, attack) scenario.

    Returns:
        history: {'accuracy': [...], 'loss': [...]}
    """
    model = get_model()
    num_malicious_est = int(NUM_CLIENTS * ATTACK_RATE)

    clients = [
        FLClient(i, get_model(), client_datasets[i], is_malicious=(i in malicious_ids))
        for i in range(NUM_CLIENTS)
    ]

    attackers = {
        cid: StrongAttacker(cid, attack_type=attack_type)
        for cid in malicious_ids
    }

    if defense_name == "krum":

        aggregator = KrumAggregator(num_malicious=num_malicious_est)
    elif defense_name == "trimmed_mean":
        aggregator = TrimmedMeanAggregator(trim_ratio=ATTACK_RATE) 
    else:
        aggregator = FedAvgAggregator()

    history = {'accuracy': [], 'loss': []}

    for round_num in tqdm(
        range(num_rounds),
        desc=f"{defense_name.upper():12s} | {attack_type:12s}",
        leave=False,
        ncols=80,
    ):
        selected_idx = np.random.choice(NUM_CLIENTS, CLIENTS_PER_ROUND, replace=False)
        selected = [clients[i] for i in selected_idx]


        global_params = {n: p.data.clone() for n, p in model.named_parameters()}
        for c in selected:
            c.set_parameters(global_params)

        raw_updates = []
        for client in selected:
            upd, _ = client.train()
            raw_updates.append(upd)

        updates, client_ids, metrics_list = [], [], []
        for idx, client in enumerate(selected):
            cid = client.client_id
            client_ids.append(cid)
            metrics_list.append({'loss': 0.1, 'accuracy': 0.9, 'gradient_norm': 1.0})

            if client.is_malicious and cid in attackers:
                attacker = attackers[cid]
                if attacker.should_attack(round_num):

                    benign_upds = [raw_updates[j] for j, c2 in enumerate(selected)
                                if c2.client_id not in malicious_ids]
                    poisoned = attacker.poison_gradient(
                        raw_updates[idx],
                        all_updates=benign_upds if benign_upds else None,
                    )
                    updates.append(poisoned)
                else:
                    updates.append(raw_updates[idx])
            else:
                updates.append(raw_updates[idx])

        agg_update = aggregator.aggregate(updates, client_ids, metrics_list)
        for name, param in model.named_parameters():
            param.data += agg_update[name]

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

    return history

def compute_asr(clean_acc: float, poisoned_acc: float, baseline: float = 0.1) -> float:
    if clean_acc <= baseline:
        return 0.0
    return float(np.clip((clean_acc - poisoned_acc) / (clean_acc - baseline), 0, 1))

def plot_accuracy_curves(results: Dict, clean_acc: Dict, save_dir: str):
    """
    One subplot per attack type, showing all 3 defenses.
    """
    attack_display = {
        "no_attack": "No Attack (Clean)",
        "static":    "Static Attack",
        "delayed":   "Delayed Attack",
        "adaptive":  "Adaptive Attack",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flatten()

    for ax_idx, attack_type in enumerate(ATTACK_TYPES):
        ax = axes[ax_idx]

        for defense in DEFENSES:
            acc = results[defense][attack_type]['accuracy']
            color = COLORS[defense]
            linestyle = 'solid' if attack_type == 'no_attack' else 'dashed'
            ax.plot(acc, label=defense.replace('_', ' ').title(),
                    color=color, linewidth=2.2, linestyle=linestyle)

        ax.set_title(attack_display[attack_type], fontsize=13, fontweight='bold')
        ax.set_xlabel('Round', fontsize=11)
        ax.set_ylabel('Test Accuracy', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

        for defense in DEFENSES:
            final = results[defense][attack_type]['accuracy'][-1]
            ax.annotate(
                f"{final*100:.1f}%",
                xy=(NUM_ROUNDS - 1, final),
                xytext=(-35, 8), textcoords='offset points',
                fontsize=8, color=COLORS[defense], fontweight='bold'
            )

    plt.suptitle(
        f"Old Methods Comparison Under Strong Attacks\n"
        f"(attack_rate={ATTACK_RATE*100:.0f}%, poisoning_scale={POISONING_SCALE}, "
        f"n={NUM_CLIENTS}, clients/round={CLIENTS_PER_ROUND})",
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    path = os.path.join(save_dir, 'exp_oldmethod_accuracy.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {path}")


def plot_heatmap(results: Dict, clean_accs: Dict, save_dir: str):
    """
    Heatmap: rows=defenses, cols=attack types, values=final accuracy %.
    """
    attack_labels = {
        "no_attack": "No\nAttack",
        "static":    "Static",
        "delayed":   "Delayed",
        "adaptive":  "Adaptive",
    }
    defense_labels = {
        "fedavg":       "FedAvg",
        "krum":         "Krum",
        "trimmed_mean": "Trimmed\nMean",
    }

    data = np.zeros((len(DEFENSES), len(ATTACK_TYPES)))
    for i, defense in enumerate(DEFENSES):
        for j, attack in enumerate(ATTACK_TYPES):
            data[i, j] = results[defense][attack]['accuracy'][-1] * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    sns.heatmap(
        data, ax=axes[0],
        xticklabels=[attack_labels[a] for a in ATTACK_TYPES],
        yticklabels=[defense_labels[d] for d in DEFENSES],
        annot=True, fmt='.1f', cmap='RdYlGn',
        vmin=10, vmax=100,
        cbar_kws={'label': 'Final Accuracy (%)'},
        linewidths=0.5
    )
    axes[0].set_title('Final Test Accuracy (%) by Defense × Attack', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Attack Type', fontsize=11)
    axes[0].set_ylabel('Defense', fontsize=11)

    asr_data = np.zeros_like(data)
    for i, defense in enumerate(DEFENSES):
        clean = clean_accs[defense]
        for j, attack in enumerate(ATTACK_TYPES):
            final = results[defense][attack]['accuracy'][-1]
            asr_data[i, j] = compute_asr(clean, final) * 100

    sns.heatmap(
        asr_data, ax=axes[1],
        xticklabels=[attack_labels[a] for a in ATTACK_TYPES],
        yticklabels=[defense_labels[d] for d in DEFENSES],
        annot=True, fmt='.1f', cmap='Reds',
        vmin=0, vmax=100,
        cbar_kws={'label': 'ASR (%)'},
        linewidths=0.5
    )
    axes[1].set_title('Attack Success Rate (%) by Defense × Attack', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Attack Type', fontsize=11)
    axes[1].set_ylabel('Defense', fontsize=11)

    plt.suptitle(
        f"Defense Comparison Heatmap  |  attack_rate={ATTACK_RATE*100:.0f}%  "
        f"poisoning_scale={POISONING_SCALE}",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    path = os.path.join(save_dir, 'exp_oldmethod_heatmap.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {path}")


def plot_asr_bars(results: Dict, clean_accs: Dict, save_dir: str):
    """
    Grouped bar chart: ASR per (defense, attack) pair.
    """
    attack_display = {
        "no_attack": "No Attack",
        "static":    "Static",
        "delayed":   "Delayed",
        "adaptive":  "Adaptive",
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    x = np.arange(len(ATTACK_TYPES))
    width = 0.25
    for i, defense in enumerate(DEFENSES):
        accs = [results[defense][at]['accuracy'][-1] * 100 for at in ATTACK_TYPES]
        bars = axes[0].bar(x + i * width, accs, width,
                           label=defense.replace('_', ' ').title(),
                           color=COLORS[defense], alpha=0.85, edgecolor='black', linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                         f'{h:.1f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels([attack_display[a] for a in ATTACK_TYPES], fontsize=11)
    axes[0].set_ylabel('Final Test Accuracy (%)', fontsize=12)
    axes[0].set_title('Final Accuracy by Defense & Attack', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].set_ylim([0, 115])
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].axhline(y=10, color='gray', linestyle=':', linewidth=1, label='Random guess (10%)')

    # --- Right: ASR bars ---
    for i, defense in enumerate(DEFENSES):
        clean = clean_accs[defense]
        asrs = [compute_asr(clean, results[defense][at]['accuracy'][-1]) * 100
                for at in ATTACK_TYPES]
        bars = axes[1].bar(x + i * width, asrs, width,
                           label=defense.replace('_', ' ').title(),
                           color=COLORS[defense], alpha=0.85, edgecolor='black', linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                         f'{h:.1f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels([attack_display[a] for a in ATTACK_TYPES], fontsize=11)
    axes[1].set_ylabel('Attack Success Rate (%)', fontsize=12)
    axes[1].set_title('ASR by Defense & Attack\n(Higher = Attack More Effective)',
                      fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].set_ylim([0, 115])
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle(
        f"Old Methods Benchmark  |  {ATTACK_RATE*100:.0f}% malicious, "
        f"scale={POISONING_SCALE}, {NUM_ROUNDS} rounds",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    path = os.path.join(save_dir, 'exp_oldmethod_asr.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {path}")


def plot_loss_curves(results: Dict, save_dir: str):
    """Loss curves for all scenarios."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax_idx, defense in enumerate(DEFENSES):
        ax = axes[ax_idx]
        linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
        markers = ['', '', '', '']
        for j, attack in enumerate(ATTACK_TYPES):
            loss = results[defense][attack]['loss']
            ax.plot(loss, linestyle=linestyles[j], linewidth=2,
                    label=attack.replace('_', ' ').title())
        ax.set_title(defense.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_xlabel('Round', fontsize=11)
        ax.set_ylabel('Test Loss', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Loss Evolution by Defense Method', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'exp_oldmethod_loss.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {path}")


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("\n" + "=" * 72)
    print("  OLD METHODS COMPARISON: FedAvg vs Krum vs Trimmed-Mean")
    print(f"  Attack Rate: {ATTACK_RATE*100:.0f}%  |  Poisoning Scale: {POISONING_SCALE}")
    print(f"  Clients: {NUM_CLIENTS}  |  Per Round: {CLIENTS_PER_ROUND}  |  Rounds: {NUM_ROUNDS}")
    print("=" * 72)

    save_dir = 'results/figures'
    os.makedirs(save_dir, exist_ok=True)

    # Fix malicious IDs across all runs for fair comparison
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    malicious_ids = set(np.random.choice(NUM_CLIENTS, int(NUM_CLIENTS * ATTACK_RATE), replace=False))
    print(f"\n  Malicious client IDs (first 10): {sorted(malicious_ids)[:10]} ...")

    # Load data once
    print("\n  Loading MNIST data ...")
    client_datasets, test_loader = load_data()

    # ---------------------------------------------------------------
    # Run all scenarios
    # ---------------------------------------------------------------
    results = {d: {} for d in DEFENSES}
    clean_accs = {}

    total_scenarios = len(DEFENSES) * len(ATTACK_TYPES)
    scenario_idx = 0

    for defense in DEFENSES:
        for attack_type in ATTACK_TYPES:
            scenario_idx += 1
            print(f"\n  [{scenario_idx}/{total_scenarios}] "
                  f"{defense.upper():12s} × {attack_type.upper():12s}")

            torch.manual_seed(SEED)
            np.random.seed(SEED)

            history = run_scenario(
                defense_name=defense,
                attack_type=attack_type,
                client_datasets=client_datasets,
                test_loader=test_loader,
                malicious_ids=malicious_ids,
                num_rounds=NUM_ROUNDS,
            )
            results[defense][attack_type] = history
            final_acc = history['accuracy'][-1]
            print(f"         → Final Accuracy: {final_acc*100:.2f}%")

        # Use no_attack as clean baseline for ASR computation
        clean_accs[defense] = results[defense]['no_attack']['accuracy'][-1]

    # ---------------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------------
    print("\n  Generating figures ...")
    plot_accuracy_curves(results, clean_accs, save_dir)
    plot_heatmap(results, clean_accs, save_dir)
    plot_asr_bars(results, clean_accs, save_dir)
    plot_loss_curves(results, save_dir)

    # ---------------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  FINAL ACCURACY SUMMARY (%)")
    print("=" * 72)
    header = f"  {'Defense':<16}" + "".join(f"{at.replace('_',' ').title():>14}" for at in ATTACK_TYPES)
    print(header)
    print("  " + "-" * 70)
    for defense in DEFENSES:
        row = f"  {defense.replace('_',' ').title():<16}"
        for attack in ATTACK_TYPES:
            acc = results[defense][attack]['accuracy'][-1] * 100
            row += f"{acc:>14.2f}%"
        print(row)

    print("\n" + "=" * 72)
    print("  ATTACK SUCCESS RATE SUMMARY (%)")
    print("=" * 72)
    print(f"  ASR = (clean_acc - poisoned_acc) / (clean_acc - 10%)  [0=no effect, 100=full degradation]")
    print(header)
    print("  " + "-" * 70)
    for defense in DEFENSES:
        row = f"  {defense.replace('_',' ').title():<16}"
        for attack in ATTACK_TYPES:
            final = results[defense][attack]['accuracy'][-1]
            asr = compute_asr(clean_accs[defense], final) * 100
            row += f"{asr:>14.2f}%"
        print(row)

    print("\n" + "=" * 72)
    print("  ACCURACY DROP vs CLEAN BASELINE (%)")
    print("=" * 72)
    print(header)
    print("  " + "-" * 70)
    for defense in DEFENSES:
        row = f"  {defense.replace('_',' ').title():<16}"
        for attack in ATTACK_TYPES:
            final = results[defense][attack]['accuracy'][-1] * 100
            clean = clean_accs[defense] * 100
            drop = clean - final
            sym = "↓" if drop > 0 else " "
            row += f"{sym}{drop:>12.2f}%"
        print(row)

    print("\n" + "=" * 72)
    print("  KEY OBSERVATIONS")
    print("=" * 72)

    # Find best/worst
    for attack in [at for at in ATTACK_TYPES if at != 'no_attack']:
        accs = {d: results[d][attack]['accuracy'][-1] for d in DEFENSES}
        best  = max(accs, key=accs.get)
        worst = min(accs, key=accs.get)
        print(f"  Under {attack:12s}: BEST={best.upper()} ({accs[best]*100:.1f}%)  "
              f"WORST={worst.upper()} ({accs[worst]*100:.1f}%)")

    print("\n  Figures saved to: ./results/figures/exp_oldmethod_*.png")
    print("=" * 72)


if __name__ == "__main__":
    main()