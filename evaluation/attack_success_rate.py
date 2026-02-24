"""
Attack Success Rate Evaluation
Comprehensive evaluation of attack effectiveness and defense performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple


class AttackSuccessEvaluator:
    """
    Evaluates attack success rate across training.
    
    Tracks:
        - Model accuracy degradation
        - Attack effectiveness per round
        - Defense recovery behavior
        - Comparative analysis across defenses
    """

    def __init__(self, clean_accuracy: float, baseline_accuracy: float = 0.1):
        """
        Args:
            clean_accuracy:    Accuracy of clean (no attack) model
            baseline_accuracy: Random-guess accuracy (1/num_classes)
        """
        self.clean_accuracy = clean_accuracy
        self.baseline_accuracy = baseline_accuracy

        # Per-round tracking
        self.round_accuracies: List[float] = []
        self.round_asr: List[float] = []

    def compute_asr(
        self,
        poisoned_accuracy: float,
        clean_accuracy: Optional[float] = None,
        baseline_accuracy: Optional[float] = None
    ) -> float:
        """
        Compute Attack Success Rate (ASR).

        Definition:
            ASR = (clean_acc - poisoned_acc) / (clean_acc - baseline_acc)

        Interpretation:
            ASR = 0: Attack had no effect
            ASR = 1: Attack completely degraded model to random-guess level

        Args:
            poisoned_accuracy: Model accuracy under attack
            clean_accuracy:    Override instance's clean accuracy
            baseline_accuracy: Override instance's baseline accuracy

        Returns:
            asr: Attack success rate [0, 1]
        """
        clean = clean_accuracy if clean_accuracy is not None else self.clean_accuracy
        baseline = baseline_accuracy if baseline_accuracy is not None else self.baseline_accuracy

        if clean <= baseline:
            return 0.0

        asr = (clean - poisoned_accuracy) / (clean - baseline)
        return float(np.clip(asr, 0.0, 1.0))

    def record_round(self, round_num: int, accuracy: float):
        """Record accuracy for a round and compute ASR."""
        self.round_accuracies.append(accuracy)
        asr = self.compute_asr(accuracy)
        self.round_asr.append(asr)

    def get_summary(self) -> Dict[str, float]:
        """Return summary statistics."""
        if not self.round_accuracies:
            return {}

        return {
            'clean_accuracy': self.clean_accuracy,
            'final_accuracy': self.round_accuracies[-1],
            'min_accuracy': float(np.min(self.round_accuracies)),
            'max_accuracy': float(np.max(self.round_accuracies)),
            'mean_accuracy': float(np.mean(self.round_accuracies)),
            'final_asr': self.round_asr[-1] if self.round_asr else 0.0,
            'max_asr': float(np.max(self.round_asr)) if self.round_asr else 0.0,
            'mean_asr': float(np.mean(self.round_asr)) if self.round_asr else 0.0,
            'accuracy_drop': self.clean_accuracy - self.round_accuracies[-1],
            'num_rounds': len(self.round_accuracies)
        }


def compute_defense_effectiveness(
    clean_history: List[float],
    attacked_history: List[float],
    defended_history: List[float],
    baseline_accuracy: float = 0.1
) -> Dict[str, float]:
    """
    Compare three training runs:
        1. Clean (no attack, no defense)
        2. Attacked (attack, no defense = FedAvg)
        3. Defended (attack + defense)

    Args:
        clean_history:    Accuracy per round without attack
        attacked_history: Accuracy per round with attack, no defense
        defended_history: Accuracy per round with attack + defense
        baseline_accuracy: Random-guess accuracy

    Returns:
        metrics: Comprehensive effectiveness metrics
    """
    clean_final = clean_history[-1]
    attacked_final = attacked_history[-1]
    defended_final = defended_history[-1]

    # ASR for attacked (no defense)
    denom = max(1e-6, clean_final - baseline_accuracy)
    asr_undefended = float(np.clip((clean_final - attacked_final) / denom, 0, 1))
    asr_defended = float(np.clip((clean_final - defended_final) / denom, 0, 1))

    # ASR reduction
    asr_reduction = asr_undefended - asr_defended
    asr_reduction_pct = (asr_reduction / max(1e-6, asr_undefended)) * 100

    # Accuracy preservation
    accuracy_preserved = defended_final / max(1e-6, clean_final)

    # Convergence comparison (round to reach 90% of clean final)
    target = 0.9 * clean_final

    def rounds_to_target(history, tgt):
        for i, acc in enumerate(history):
            if acc >= tgt:
                return i
        return -1

    return {
        'clean_final': clean_final,
        'attacked_final': attacked_final,
        'defended_final': defended_final,
        'asr_undefended': asr_undefended,
        'asr_defended': asr_defended,
        'asr_reduction': asr_reduction,
        'asr_reduction_pct': asr_reduction_pct,
        'accuracy_drop_undefended': clean_final - attacked_final,
        'accuracy_drop_defended': clean_final - defended_final,
        'accuracy_preserved': accuracy_preserved,
        'rounds_to_target_clean': rounds_to_target(clean_history, target),
        'rounds_to_target_attacked': rounds_to_target(attacked_history, target),
        'rounds_to_target_defended': rounds_to_target(defended_history, target),
    }


def plot_asr_comparison(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
    title: str = "Attack Success Rate Comparison"
):
    """
    Plot ASR and accuracy comparison across multiple defenses.

    Args:
        results:   {defense_name: {'history': {'accuracy': [...], ...}, 'asr': float}}
        save_path: Path to save figure
        title:     Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))

    # Accuracy over rounds
    for (name, data), color in zip(results.items(), colors):
        acc = data.get('history', {}).get('accuracy', [])
        if acc:
            axes[0].plot(acc, label=name, linewidth=2, color=color)

    axes[0].set_xlabel('Round', fontsize=12)
    axes[0].set_ylabel('Test Accuracy', fontsize=12)
    axes[0].set_title('Accuracy Under Attack', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    # Final ASR bar chart
    names = list(results.keys())
    asrs = [results[n].get('attack_success_rate', 0) * 100 for n in names]
    accs = [results[n].get('final_accuracy', 0) * 100 for n in names]

    x = np.arange(len(names))
    width = 0.35

    bars1 = axes[1].bar(x - width / 2, accs, width, label='Final Accuracy (%)',
                        color='steelblue', alpha=0.8)
    bars2 = axes[1].bar(x + width / 2, asrs, width, label='ASR (%)',
                        color='tomato', alpha=0.8)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([n.upper() for n in names], rotation=15, ha='right')
    axes[1].set_ylabel('Percentage (%)', fontsize=12)
    axes[1].set_title('Final Accuracy vs ASR', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].set_ylim([0, 115])
    axes[1].grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2., h,
                     f'{h:.1f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2., h,
                     f'{h:.1f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def print_asr_table(results: Dict[str, Dict]):
    """Print formatted comparison table."""
    print("\n" + "=" * 75)
    print("ATTACK SUCCESS RATE ANALYSIS")
    print("=" * 75)
    print(f"{'Defense':<20} {'Final Acc':<14} {'ASR':<14} {'Acc Drop':<14} {'Improvement':<12}")
    print("-" * 75)

    baseline_asr = None
    for name, data in results.items():
        acc = data.get('final_accuracy', 0) * 100
        asr = data.get('attack_success_rate', 0) * 100
        drop = data.get('accuracy_drop', 0) * 100

        if baseline_asr is None:
            baseline_asr = asr
            improvement = "  —"
        else:
            improvement = f"{baseline_asr - asr:+.1f}%"

        print(f"{name.upper():<20} {acc:<14.2f}% {asr:<14.2f}% {drop:<14.2f}% {improvement:<12}")

    print("=" * 75)