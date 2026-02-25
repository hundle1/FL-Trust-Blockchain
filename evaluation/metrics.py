"""
Evaluation Metrics
ASR, convergence, trust, and defense effectiveness metrics.
"""

import numpy as np
from typing import List, Dict, Optional


class AttackSuccessRate:
    """Attack Success Rate (ASR) Calculator."""
    
    @staticmethod
    def calculate_asr(
        clean_accuracy: float,
        poisoned_accuracy: float,
        baseline_accuracy: float = 0.1
    ) -> float:
        """
        ASR = (clean_acc - poisoned_acc) / (clean_acc - baseline)

        Returns:
            asr: [0, 1]  0=no effect, 1=degraded to random-guess
        """
        if clean_accuracy <= baseline_accuracy:
            return 0.0
        asr = (clean_accuracy - poisoned_accuracy) / (clean_accuracy - baseline_accuracy)
        return float(np.clip(asr, 0.0, 1.0))
    
    @staticmethod
    def calculate_accuracy_drop(clean_accuracy: float, poisoned_accuracy: float) -> float:
        return float(max(0.0, clean_accuracy - poisoned_accuracy))


class ConvergenceMetrics:
    """Convergence Analysis Metrics."""
    
    @staticmethod
    def convergence_round(accuracy_history: List[float], threshold: float = 0.90) -> int:
        """Round when model first reaches threshold (-1 if never)."""
        for i, acc in enumerate(accuracy_history):
            if acc >= threshold:
                return i
        return -1
    
    @staticmethod
    def convergence_speed(accuracy_history: List[float], target_accuracy: float = 0.90) -> float:
        """Inverse of rounds to convergence (higher = faster). 0 if never converges."""
        r = ConvergenceMetrics.convergence_round(accuracy_history, target_accuracy)
        return 0.0 if r == -1 else 1.0 / (r + 1)
    
    @staticmethod
    def stability_score(accuracy_history: List[float]) -> float:
        """Stability in last 20% of training. Returns [0, 1]."""
        if len(accuracy_history) < 10:
            return 0.0
        tail = accuracy_history[-max(1, len(accuracy_history) // 5):]
        return float(1.0 / (1.0 + np.var(tail) * 100))


class TrustMetrics:
    """Trust-specific metrics."""
    
    @staticmethod
    def trust_separation(benign_trust: List[float], malicious_trust: List[float]) -> float:
        """Average trust gap: benign_mean - malicious_mean (higher = better)."""
        if not benign_trust or not malicious_trust:
            return 0.0
        return float(max(0.0, np.mean(benign_trust) - np.mean(malicious_trust)))
    
    @staticmethod
    def trust_consistency(trust_history: List[float]) -> float:
        """Inverse of std dev (lower variance = higher consistency)."""
        if len(trust_history) < 2:
            return 1.0
        return float(1.0 / (1.0 + np.std(trust_history)))
    
    @staticmethod
    def detection_rate(malicious_trust: List[float], threshold: float = 0.3) -> float:
        """Fraction of malicious clients below trust threshold (= detected)."""
        if not malicious_trust:
            return 0.0
        return float(sum(1 for t in malicious_trust if t < threshold) / len(malicious_trust))


class DefenseEffectiveness:
    """Overall defense effectiveness."""
    
    @staticmethod
    def calculate_all_metrics(
        clean_acc: float,
        defended_acc: float,
        undefended_acc: float,
        benign_trust: Optional[List[float]] = None,
        malicious_trust: Optional[List[float]] = None
    ) -> Dict[str, float]:
        metrics = {}
        metrics['asr_undefended'] = AttackSuccessRate.calculate_asr(clean_acc, undefended_acc)
        metrics['asr_defended'] = AttackSuccessRate.calculate_asr(clean_acc, defended_acc)
        metrics['asr_reduction'] = metrics['asr_undefended'] - metrics['asr_defended']
        metrics['asr_reduction_pct'] = (
            metrics['asr_reduction'] / metrics['asr_undefended'] * 100
        ) if metrics['asr_undefended'] > 0 else 0.0
        metrics['accuracy_drop_undefended'] = clean_acc - undefended_acc
        metrics['accuracy_drop_defended'] = clean_acc - defended_acc
        metrics['accuracy_preserved'] = defended_acc / clean_acc if clean_acc > 0 else 0.0
        
        if benign_trust and malicious_trust:
            metrics['trust_separation'] = TrustMetrics.trust_separation(benign_trust, malicious_trust)
            metrics['detection_rate'] = TrustMetrics.detection_rate(malicious_trust)
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float]):
        print("\n" + "=" * 70)
        print("DEFENSE EFFECTIVENESS METRICS")
        print("=" * 70)
        print(f"\n  Undefended ASR:     {metrics.get('asr_undefended', 0)*100:.2f}%")
        print(f"  Defended ASR:       {metrics.get('asr_defended', 0)*100:.2f}%")
        print(f"  ASR Reduction:      {metrics.get('asr_reduction', 0)*100:.2f}% ({metrics.get('asr_reduction_pct', 0):.1f}% relative)")
        print(f"\n  Acc Drop (No Def):  {metrics.get('accuracy_drop_undefended', 0)*100:.2f}%")
        print(f"  Acc Drop (Def):     {metrics.get('accuracy_drop_defended', 0)*100:.2f}%")
        print(f"  Accuracy Preserved: {metrics.get('accuracy_preserved', 0)*100:.2f}%")
        if 'trust_separation' in metrics:
            print(f"\n  Trust Separation:   {metrics['trust_separation']:.3f}")
            print(f"  Detection Rate:     {metrics['detection_rate']*100:.2f}%")
        print("=" * 70)


def calculate_comparative_metrics(results: Dict[str, Dict]) -> None:
    print("\n" + "=" * 70)
    print("COMPARATIVE ANALYSIS")
    print("=" * 70)
    print(f"\n{'Defense':<20} {'Final Acc':<15} {'ASR':<15} {'Improvement':<15}")
    print("-" * 70)
    
    baseline_asr = None
    for defense, data in results.items():
        final_acc = data.get('final_accuracy', 0) * 100
        asr = data.get('attack_success_rate', 0) * 100
        if baseline_asr is None:
            baseline_asr = asr
            improvement = "—"
        else:
            improvement = f"-{baseline_asr - asr:.1f}%"
        print(f"{defense:<20} {final_acc:<15.2f}% {asr:<15.2f}% {improvement:<15}")
    print("=" * 70)