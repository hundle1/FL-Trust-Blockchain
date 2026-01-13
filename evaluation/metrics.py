"""
Evaluation Metrics
Calculate ASR, convergence, and other metrics
"""

import numpy as np
from typing import List, Dict


class AttackSuccessRate:
    """
    Attack Success Rate (ASR) Calculator
    Measures how much attack degrades model performance
    """
    
    @staticmethod
    def calculate_asr(
        clean_accuracy: float,
        poisoned_accuracy: float,
        baseline_accuracy: float = 1.0
    ) -> float:
        """
        Calculate Attack Success Rate
        
        ASR = (clean_acc - poisoned_acc) / (clean_acc - baseline)
        
        Args:
            clean_accuracy: Accuracy without attack
            poisoned_accuracy: Accuracy with attack
            baseline_accuracy: Random guess accuracy (1/num_classes)
            
        Returns:
            asr: Attack success rate [0, 1]
        """
        if clean_accuracy <= baseline_accuracy:
            return 0.0
        
        asr = max(0, (clean_accuracy - poisoned_accuracy) / (clean_accuracy - baseline_accuracy))
        return min(1.0, asr)
    
    @staticmethod
    def calculate_accuracy_drop(
        clean_accuracy: float,
        poisoned_accuracy: float
    ) -> float:
        """Calculate absolute accuracy drop"""
        return max(0, clean_accuracy - poisoned_accuracy)


class ConvergenceMetrics:
    """
    Convergence Analysis Metrics
    """
    
    @staticmethod
    def convergence_round(
        accuracy_history: List[float],
        threshold: float = 0.90
    ) -> int:
        """
        Find round when model reaches convergence threshold
        
        Args:
            accuracy_history: List of accuracy values per round
            threshold: Convergence threshold (e.g., 90% accuracy)
            
        Returns:
            round_num: Round when threshold is reached (-1 if never)
        """
        for i, acc in enumerate(accuracy_history):
            if acc >= threshold:
                return i
        return -1
    
    @staticmethod
    def convergence_speed(
        accuracy_history: List[float],
        target_accuracy: float = 0.90
    ) -> float:
        """
        Calculate convergence speed (rounds needed to reach target)
        
        Returns:
            speed: Inverse of rounds needed (higher = faster)
        """
        conv_round = ConvergenceMetrics.convergence_round(accuracy_history, target_accuracy)
        
        if conv_round == -1:
            return 0.0
        
        return 1.0 / (conv_round + 1)
    
    @staticmethod
    def stability_score(accuracy_history: List[float]) -> float:
        """
        Calculate stability (inverse of variance in last 20% of training)
        
        Returns:
            stability: Stability score [0, 1]
        """
        if len(accuracy_history) < 10:
            return 0.0
        
        # Look at last 20% of rounds
        last_portion = accuracy_history[-len(accuracy_history)//5:]
        variance = np.var(last_portion)
        
        # Convert to stability score
        stability = 1.0 / (1.0 + variance * 100)
        return stability


class TrustMetrics:
    """
    Trust-specific Metrics
    """
    
    @staticmethod
    def trust_separation(
        benign_trust: List[float],
        malicious_trust: List[float]
    ) -> float:
        """
        Calculate separation between benign and malicious trust scores
        
        Returns:
            separation: Average difference (higher = better separation)
        """
        if not benign_trust or not malicious_trust:
            return 0.0
        
        avg_benign = np.mean(benign_trust)
        avg_malicious = np.mean(malicious_trust)
        
        return max(0, avg_benign - avg_malicious)
    
    @staticmethod
    def trust_consistency(trust_history: List[float]) -> float:
        """
        Calculate consistency of trust scores (lower variance = better)
        
        Returns:
            consistency: Inverse of standard deviation
        """
        if len(trust_history) < 2:
            return 1.0
        
        std = np.std(trust_history)
        consistency = 1.0 / (1.0 + std)
        return consistency
    
    @staticmethod
    def detection_rate(
        malicious_trust: List[float],
        threshold: float = 0.3
    ) -> float:
        """
        Calculate what fraction of malicious clients fall below threshold
        
        Returns:
            detection_rate: Fraction detected [0, 1]
        """
        if not malicious_trust:
            return 0.0
        
        below_threshold = sum(1 for t in malicious_trust if t < threshold)
        return below_threshold / len(malicious_trust)


class DefenseEffectiveness:
    """
    Overall Defense Effectiveness Metrics
    """
    
    @staticmethod
    def calculate_all_metrics(
        clean_acc: float,
        defended_acc: float,
        undefended_acc: float,
        benign_trust: List[float] = None,
        malicious_trust: List[float] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive defense metrics
        
        Returns:
            metrics: Dictionary of all metrics
        """
        metrics = {}
        
        # ASR metrics
        metrics['asr_undefended'] = AttackSuccessRate.calculate_asr(clean_acc, undefended_acc)
        metrics['asr_defended'] = AttackSuccessRate.calculate_asr(clean_acc, defended_acc)
        metrics['asr_reduction'] = metrics['asr_undefended'] - metrics['asr_defended']
        metrics['asr_reduction_pct'] = (metrics['asr_reduction'] / metrics['asr_undefended'] * 100) \
            if metrics['asr_undefended'] > 0 else 0
        
        # Accuracy metrics
        metrics['accuracy_drop_undefended'] = clean_acc - undefended_acc
        metrics['accuracy_drop_defended'] = clean_acc - defended_acc
        metrics['accuracy_preserved'] = defended_acc / clean_acc if clean_acc > 0 else 0
        
        # Trust metrics (if available)
        if benign_trust and malicious_trust:
            metrics['trust_separation'] = TrustMetrics.trust_separation(benign_trust, malicious_trust)
            metrics['detection_rate'] = TrustMetrics.detection_rate(malicious_trust)
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float]):
        """Pretty print metrics"""
        print("\n" + "="*70)
        print("DEFENSE EFFECTIVENESS METRICS")
        print("="*70)
        
        print("\n📊 Attack Success Rate (ASR):")
        print(f"  Undefended ASR:     {metrics.get('asr_undefended', 0)*100:.2f}%")
        print(f"  Defended ASR:       {metrics.get('asr_defended', 0)*100:.2f}%")
        print(f"  ASR Reduction:      {metrics.get('asr_reduction', 0)*100:.2f}% " + 
              f"({metrics.get('asr_reduction_pct', 0):.1f}% relative reduction)")
        
        print("\n🎯 Accuracy Metrics:")
        print(f"  Acc Drop (No Def):  {metrics.get('accuracy_drop_undefended', 0)*100:.2f}%")
        print(f"  Acc Drop (Def):     {metrics.get('accuracy_drop_defended', 0)*100:.2f}%")
        print(f"  Accuracy Preserved: {metrics.get('accuracy_preserved', 0)*100:.2f}%")
        
        if 'trust_separation' in metrics:
            print("\n🔒 Trust Metrics:")
            print(f"  Trust Separation:   {metrics['trust_separation']:.3f}")
            print(f"  Detection Rate:     {metrics['detection_rate']*100:.2f}%")
        
        print("="*70)


def calculate_comparative_metrics(results: Dict[str, Dict]) -> None:
    """
    Calculate and display comparative metrics across multiple defenses
    
    Args:
        results: Dictionary mapping defense name to results dict
    """
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS")
    print("="*70)
    
    defense_names = list(results.keys())
    
    print(f"\n{'Defense':<20} {'Final Acc':<15} {'ASR':<15} {'Improvement':<15}")
    print("-"*70)
    
    baseline_asr = None
    
    for defense in defense_names:
        final_acc = results[defense].get('final_accuracy', 0) * 100
        asr = results[defense].get('attack_success_rate', 0) * 100
        
        if baseline_asr is None:
            baseline_asr = asr
            improvement = "—"
        else:
            improvement = f"-{baseline_asr - asr:.1f}%"
        
        print(f"{defense:<20} {final_acc:<15.2f}% {asr:<15.2f}% {improvement:<15}")
    
    print("="*70)