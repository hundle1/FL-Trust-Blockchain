"""
Convergence Analysis Module
Analyze training convergence speed and stability.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


class ConvergenceAnalyzer:
    """Analyze convergence properties of FL training."""
    
    @staticmethod
    def find_convergence_round(
        accuracy_history: List[float],
        threshold: float = 0.90,
        window: int = 5
    ) -> int:
        """
        First round after which accuracy stays ≥ threshold for `window` consecutive rounds.
        Returns -1 if never converges.
        """
        consecutive = 0
        for i, acc in enumerate(accuracy_history):
            if acc >= threshold:
                consecutive += 1
                if consecutive >= window:
                    return i - window + 1
            else:
                consecutive = 0
        return -1
    
    @staticmethod
    def calculate_convergence_speed(
        accuracy_history: List[float],
        target: float = 0.90
    ) -> Dict:
        conv = ConvergenceAnalyzer.find_convergence_round(accuracy_history, target)
        if conv == -1:
            return {'converged': False, 'rounds_to_converge': -1,
                    'convergence_speed': 0.0,
                    'final_accuracy': accuracy_history[-1] if accuracy_history else 0.0}
        return {
            'converged': True,
            'rounds_to_converge': conv,
            'convergence_speed': 1.0 / (conv + 1),
            'final_accuracy': accuracy_history[-1]
        }
    
    @staticmethod
    def calculate_stability(
        accuracy_history: List[float],
        window: Optional[int] = None
    ) -> Dict:
        if len(accuracy_history) < 10:
            return {'variance': 0.0, 'std': 0.0, 'stability_score': 1.0, 'oscillation_score': 0.0}
        
        window = window or max(10, len(accuracy_history) // 5)
        region = accuracy_history[-window:]
        
        variance = float(np.var(region))
        std = float(np.std(region))
        stability_score = float(1.0 / (1.0 + variance * 100))
        
        grads = np.diff(region)
        sign_changes = int(np.sum(np.diff(np.sign(grads)) != 0))
        oscillation_score = sign_changes / max(1, len(grads))
        
        return {
            'variance': variance, 'std': std,
            'stability_score': stability_score,
            'oscillation_score': oscillation_score,
            'mean': float(np.mean(region)),
            'min': float(np.min(region)),
            'max': float(np.max(region))
        }
    
    @staticmethod
    def compare_convergence(
        history1: List[float],
        history2: List[float],
        labels: Tuple[str, str] = ("Model 1", "Model 2")
    ) -> Dict:
        conv1 = ConvergenceAnalyzer.calculate_convergence_speed(history1)
        conv2 = ConvergenceAnalyzer.calculate_convergence_speed(history2)
        stab1 = ConvergenceAnalyzer.calculate_stability(history1)
        stab2 = ConvergenceAnalyzer.calculate_stability(history2)
        
        if conv1['converged'] and not conv2['converged']:
            faster = labels[0]
        elif conv2['converged'] and not conv1['converged']:
            faster = labels[1]
        elif conv1['converged'] and conv2['converged']:
            faster = labels[0] if conv1['rounds_to_converge'] <= conv2['rounds_to_converge'] else labels[1]
        else:
            faster = "Neither converged"
        
        more_stable = labels[0] if stab1['stability_score'] >= stab2['stability_score'] else labels[1]
        
        return {
            'convergence': {labels[0]: conv1, labels[1]: conv2, 'faster': faster},
            'stability': {labels[0]: stab1, labels[1]: stab2, 'more_stable': more_stable},
            'final_accuracy': {
                labels[0]: history1[-1] if history1 else 0.0,
                labels[1]: history2[-1] if history2 else 0.0
            }
        }
    
    @staticmethod
    def print_convergence_report(metrics: Dict, label: str = "Model"):
        print(f"\n{'='*60}")
        print(f"CONVERGENCE ANALYSIS: {label}")
        print(f"{'='*60}")
        if 'converged' in metrics:
            print(f"  Converged:          {'Yes' if metrics['converged'] else 'No'}")
            if metrics['converged']:
                print(f"  Rounds:             {metrics['rounds_to_converge']}")
            print(f"  Final Accuracy:     {metrics['final_accuracy']*100:.2f}%")
        if 'stability_score' in metrics:
            print(f"  Stability Score:    {metrics['stability_score']:.4f}")
            print(f"  Std Dev:            {metrics['std']:.4f}")
        print(f"{'='*60}")