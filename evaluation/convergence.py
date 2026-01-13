"""
Convergence Analysis Module
Analyze training convergence speed and stability
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy import stats


class ConvergenceAnalyzer:
    """
    Analyze convergence properties of FL training
    """
    
    @staticmethod
    def find_convergence_round(
        accuracy_history: List[float],
        threshold: float = 0.90,
        window: int = 5
    ) -> int:
        """
        Find round when model converges to threshold
        
        Args:
            accuracy_history: Accuracy values per round
            threshold: Convergence threshold
            window: Number of consecutive rounds above threshold
            
        Returns:
            convergence_round: Round when converged (-1 if never)
        """
        consecutive_count = 0
        
        for i, acc in enumerate(accuracy_history):
            if acc >= threshold:
                consecutive_count += 1
                if consecutive_count >= window:
                    return i - window + 1
            else:
                consecutive_count = 0
        
        return -1
    
    @staticmethod
    def calculate_convergence_speed(
        accuracy_history: List[float],
        target: float = 0.90
    ) -> Dict[str, float]:
        """
        Calculate convergence speed metrics
        
        Returns:
            metrics: Speed metrics
        """
        conv_round = ConvergenceAnalyzer.find_convergence_round(accuracy_history, target)
        
        if conv_round == -1:
            return {
                'converged': False,
                'rounds_to_converge': -1,
                'convergence_speed': 0.0,
                'final_accuracy': accuracy_history[-1] if accuracy_history else 0.0
            }
        
        # Speed = 1 / rounds (higher = faster)
        speed = 1.0 / (conv_round + 1)
        
        return {
            'converged': True,
            'rounds_to_converge': conv_round,
            'convergence_speed': speed,
            'final_accuracy': accuracy_history[-1]
        }
    
    @staticmethod
    def calculate_stability(
        accuracy_history: List[float],
        window: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Calculate training stability
        
        Args:
            accuracy_history: Accuracy history
            window: Window for stability (None = use last 20%)
            
        Returns:
            stability_metrics: Variance, std, etc.
        """
        if len(accuracy_history) < 10:
            return {
                'variance': 0.0,
                'std': 0.0,
                'stability_score': 1.0,
                'oscillation_score': 0.0
            }
        
        # Use last portion of training
        if window is None:
            window = max(10, len(accuracy_history) // 5)
        
        stable_region = accuracy_history[-window:]
        
        variance = np.var(stable_region)
        std = np.std(stable_region)
        
        # Stability score (inverse of variance)
        stability_score = 1.0 / (1.0 + variance * 100)
        
        # Oscillation score (number of sign changes in gradient)
        gradients = np.diff(stable_region)
        sign_changes = np.sum(np.diff(np.sign(gradients)) != 0)
        oscillation_score = sign_changes / len(gradients) if len(gradients) > 0 else 0.0
        
        return {
            'variance': float(variance),
            'std': float(std),
            'stability_score': float(stability_score),
            'oscillation_score': float(oscillation_score),
            'mean': float(np.mean(stable_region)),
            'min': float(np.min(stable_region)),
            'max': float(np.max(stable_region))
        }
    
    @staticmethod
    def calculate_learning_curve_fit(
        accuracy_history: List[float]
    ) -> Dict[str, float]:
        """
        Fit learning curve to data
        
        Returns:
            fit_params: Parameters of fitted curve
        """
        if len(accuracy_history) < 10:
            return {'r_squared': 0.0, 'slope': 0.0, 'intercept': 0.0}
        
        rounds = np.arange(len(accuracy_history))
        acc = np.array(accuracy_history)
        
        # Fit logarithmic curve: acc = a * log(round + 1) + b
        log_rounds = np.log(rounds + 1)
        
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_rounds, acc)
            
            return {
                'slope': float(slope),
                'intercept': float(intercept),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value),
                'std_err': float(std_err)
            }
        except:
            return {'r_squared': 0.0, 'slope': 0.0, 'intercept': 0.0}
    
    @staticmethod
    def compare_convergence(
        history1: List[float],
        history2: List[float],
        labels: Tuple[str, str] = ("Model 1", "Model 2")
    ) -> Dict[str, any]:
        """
        Compare convergence of two training runs
        
        Returns:
            comparison: Comparison metrics
        """
        conv1 = ConvergenceAnalyzer.calculate_convergence_speed(history1)
        conv2 = ConvergenceAnalyzer.calculate_convergence_speed(history2)
        
        stab1 = ConvergenceAnalyzer.calculate_stability(history1)
        stab2 = ConvergenceAnalyzer.calculate_stability(history2)
        
        # Determine winner
        if conv1['converged'] and not conv2['converged']:
            faster = labels[0]
        elif conv2['converged'] and not conv1['converged']:
            faster = labels[1]
        elif conv1['converged'] and conv2['converged']:
            if conv1['rounds_to_converge'] < conv2['rounds_to_converge']:
                faster = labels[0]
            else:
                faster = labels[1]
        else:
            faster = "Neither converged"
        
        more_stable = labels[0] if stab1['stability_score'] > stab2['stability_score'] else labels[1]
        
        return {
            'convergence': {
                labels[0]: conv1,
                labels[1]: conv2,
                'faster': faster
            },
            'stability': {
                labels[0]: stab1,
                labels[1]: stab2,
                'more_stable': more_stable
            },
            'final_accuracy': {
                labels[0]: history1[-1] if history1 else 0.0,
                labels[1]: history2[-1] if history2 else 0.0
            }
        }
    
    @staticmethod
    def print_convergence_report(metrics: Dict[str, any], label: str = "Model"):
        """Print formatted convergence report"""
        print(f"\n{'='*70}")
        print(f"CONVERGENCE ANALYSIS: {label}")
        print(f"{'='*70}")
        
        if 'converged' in metrics:
            conv = metrics
            print(f"\n🎯 Convergence:")
            print(f"  Converged:              {'Yes' if conv['converged'] else 'No'}")
            if conv['converged']:
                print(f"  Rounds to Converge:     {conv['rounds_to_converge']}")
                print(f"  Convergence Speed:      {conv['convergence_speed']:.4f}")
            print(f"  Final Accuracy:         {conv['final_accuracy']*100:.2f}%")
        
        if 'variance' in metrics:
            stab = metrics
            print(f"\n📊 Stability:")
            print(f"  Variance:               {stab['variance']:.6f}")
            print(f"  Standard Deviation:     {stab['std']:.4f}")
            print(f"  Stability Score:        {stab['stability_score']:.4f}")
            print(f"  Oscillation Score:      {stab['oscillation_score']:.4f}")
        
        print(f"{'='*70}")