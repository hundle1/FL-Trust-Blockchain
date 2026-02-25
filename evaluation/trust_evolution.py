"""
Trust Evolution Analysis and Visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import seaborn as sns


class TrustEvolutionAnalyzer:
    """Analyze and visualize trust score evolution over FL training."""
    
    @staticmethod
    def calculate_trust_statistics(trust_history: List[float]) -> Dict[str, float]:
        if not trust_history:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'trend': 0.0, 'final': 0.0}
        arr = np.array(trust_history)
        trend = float(np.polyfit(np.arange(len(arr)), arr, 1)[0]) if len(arr) > 1 else 0.0
        return {
            'mean': float(np.mean(arr)), 'std': float(np.std(arr)),
            'min': float(np.min(arr)), 'max': float(np.max(arr)),
            'trend': trend, 'final': float(trust_history[-1])
        }
    
    @staticmethod
    def calculate_separation(
        benign_history: List[List[float]],
        malicious_history: List[List[float]]
    ) -> Dict:
        benign_avg = np.mean(benign_history, axis=0)
        malicious_avg = np.mean(malicious_history, axis=0)
        sep_hist = (benign_avg - malicious_avg).tolist()
        return {
            'average_separation': float(np.mean(sep_hist)),
            'min_separation': float(np.min(sep_hist)),
            'final_separation': sep_hist[-1] if sep_hist else 0.0,
            'separation_history': sep_hist
        }
    
    @staticmethod
    def plot_trust_evolution(
        trust_histories: Dict,
        labels: Dict,
        threshold: Optional[float] = None,
        save_path: Optional[str] = None,
        title: str = "Trust Evolution"
    ):
        plt.figure(figsize=(12, 6))
        
        benign_hist, malicious_hist = [], []
        for cid, history in trust_histories.items():
            if labels.get(cid, 'benign') == 'benign':
                benign_hist.append(history)
            else:
                malicious_hist.append(history)
        
        if benign_hist:
            avg = np.mean(benign_hist, axis=0)
            std = np.std(benign_hist, axis=0)
            r = np.arange(len(avg))
            plt.plot(r, avg, 'b-', linewidth=2, label='Benign (avg)')
            plt.fill_between(r, avg - std, avg + std, alpha=0.2, color='blue')
        
        if malicious_hist:
            avg = np.mean(malicious_hist, axis=0)
            std = np.std(malicious_hist, axis=0)
            r = np.arange(len(avg))
            plt.plot(r, avg, 'r-', linewidth=2, label='Malicious (avg)')
            plt.fill_between(r, avg - std, avg + std, alpha=0.2, color='red')
        
        if threshold is not None:
            plt.axhline(y=threshold, color='k', linestyle='--',
                        linewidth=1.5, label=f'τ={threshold}')
        
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Trust Score', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.05])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.tight_layout(); plt.show()
        plt.close()
    
    @staticmethod
    def plot_trust_heatmap(
        trust_matrix: np.ndarray,
        client_labels: List[str],
        save_path: Optional[str] = None,
        title: str = "Trust Score Heatmap"
    ):
        plt.figure(figsize=(14, 8))
        sns.heatmap(trust_matrix, cmap='RdYlGn', vmin=0, vmax=1,
                    cbar_kws={'label': 'Trust Score'},
                    yticklabels=client_labels, xticklabels=10)
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Client', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.tight_layout(); plt.show()
        plt.close()
    
    @staticmethod
    def detect_trust_anomalies(
        trust_history: List[float],
        window: int = 5,
        threshold: float = 0.3
    ) -> List[int]:
        """Returns list of rounds where trust drops anomalously."""
        anomalies = []
        for i in range(window, len(trust_history)):
            if np.mean(trust_history[i-window:i]) - trust_history[i] > threshold:
                anomalies.append(i)
        return anomalies
    
    @staticmethod
    def print_trust_report(benign_stats, malicious_stats, separation):
        print("\n" + "=" * 70)
        print("TRUST EVOLUTION ANALYSIS")
        print("=" * 70)
        print(f"\n  [Benign]  mean={benign_stats['mean']:.3f}  "
              f"std={benign_stats['std']:.3f}  final={benign_stats['final']:.3f}  "
              f"trend={benign_stats['trend']:+.5f}")
        print(f"  [Malicious]  mean={malicious_stats['mean']:.3f}  "
              f"std={malicious_stats['std']:.3f}  final={malicious_stats['final']:.3f}  "
              f"trend={malicious_stats['trend']:+.5f}")
        print(f"\n  Separation (avg):   {separation['average_separation']:.3f}")
        print(f"  Separation (min):   {separation['min_separation']:.3f}")
        print(f"  Separation (final): {separation['final_separation']:.3f}")
        
        sep = separation['average_separation']
        if sep > 0.3:
            interpretation = "✓ Excellent"
        elif sep > 0.15:
            interpretation = "~ Moderate"
        else:
            interpretation = "✗ Poor"
        print(f"  Interpretation:     {interpretation} separation")
        print("=" * 70)