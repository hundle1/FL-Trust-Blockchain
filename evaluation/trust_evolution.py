"""
Trust Evolution Analysis and Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import seaborn as sns


class TrustEvolutionAnalyzer:
    """
    Analyze and visualize trust score evolution over training
    """
    
    @staticmethod
    def calculate_trust_statistics(
        trust_history: List[float]
    ) -> Dict[str, float]:
        """
        Calculate statistics for trust evolution
        
        Returns:
            stats: Trust statistics
        """
        if not trust_history:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'trend': 0.0
            }
        
        trust_array = np.array(trust_history)
        
        # Trend (linear regression slope)
        rounds = np.arange(len(trust_history))
        if len(rounds) > 1:
            trend = np.polyfit(rounds, trust_array, 1)[0]
        else:
            trend = 0.0
        
        return {
            'mean': float(np.mean(trust_array)),
            'std': float(np.std(trust_array)),
            'min': float(np.min(trust_array)),
            'max': float(np.max(trust_array)),
            'trend': float(trend),
            'final': float(trust_history[-1])
        }
    
    @staticmethod
    def calculate_separation(
        benign_history: List[List[float]],
        malicious_history: List[List[float]]
    ) -> Dict[str, float]:
        """
        Calculate separation between benign and malicious trust
        
        Args:
            benign_history: List of trust histories for benign clients
            malicious_history: List of trust histories for malicious clients
            
        Returns:
            separation_metrics: Separation statistics
        """
        # Average trust per round
        benign_avg = np.mean([h for h in benign_history], axis=0)
        malicious_avg = np.mean([h for h in malicious_history], axis=0)
        
        # Overall separation
        separation = float(np.mean(benign_avg - malicious_avg))
        
        # Separation over time
        separation_history = (benign_avg - malicious_avg).tolist()
        
        # Minimum separation (worst case)
        min_separation = float(np.min(benign_avg - malicious_avg))
        
        return {
            'average_separation': separation,
            'min_separation': min_separation,
            'final_separation': separation_history[-1] if separation_history else 0.0,
            'separation_history': separation_history
        }
    
    @staticmethod
    def plot_trust_evolution(
        trust_histories: Dict[str, List[float]],
        labels: Dict[str, str],
        threshold: Optional[float] = None,
        save_path: Optional[str] = None,
        title: str = "Trust Evolution"
    ):
        """
        Plot trust evolution for multiple clients/groups
        
        Args:
            trust_histories: Dict mapping client_id to trust history
            labels: Dict mapping client_id to label (benign/malicious)
            threshold: Trust threshold to plot
            save_path: Path to save figure
            title: Plot title
        """
        plt.figure(figsize=(12, 6))
        
        # Group by label
        benign_histories = []
        malicious_histories = []
        
        for client_id, history in trust_histories.items():
            if labels.get(client_id, 'benign') == 'benign':
                benign_histories.append(history)
            else:
                malicious_histories.append(history)
        
        # Plot benign clients
        if benign_histories:
            benign_avg = np.mean(benign_histories, axis=0)
            benign_std = np.std(benign_histories, axis=0)
            rounds = np.arange(len(benign_avg))
            
            plt.plot(rounds, benign_avg, 'b-', linewidth=2, label='Benign (avg)')
            plt.fill_between(rounds, 
                           benign_avg - benign_std, 
                           benign_avg + benign_std,
                           alpha=0.2, color='blue')
        
        # Plot malicious clients
        if malicious_histories:
            malicious_avg = np.mean(malicious_histories, axis=0)
            malicious_std = np.std(malicious_histories, axis=0)
            rounds = np.arange(len(malicious_avg))
            
            plt.plot(rounds, malicious_avg, 'r-', linewidth=2, label='Malicious (avg)')
            plt.fill_between(rounds,
                           malicious_avg - malicious_std,
                           malicious_avg + malicious_std,
                           alpha=0.2, color='red')
        
        # Plot threshold
        if threshold is not None:
            plt.axhline(y=threshold, color='k', linestyle='--', 
                       linewidth=1.5, label=f'Threshold (τ={threshold})')
        
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Trust Score', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.05])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.tight_layout()
            plt.show()
        
        plt.close()
    
    @staticmethod
    def plot_trust_heatmap(
        trust_matrix: np.ndarray,
        client_labels: List[str],
        save_path: Optional[str] = None,
        title: str = "Trust Score Heatmap"
    ):
        """
        Plot heatmap of trust scores over time
        
        Args:
            trust_matrix: Matrix of shape (num_clients, num_rounds)
            client_labels: Labels for each client
            save_path: Path to save figure
            title: Plot title
        """
        plt.figure(figsize=(14, 8))
        
        sns.heatmap(trust_matrix, 
                   cmap='RdYlGn',
                   vmin=0, vmax=1,
                   cbar_kws={'label': 'Trust Score'},
                   yticklabels=client_labels,
                   xticklabels=10)  # Show every 10th round
        
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Client', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.tight_layout()
            plt.show()
        
        plt.close()
    
    @staticmethod
    def detect_trust_anomalies(
        trust_history: List[float],
        window: int = 5,
        threshold: float = 0.3
    ) -> List[int]:
        """
        Detect anomalies in trust evolution
        
        Args:
            trust_history: Trust history for a client
            window: Window for anomaly detection
            threshold: Threshold for anomaly detection
            
        Returns:
            anomaly_rounds: List of rounds with anomalies
        """
        anomalies = []
        
        for i in range(window, len(trust_history)):
            # Check for sudden drops
            recent_avg = np.mean(trust_history[i-window:i])
            current = trust_history[i]
            
            if recent_avg - current > threshold:
                anomalies.append(i)
        
        return anomalies
    
    @staticmethod
    def print_trust_report(
        benign_stats: Dict[str, float],
        malicious_stats: Dict[str, float],
        separation: Dict[str, float]
    ):
        """Print formatted trust analysis report"""
        print("\n" + "="*70)
        print("TRUST EVOLUTION ANALYSIS")
        print("="*70)
        
        print("\n👥 Benign Clients:")
        print(f"  Mean Trust:         {benign_stats['mean']:.3f}")
        print(f"  Std Dev:            {benign_stats['std']:.3f}")
        print(f"  Range:              [{benign_stats['min']:.3f}, {benign_stats['max']:.3f}]")
        print(f"  Trend:              {benign_stats['trend']:+.5f}")
        print(f"  Final:              {benign_stats['final']:.3f}")
        
        print("\n⚠️ Malicious Clients:")
        print(f"  Mean Trust:         {malicious_stats['mean']:.3f}")
        print(f"  Std Dev:            {malicious_stats['std']:.3f}")
        print(f"  Range:              [{malicious_stats['min']:.3f}, {malicious_stats['max']:.3f}]")
        print(f"  Trend:              {malicious_stats['trend']:+.5f}")
        print(f"  Final:              {malicious_stats['final']:.3f}")
        
        print("\n📊 Separation Metrics:")
        print(f"  Average Separation: {separation['average_separation']:.3f}")
        print(f"  Min Separation:     {separation['min_separation']:.3f}")
        print(f"  Final Separation:   {separation['final_separation']:.3f}")
        
        # Interpretation
        if separation['average_separation'] > 0.3:
            interpretation = "✓ Excellent separation"
        elif separation['average_separation'] > 0.15:
            interpretation = "~ Moderate separation"
        else:
            interpretation = "✗ Poor separation"
        
        print(f"\n  Interpretation:     {interpretation}")
        
        print("="*70)