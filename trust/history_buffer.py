"""
History Buffer Module
Manages historical data for trust scoring
"""

from collections import deque
from typing import Any, List


class HistoryBuffer:
    """
    Circular buffer for storing historical metrics
    Used to track client behavior over time
    """
    
    def __init__(self, maxlen: int = 10):
        """
        Args:
            maxlen: Maximum number of items to store
        """
        self.maxlen = maxlen
        self.buffer = deque(maxlen=maxlen)
    
    def append(self, item: Any):
        """Add item to buffer"""
        self.buffer.append(item)
    
    def get_all(self) -> List[Any]:
        """Get all items in buffer"""
        return list(self.buffer)
    
    def get_recent(self, n: int) -> List[Any]:
        """Get n most recent items"""
        return list(self.buffer)[-n:]
    
    def get_average(self) -> float:
        """Get average of numeric values"""
        if not self.buffer:
            return 0.0
        return sum(self.buffer) / len(self.buffer)
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
    
    def __len__(self) -> int:
        """Get buffer length"""
        return len(self.buffer)
    
    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        return len(self.buffer) == 0
    
    def is_full(self) -> bool:
        """Check if buffer is full"""
        return len(self.buffer) == self.maxlen


class ClientHistoryManager:
    """
    Manages history buffers for multiple clients
    """
    
    def __init__(self, num_clients: int, window_size: int = 10):
        """
        Args:
            num_clients: Number of clients
            window_size: Size of history window for each client
        """
        self.num_clients = num_clients
        self.window_size = window_size
        
        # Create buffers for each metric
        self.gradient_history = [HistoryBuffer(window_size) for _ in range(num_clients)]
        self.loss_history = [HistoryBuffer(window_size) for _ in range(num_clients)]
        self.accuracy_history = [HistoryBuffer(window_size) for _ in range(num_clients)]
        self.trust_history = [HistoryBuffer(window_size) for _ in range(num_clients)]
    
    def add_gradient_similarity(self, client_id: int, similarity: float):
        """Store gradient similarity for client"""
        self.gradient_history[client_id].append(similarity)
    
    def add_loss(self, client_id: int, loss: float):
        """Store loss for client"""
        self.loss_history[client_id].append(loss)
    
    def add_accuracy(self, client_id: int, accuracy: float):
        """Store accuracy for client"""
        self.accuracy_history[client_id].append(accuracy)
    
    def add_trust(self, client_id: int, trust: float):
        """Store trust score for client"""
        self.trust_history[client_id].append(trust)
    
    def get_gradient_history(self, client_id: int) -> List[float]:
        """Get gradient history for client"""
        return self.gradient_history[client_id].get_all()
    
    def get_loss_history(self, client_id: int) -> List[float]:
        """Get loss history for client"""
        return self.loss_history[client_id].get_all()
    
    def get_accuracy_history(self, client_id: int) -> List[float]:
        """Get accuracy history for client"""
        return self.accuracy_history[client_id].get_all()
    
    def get_trust_history(self, client_id: int) -> List[float]:
        """Get trust history for client"""
        return self.trust_history[client_id].get_all()
    
    def clear_client(self, client_id: int):
        """Clear all history for a specific client"""
        self.gradient_history[client_id].clear()
        self.loss_history[client_id].clear()
        self.accuracy_history[client_id].clear()
        self.trust_history[client_id].clear()
    
    def get_statistics(self, client_id: int) -> dict:
        """Get statistics for a client"""
        return {
            'avg_gradient_similarity': self.gradient_history[client_id].get_average(),
            'avg_loss': self.loss_history[client_id].get_average(),
            'avg_accuracy': self.accuracy_history[client_id].get_average(),
            'avg_trust': self.trust_history[client_id].get_average(),
            'history_length': len(self.gradient_history[client_id])
        }