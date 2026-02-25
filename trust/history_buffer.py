"""
History Buffer Module
Manages per-client historical data for trust scoring.
"""

from collections import deque
from typing import Any, List


class HistoryBuffer:
    """Circular buffer for storing historical metrics."""
    
    def __init__(self, maxlen: int = 10):
        self.maxlen = maxlen
        self.buffer = deque(maxlen=maxlen)
    
    def append(self, item: Any):
        self.buffer.append(item)
    
    def get_all(self) -> List[Any]:
        return list(self.buffer)
    
    def get_recent(self, n: int) -> List[Any]:
        return list(self.buffer)[-n:]
    
    def get_average(self) -> float:
        return sum(self.buffer) / len(self.buffer) if self.buffer else 0.0
    
    def clear(self):
        self.buffer.clear()
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def is_empty(self) -> bool:
        return len(self.buffer) == 0
    
    def is_full(self) -> bool:
        return len(self.buffer) == self.maxlen


class ClientHistoryManager:
    """Manages history buffers for all clients."""
    
    def __init__(self, num_clients: int, window_size: int = 10):
        self.num_clients = num_clients
        self.window_size = window_size
        
        self.gradient_history = [HistoryBuffer(window_size) for _ in range(num_clients)]
        self.loss_history = [HistoryBuffer(window_size) for _ in range(num_clients)]
        self.accuracy_history = [HistoryBuffer(window_size) for _ in range(num_clients)]
        self.trust_history = [HistoryBuffer(window_size) for _ in range(num_clients)]
    
    def add_gradient_similarity(self, client_id: int, similarity: float):
        self.gradient_history[client_id].append(similarity)
    
    def add_loss(self, client_id: int, loss: float):
        self.loss_history[client_id].append(loss)
    
    def add_accuracy(self, client_id: int, accuracy: float):
        self.accuracy_history[client_id].append(accuracy)
    
    def add_trust(self, client_id: int, trust: float):
        self.trust_history[client_id].append(trust)
    
    def get_gradient_history(self, client_id: int) -> List[float]:
        return self.gradient_history[client_id].get_all()
    
    def get_loss_history(self, client_id: int) -> List[float]:
        return self.loss_history[client_id].get_all()
    
    def get_accuracy_history(self, client_id: int) -> List[float]:
        return self.accuracy_history[client_id].get_all()
    
    def get_trust_history(self, client_id: int) -> List[float]:
        return self.trust_history[client_id].get_all()
    
    def clear_client(self, client_id: int):
        for buf in [self.gradient_history[client_id], self.loss_history[client_id],
                    self.accuracy_history[client_id], self.trust_history[client_id]]:
            buf.clear()
    
    def get_statistics(self, client_id: int) -> dict:
        return {
            'avg_gradient_similarity': self.gradient_history[client_id].get_average(),
            'avg_loss': self.loss_history[client_id].get_average(),
            'avg_accuracy': self.accuracy_history[client_id].get_average(),
            'avg_trust': self.trust_history[client_id].get_average(),
            'history_length': len(self.gradient_history[client_id])
        }