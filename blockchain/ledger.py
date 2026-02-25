import hashlib
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime


class Block:
    """Single block in the blockchain"""
    
    def __init__(self, index: int, timestamp: float, data: Dict, previous_hash: str):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """Calculate SHA256 hash of block"""
        block_string = json.dumps({
            'index': self.index,
            'timestamp': self.timestamp,
            'data': str(self.data),
            'previous_hash': self.previous_hash
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        return {
            'index': self.index,
            'timestamp': self.timestamp,
            'data': self.data,
            'previous_hash': self.previous_hash,
            'hash': self.hash
        }


class MockBlockchain:
    """
    Mock Blockchain for Audit Logging
    Simulates blockchain overhead without real consensus.

    NOTE: Default block_time is 0.05s (simulation).
    Original value was 5.0s which caused 250s overhead for 50-round experiments.
    """
    
    def __init__(
        self,
        consensus_latency: float = 0.001,   # latency per transaction
        block_time: float = 0.05,            # FIX: was 5.0 → too slow for experiments
        track_storage: bool = True
    ):
        """
        Args:
            consensus_latency: Simulated consensus delay per write (seconds)
            block_time:        Simulated block mining time (seconds)
            track_storage:     Whether to track storage overhead
        """
        self.chain: List[Block] = []
        self.pending_transactions: List[Dict] = []
        self.consensus_latency = consensus_latency
        self.block_time = block_time
        self.track_storage = track_storage
        
        # Overhead metrics
        self.total_write_time = 0.0
        self.total_read_time = 0.0
        self.total_storage_bytes = 0
        
        self._create_genesis_block()
    
    def _create_genesis_block(self):
        genesis = Block(0, time.time(), {"message": "Genesis Block"}, "0")
        self.chain.append(genesis)
        self._update_storage(genesis)
    
    def add_transaction(self, transaction: Dict):
        """Add transaction to pending pool"""
        self.pending_transactions.append(transaction)
    
    def log_client_update(
        self,
        round_num: int,
        client_id: int,
        trust_score: float,
        metrics: Dict[str, float]
    ):
        """Log client update event"""
        start_time = time.time()
        self.add_transaction({
            'type': 'client_update',
            'round': round_num,
            'client_id': client_id,
            'trust_score': trust_score,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
        time.sleep(self.consensus_latency)
        self.total_write_time += time.time() - start_time
    
    def log_aggregation(
        self,
        round_num: int,
        selected_clients: List[int],
        aggregation_method: str,
        global_metrics: Dict[str, float]
    ):
        """Log aggregation results"""
        start_time = time.time()
        self.add_transaction({
            'type': 'aggregation',
            'round': round_num,
            'selected_clients': selected_clients,
            'aggregation_method': aggregation_method,
            'global_metrics': global_metrics,
            'timestamp': datetime.now().isoformat()
        })
        time.sleep(self.consensus_latency)
        self.total_write_time += time.time() - start_time
    
    def log_attack_flag(
        self,
        round_num: int,
        client_id: int,
        reason: str,
        trust_score: float
    ):
        """Log detected anomaly / attack flag"""
        start_time = time.time()
        self.add_transaction({
            'type': 'attack_flag',
            'round': round_num,
            'client_id': client_id,
            'reason': reason,
            'trust_score': trust_score,
            'timestamp': datetime.now().isoformat()
        })
        time.sleep(self.consensus_latency)
        self.total_write_time += time.time() - start_time
    
    def create_block(self) -> Optional[Block]:
        """Create new block from pending transactions"""
        if not self.pending_transactions:
            return None
        
        start_time = time.time()
        
        previous_block = self.chain[-1]
        new_block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            data={'transactions': self.pending_transactions.copy()},
            previous_hash=previous_block.hash
        )
        
        time.sleep(self.block_time)   # simulate block mining
        
        self.chain.append(new_block)
        self.pending_transactions = []
        self._update_storage(new_block)
        
        self.total_write_time += time.time() - start_time
        return new_block
    
    def _update_storage(self, block: Block):
        if self.track_storage:
            block_json = json.dumps(block.to_dict())
            self.total_storage_bytes += len(block_json.encode('utf-8'))
    
    def get_client_history(self, client_id: int) -> List[Dict]:
        start_time = time.time()
        history = []
        for block in self.chain:
            if 'transactions' in block.data:
                for tx in block.data['transactions']:
                    if tx.get('client_id') == client_id:
                        history.append(tx)
        self.total_read_time += time.time() - start_time
        return history
    
    def get_round_data(self, round_num: int) -> List[Dict]:
        start_time = time.time()
        data = []
        for block in self.chain:
            if 'transactions' in block.data:
                for tx in block.data['transactions']:
                    if tx.get('round') == round_num:
                        data.append(tx)
        self.total_read_time += time.time() - start_time
        return data
    
    def verify_chain(self) -> bool:
        """Verify blockchain integrity"""
        for i in range(1, len(self.chain)):
            cur = self.chain[i]
            prev = self.chain[i - 1]
            if cur.hash != cur.calculate_hash():
                return False
            if cur.previous_hash != prev.hash:
                return False
        return True
    
    def get_overhead_metrics(self) -> Dict[str, Any]:
        return {
            'total_blocks': len(self.chain),
            'total_write_time': self.total_write_time,
            'total_read_time': self.total_read_time,
            'total_storage_kb': self.total_storage_bytes / 1024,
            'avg_write_time_per_block': self.total_write_time / max(1, len(self.chain)),
            'storage_per_block_kb': (self.total_storage_bytes / 1024) / max(1, len(self.chain))
        }
    
    def get_chain_summary(self) -> Dict:
        return {
            'chain_length': len(self.chain),
            'pending_transactions': len(self.pending_transactions),
            'is_valid': self.verify_chain(),
            'overhead': self.get_overhead_metrics()
        }