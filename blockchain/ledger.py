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
        """Convert block to dictionary"""
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
    Simulates blockchain overhead without real consensus
    """
    
    def __init__(
        self,
        consensus_latency: float = 0.1,
        block_time: float = 5.0,
        track_storage: bool = True
    ):
        """
        Args:
            consensus_latency: Simulated consensus delay (seconds)
            block_time: Time to create new block (seconds)
            track_storage: Whether to track storage overhead
        """
        self.chain: List[Block] = []
        self.pending_transactions: List[Dict] = []
        self.consensus_latency = consensus_latency
        self.block_time = block_time
        self.track_storage = track_storage
        
        # Metrics
        self.total_write_time = 0.0
        self.total_read_time = 0.0
        self.total_storage_bytes = 0
        
        # Create genesis block
        self._create_genesis_block()
    
    def _create_genesis_block(self):
        """Create the first block in the chain"""
        genesis_block = Block(0, time.time(), {"message": "Genesis Block"}, "0")
        self.chain.append(genesis_block)
        self._update_storage(genesis_block)
    
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
        """
        Log client update to blockchain
        
        Args:
            round_num: Current round number
            client_id: Client ID
            trust_score: Current trust score
            metrics: Training metrics
        """
        start_time = time.time()
        
        transaction = {
            'type': 'client_update',
            'round': round_num,
            'client_id': client_id,
            'trust_score': trust_score,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        self.add_transaction(transaction)
        
        # Simulate consensus latency
        time.sleep(self.consensus_latency)
        
        write_time = time.time() - start_time
        self.total_write_time += write_time
    
    def log_aggregation(
        self,
        round_num: int,
        selected_clients: List[int],
        aggregation_method: str,
        global_metrics: Dict[str, float]
    ):
        """Log aggregation results"""
        start_time = time.time()
        
        transaction = {
            'type': 'aggregation',
            'round': round_num,
            'selected_clients': selected_clients,
            'aggregation_method': aggregation_method,
            'global_metrics': global_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        self.add_transaction(transaction)
        time.sleep(self.consensus_latency)
        
        write_time = time.time() - start_time
        self.total_write_time += write_time
    
    def log_attack_flag(
        self,
        round_num: int,
        client_id: int,
        reason: str,
        trust_score: float
    ):
        """Log detected anomaly/attack"""
        start_time = time.time()
        
        transaction = {
            'type': 'attack_flag',
            'round': round_num,
            'client_id': client_id,
            'reason': reason,
            'trust_score': trust_score,
            'timestamp': datetime.now().isoformat()
        }
        
        self.add_transaction(transaction)
        time.sleep(self.consensus_latency)
        
        write_time = time.time() - start_time
        self.total_write_time += write_time
    
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
        
        # Simulate block creation time
        time.sleep(self.block_time)
        
        self.chain.append(new_block)
        self.pending_transactions = []
        
        self._update_storage(new_block)
        
        write_time = time.time() - start_time
        self.total_write_time += write_time
        
        return new_block
    
    def _update_storage(self, block: Block):
        """Update storage metrics"""
        if self.track_storage:
            block_json = json.dumps(block.to_dict())
            block_size = len(block_json.encode('utf-8'))
            self.total_storage_bytes += block_size
    
    def get_client_history(self, client_id: int) -> List[Dict]:
        """Retrieve all transactions for a specific client"""
        start_time = time.time()
        
        history = []
        for block in self.chain:
            if 'transactions' in block.data:
                for tx in block.data['transactions']:
                    if tx.get('client_id') == client_id:
                        history.append(tx)
        
        read_time = time.time() - start_time
        self.total_read_time += read_time
        
        return history
    
    def get_round_data(self, round_num: int) -> List[Dict]:
        """Retrieve all transactions for a specific round"""
        start_time = time.time()
        
        round_data = []
        for block in self.chain:
            if 'transactions' in block.data:
                for tx in block.data['transactions']:
                    if tx.get('round') == round_num:
                        round_data.append(tx)
        
        read_time = time.time() - start_time
        self.total_read_time += read_time
        
        return round_data
    
    def verify_chain(self) -> bool:
        """Verify blockchain integrity"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Check hash
            if current_block.hash != current_block.calculate_hash():
                return False
            
            # Check previous hash link
            if current_block.previous_hash != previous_block.hash:
                return False
        
        return True
    
    def get_overhead_metrics(self) -> Dict[str, Any]:
        """Get blockchain overhead metrics"""
        return {
            'total_blocks': len(self.chain),
            'total_write_time': self.total_write_time,
            'total_read_time': self.total_read_time,
            'total_storage_kb': self.total_storage_bytes / 1024,
            'avg_write_time_per_block': self.total_write_time / max(1, len(self.chain)),
            'storage_per_block_kb': (self.total_storage_bytes / 1024) / max(1, len(self.chain))
        }
    
    def get_chain_summary(self) -> Dict:
        """Get summary of blockchain state"""
        return {
            'chain_length': len(self.chain),
            'pending_transactions': len(self.pending_transactions),
            'is_valid': self.verify_chain(),
            'overhead': self.get_overhead_metrics()
        }