"""
Mock Chain - Higher-level Blockchain API
Wraps MockBlockchain (ledger.py) with a cleaner interface
for use in FL experiments.
"""

import time
import json
from typing import Dict, List, Optional, Any
from blockchain.ledger import MockBlockchain


class MockChain:
    """
    High-level blockchain interface for FL trust logging.
    
    Wraps the low-level MockBlockchain (ledger.py) and provides:
        - Simplified logging API
        - Automatic block creation
        - Overhead tracking
        - Chain integrity verification
    """

    def __init__(
        self,
        consensus_latency: float = 0.001,
        block_time: float = 0.05,
        transactions_per_block: int = 10,
        enabled: bool = True
    ):
        """
        Args:
            consensus_latency:       Simulated consensus delay per write (s)
            block_time:              Simulated block mining time (s)
            transactions_per_block:  Auto-create block after N transactions
            enabled:                 Disable to run without any blockchain overhead
        """
        self.enabled = enabled
        self.transactions_per_block = transactions_per_block

        if enabled:
            self._chain = MockBlockchain(
                consensus_latency=consensus_latency,
                block_time=block_time,
                track_storage=True
            )
        else:
            self._chain = None

        self._tx_count = 0
        self._round_logs: Dict[int, List[Dict]] = {}

    # ------------------------------------------------------------------
    # Logging API
    # ------------------------------------------------------------------

    def log_client_update(
        self,
        round_num: int,
        client_id: int,
        trust_score: float,
        metrics: Dict[str, float],
        is_flagged: bool = False
    ):
        """
        Log a client update event.

        Args:
            round_num:   FL training round
            client_id:   Client identifier
            trust_score: Current trust score
            metrics:     Training metrics (loss, accuracy, gradient_norm)
            is_flagged:  Whether client was flagged as suspicious
        """
        if not self.enabled:
            return

        self._chain.log_client_update(round_num, client_id, trust_score, metrics)
        self._tx_count += 1

        # Also flag if suspicious
        if is_flagged:
            self._chain.log_attack_flag(
                round_num, client_id,
                reason="low_trust",
                trust_score=trust_score
            )

        self._maybe_create_block()

    def log_round_aggregation(
        self,
        round_num: int,
        selected_clients: List[int],
        method: str,
        global_metrics: Dict[str, float]
    ):
        """
        Log round aggregation event.

        Args:
            round_num:        FL round number
            selected_clients: List of client IDs selected this round
            method:           Aggregation method name
            global_metrics:   Global model metrics after aggregation
        """
        if not self.enabled:
            return

        self._chain.log_aggregation(round_num, selected_clients, method, global_metrics)
        self._tx_count += 1
        self._maybe_create_block()

    def log_trust_update(
        self,
        round_num: int,
        client_id: int,
        old_trust: float,
        new_trust: float
    ):
        """Log a trust score change."""
        if not self.enabled:
            return

        # Store in pending transactions
        self._chain.add_transaction({
            'type': 'trust_update',
            'round': round_num,
            'client_id': client_id,
            'old_trust': old_trust,
            'new_trust': new_trust,
            'delta': new_trust - old_trust,
            'timestamp': time.time()
        })
        self._tx_count += 1
        self._maybe_create_block()

    def flag_malicious(
        self,
        round_num: int,
        client_id: int,
        trust_score: float,
        reason: str = "low_trust"
    ):
        """Flag a client as malicious on the chain."""
        if not self.enabled:
            return

        self._chain.log_attack_flag(round_num, client_id, reason, trust_score)
        self._tx_count += 1

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _maybe_create_block(self):
        """Create a new block if enough transactions have accumulated."""
        if self._tx_count > 0 and self._tx_count % self.transactions_per_block == 0:
            self._chain.create_block()

    def flush(self):
        """Force creation of a block with any remaining pending transactions."""
        if self.enabled and self._chain and self._chain.pending_transactions:
            self._chain.create_block()

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_client_history(self, client_id: int) -> List[Dict]:
        """Get all logged events for a specific client."""
        if not self.enabled:
            return []
        return self._chain.get_client_history(client_id)

    def get_round_data(self, round_num: int) -> List[Dict]:
        """Get all logged events for a specific round."""
        if not self.enabled:
            return []
        return self._chain.get_round_data(round_num)

    def is_valid(self) -> bool:
        """Verify blockchain integrity."""
        if not self.enabled:
            return True
        return self._chain.verify_chain()

    # ------------------------------------------------------------------
    # Overhead metrics
    # ------------------------------------------------------------------

    def get_overhead_metrics(self) -> Dict[str, Any]:
        """Return performance overhead metrics."""
        if not self.enabled:
            return {
                'enabled': False,
                'total_blocks': 0,
                'total_write_time': 0.0,
                'total_read_time': 0.0,
                'total_storage_kb': 0.0
            }

        metrics = self._chain.get_overhead_metrics()
        metrics['enabled'] = True
        metrics['total_transactions'] = self._tx_count
        return metrics

    def get_summary(self) -> Dict:
        """High-level summary of chain state."""
        if not self.enabled:
            return {'enabled': False}

        summary = self._chain.get_chain_summary()
        summary['total_transactions'] = self._tx_count
        return summary

    def print_summary(self):
        """Print formatted blockchain summary."""
        summary = self.get_summary()
        overhead = self.get_overhead_metrics()

        print(f"\n{'='*55}")
        print(f"BLOCKCHAIN SUMMARY")
        print(f"{'='*55}")

        if not self.enabled:
            print("  Blockchain: DISABLED")
        else:
            print(f"  Chain Length:       {summary['chain_length']} blocks")
            print(f"  Total Transactions: {summary['total_transactions']}")
            print(f"  Pending Txns:       {summary['pending_transactions']}")
            print(f"  Chain Valid:        {summary['is_valid']}")
            print(f"\n  Performance:")
            print(f"  Write Time:         {overhead['total_write_time']:.3f}s")
            print(f"  Read Time:          {overhead['total_read_time']:.3f}s")
            print(f"  Storage:            {overhead['total_storage_kb']:.2f} KB")

        print(f"{'='*55}")