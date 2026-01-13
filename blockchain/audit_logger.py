"""
Audit Logger
Structured logging for blockchain audit trail
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional
import hashlib


class AuditLogger:
    """
    Audit logger for FL training events
    Creates structured audit trail
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Args:
            log_file: Path to log file (if None, logs to memory only)
        """
        self.log_file = log_file
        self.entries = []
        self.entry_hashes = []
    
    def log_event(
        self,
        event_type: str,
        round_num: int,
        data: Dict[str, Any],
        severity: str = "INFO"
    ) -> str:
        """
        Log an event
        
        Args:
            event_type: Type of event (client_update, aggregation, etc.)
            round_num: Training round number
            data: Event data
            severity: Log severity (INFO, WARNING, CRITICAL)
            
        Returns:
            entry_hash: Hash of the log entry
        """
        timestamp = datetime.now().isoformat()
        
        entry = {
            'timestamp': timestamp,
            'event_type': event_type,
            'round': round_num,
            'severity': severity,
            'data': data
        }
        
        # Compute hash
        entry_str = json.dumps(entry, sort_keys=True)
        entry_hash = hashlib.sha256(entry_str.encode()).hexdigest()
        
        entry['hash'] = entry_hash
        
        # Store
        self.entries.append(entry)
        self.entry_hashes.append(entry_hash)
        
        # Write to file if specified
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        
        return entry_hash
    
    def log_client_update(
        self,
        round_num: int,
        client_id: int,
        trust_score: float,
        metrics: Dict[str, float],
        gradient_norm: Optional[float] = None
    ):
        """Log client update event"""
        data = {
            'client_id': client_id,
            'trust_score': trust_score,
            'metrics': metrics
        }
        
        if gradient_norm is not None:
            data['gradient_norm'] = gradient_norm
        
        return self.log_event('client_update', round_num, data)
    
    def log_aggregation(
        self,
        round_num: int,
        selected_clients: list,
        aggregation_method: str,
        global_metrics: Dict[str, float]
    ):
        """Log aggregation event"""
        data = {
            'selected_clients': selected_clients,
            'num_clients': len(selected_clients),
            'aggregation_method': aggregation_method,
            'global_metrics': global_metrics
        }
        
        return self.log_event('aggregation', round_num, data)
    
    def log_attack_detection(
        self,
        round_num: int,
        client_id: int,
        trust_score: float,
        reason: str,
        confidence: float = 1.0
    ):
        """Log detected attack"""
        data = {
            'client_id': client_id,
            'trust_score': trust_score,
            'reason': reason,
            'confidence': confidence
        }
        
        return self.log_event('attack_detection', round_num, data, severity='WARNING')
    
    def log_trust_update(
        self,
        round_num: int,
        client_id: int,
        old_trust: float,
        new_trust: float,
        similarity: float
    ):
        """Log trust score update"""
        data = {
            'client_id': client_id,
            'old_trust': old_trust,
            'new_trust': new_trust,
            'trust_delta': new_trust - old_trust,
            'gradient_similarity': similarity
        }
        
        return self.log_event('trust_update', round_num, data)
    
    def log_model_checkpoint(
        self,
        round_num: int,
        accuracy: float,
        loss: float,
        checkpoint_path: Optional[str] = None
    ):
        """Log model checkpoint"""
        data = {
            'accuracy': accuracy,
            'loss': loss
        }
        
        if checkpoint_path:
            data['checkpoint_path'] = checkpoint_path
        
        return self.log_event('model_checkpoint', round_num, data)
    
    def get_entries_by_type(self, event_type: str) -> list:
        """Get all entries of specific type"""
        return [e for e in self.entries if e['event_type'] == event_type]
    
    def get_entries_by_round(self, round_num: int) -> list:
        """Get all entries for specific round"""
        return [e for e in self.entries if e['round'] == round_num]
    
    def get_entries_by_client(self, client_id: int) -> list:
        """Get all entries for specific client"""
        return [e for e in self.entries 
                if 'data' in e and 'client_id' in e['data'] and e['data']['client_id'] == client_id]
    
    def verify_integrity(self) -> bool:
        """
        Verify integrity of audit log
        
        Returns:
            valid: True if all hashes are valid
        """
        for entry in self.entries:
            # Remove hash from entry
            entry_hash = entry['hash']
            entry_copy = entry.copy()
            del entry_copy['hash']
            
            # Recompute hash
            entry_str = json.dumps(entry_copy, sort_keys=True)
            computed_hash = hashlib.sha256(entry_str.encode()).hexdigest()
            
            if computed_hash != entry_hash:
                return False
        
        return True
    
    def get_statistics(self) -> Dict:
        """Get audit log statistics"""
        event_counts = {}
        for entry in self.entries:
            event_type = entry['event_type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        severity_counts = {}
        for entry in self.entries:
            severity = entry.get('severity', 'INFO')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_entries': len(self.entries),
            'event_counts': event_counts,
            'severity_counts': severity_counts,
            'integrity_valid': self.verify_integrity(),
            'first_entry': self.entries[0]['timestamp'] if self.entries else None,
            'last_entry': self.entries[-1]['timestamp'] if self.entries else None
        }
    
    def export_to_file(self, filepath: str):
        """Export all entries to file"""
        with open(filepath, 'w') as f:
            json.dump(self.entries, f, indent=2)
    
    def generate_report(self) -> str:
        """Generate human-readable audit report"""
        stats = self.get_statistics()
        
        report = []
        report.append("="*70)
        report.append("AUDIT LOG REPORT")
        report.append("="*70)
        report.append(f"\nTotal Entries: {stats['total_entries']}")
        report.append(f"Integrity Valid: {stats['integrity_valid']}")
        
        if stats['first_entry'] and stats['last_entry']:
            report.append(f"Time Range: {stats['first_entry']} to {stats['last_entry']}")
        
        report.append("\nEvent Breakdown:")
        for event_type, count in stats['event_counts'].items():
            report.append(f"  {event_type}: {count}")
        
        report.append("\nSeverity Breakdown:")
        for severity, count in stats['severity_counts'].items():
            report.append(f"  {severity}: {count}")
        
        report.append("="*70)
        
        return '\n'.join(report)