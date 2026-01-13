"""
FL Trainer Utility
High-level training orchestration
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
from fl_core.client import FLClient
from fl_core.server import FLServer


class FLTrainer:
    """
    Federated Learning Trainer
    Orchestrates the entire training process
    """
    
    def __init__(
        self,
        global_model: nn.Module,
        clients: List[FLClient],
        test_loader,
        aggregator,
        trust_manager=None,
        blockchain=None,
        device: str = "cpu"
    ):
        self.global_model = global_model.to(device)
        self.clients = clients
        self.test_loader = test_loader
        self.aggregator = aggregator
        self.trust_manager = trust_manager
        self.blockchain = blockchain
        self.device = device
        
        self.history = {
            'accuracy': [],
            'loss': [],
            'trust_scores': []
        }
    
    def train_round(self, round_num: int, selected_clients: List[FLClient]) -> Dict:
        """Execute one training round"""
        
        # Distribute global model
        global_params = {name: param.data.clone() 
                        for name, param in self.global_model.named_parameters()}
        
        for client in selected_clients:
            client.set_parameters(global_params)
        
        # Collect updates
        updates = []
        client_ids = []
        metrics_list = []
        
        for client in selected_clients:
            update, metrics = client.train()
            updates.append(update)
            client_ids.append(client.client_id)
            metrics_list.append(metrics)
        
        # Update trust if available
        if self.trust_manager:
            # Compute reference gradient
            benign_updates = updates  # Simplified
            ref_grad = {key: torch.mean(torch.stack([u[key] for u in benign_updates]), dim=0)
                       for key in benign_updates[0].keys()}
            
            for i, cid in enumerate(client_ids):
                self.trust_manager.update_trust(cid, updates[i], ref_grad, metrics_list[i])
                
                # Log to blockchain
                if self.blockchain:
                    self.blockchain.log_client_update(
                        round_num, cid,
                        self.trust_manager.get_trust_score(cid),
                        metrics_list[i]
                    )
        
        # Aggregate
        agg_update = self.aggregator.aggregate(updates, client_ids, metrics_list)
        
        # Apply to global model
        for name, param in self.global_model.named_parameters():
            param.data += agg_update[name]
        
        # Evaluate
        metrics = self.evaluate()
        
        return metrics
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate global model"""
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = test_loss / len(self.test_loader)
        
        return {'accuracy': accuracy, 'loss': avg_loss}
    
    def get_history(self) -> Dict:
        """Get training history"""
        return self.history