import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np
import copy
from fl_core.client import FLClient


class FLServer:
    """
    Federated Learning Server
    Coordinates training rounds and aggregation
    """
    
    def __init__(
        self,
        model: nn.Module,
        clients: List[FLClient],
        aggregation_method: str = "fedavg",
        clients_per_round: int = 10,
        device: str = "cpu"
    ):
        self.global_model = model.to(device)
        self.clients = clients
        self.aggregation_method = aggregation_method
        self.clients_per_round = clients_per_round
        self.device = device
        self.current_round = 0
        
        # Training history
        self.history = {
            'global_loss': [],
            'global_accuracy': [],
            'selected_clients': []
        }
    
    def get_global_parameters(self) -> Dict[str, torch.Tensor]:
        """Get global model parameters"""
        return {name: param.data.clone() for name, param in self.global_model.named_parameters()}
    
    def set_global_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Set global model parameters"""
        for name, param in self.global_model.named_parameters():
            param.data = parameters[name].clone()
    
    def select_clients(self, round_num: int) -> List[FLClient]:
        """
        Select clients for the current round
        
        Args:
            round_num: Current round number
            
        Returns:
            selected_clients: List of selected clients
        """
        np.random.seed(round_num)  # For reproducibility
        num_clients = min(self.clients_per_round, len(self.clients))
        selected_indices = np.random.choice(len(self.clients), num_clients, replace=False)
        selected_clients = [self.clients[i] for i in selected_indices]
        
        self.history['selected_clients'].append([c.client_id for c in selected_clients])
        
        return selected_clients
    
    def distribute_model(self, clients: List[FLClient]):
        """Distribute global model to selected clients"""
        global_params = self.get_global_parameters()
        for client in clients:
            client.set_parameters(global_params)
    
    def collect_updates(
        self, 
        clients: List[FLClient]
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, float]]]:
        """
        Collect model updates from clients
        
        Args:
            clients: List of clients to collect from
            
        Returns:
            updates: List of model updates (deltas)
            metrics: List of training metrics
        """
        updates = []
        metrics = []
        
        for client in clients:
            update, client_metrics = client.train()
            updates.append(update)
            metrics.append(client_metrics)
        
        return updates, metrics
    
    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates
        
        Args:
            updates: List of model updates from clients
            weights: Aggregation weights (if None, uses uniform weights)
            
        Returns:
            aggregated_update: Aggregated model update
        """
        if weights is None:
            weights = [1.0 / len(updates)] * len(updates)
        else:
            # Normalize weights
            total = sum(weights)
            weights = [w / total for w in weights]
        
        aggregated_update = {}
        
        # Get parameter names from first update
        param_names = list(updates[0].keys())
        
        for name in param_names:
            # Weighted average of updates
            aggregated_param = torch.zeros_like(updates[0][name])
            for update, weight in zip(updates, weights):
                aggregated_param += weight * update[name]
            aggregated_update[name] = aggregated_param
        
        return aggregated_update
    
    def apply_update(self, aggregated_update: Dict[str, torch.Tensor]):
        """Apply aggregated update to global model"""
        current_params = self.get_global_parameters()
        
        for name in current_params.keys():
            current_params[name] += aggregated_update[name]
        
        self.set_global_parameters(current_params)
    
    def evaluate_global_model(self, test_loader) -> Dict[str, float]:
        """Evaluate global model on test set"""
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = test_loss / len(test_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train_round(self, round_num: int, test_loader=None) -> Dict[str, float]:
        """
        Execute one training round
        
        Args:
            round_num: Current round number
            test_loader: Test data loader for evaluation
            
        Returns:
            metrics: Global model metrics
        """
        self.current_round = round_num
        
        # Select clients
        selected_clients = self.select_clients(round_num)
        
        # Distribute global model
        self.distribute_model(selected_clients)
        
        # Collect updates
        updates, client_metrics = self.collect_updates(selected_clients)
        
        # Aggregate updates (weights handled by aggregation method)
        aggregated_update = self.aggregate(updates)
        
        # Apply update to global model
        self.apply_update(aggregated_update)
        
        # Evaluate global model
        if test_loader is not None:
            global_metrics = self.evaluate_global_model(test_loader)
            self.history['global_loss'].append(global_metrics['loss'])
            self.history['global_accuracy'].append(global_metrics['accuracy'])
        else:
            global_metrics = {'loss': 0.0, 'accuracy': 0.0}
        
        return global_metrics
    
    def get_history(self) -> Dict:
        """Get training history"""
        return self.history