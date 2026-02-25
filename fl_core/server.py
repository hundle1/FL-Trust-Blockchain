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
        
        self.history = {
            'global_loss': [],
            'global_accuracy': [],
            'selected_clients': []
        }
    
    def get_global_parameters(self) -> Dict[str, torch.Tensor]:
        return {name: param.data.clone() for name, param in self.global_model.named_parameters()}
    
    def set_global_parameters(self, parameters: Dict[str, torch.Tensor]):
        for name, param in self.global_model.named_parameters():
            param.data = parameters[name].clone()
    
    def select_clients(self, round_num: int) -> List[FLClient]:
        """
        Select clients for the current round.

        NOTE: Does NOT set np.random.seed internally — callers control global seed
        for reproducibility.
        """
        num_clients = min(self.clients_per_round, len(self.clients))
        # FIX: removed np.random.seed(round_num) which was overriding global seed
        selected_indices = np.random.choice(len(self.clients), num_clients, replace=False)
        selected_clients = [self.clients[i] for i in selected_indices]
        
        self.history['selected_clients'].append([c.client_id for c in selected_clients])
        return selected_clients
    
    def distribute_model(self, clients: List[FLClient]):
        global_params = self.get_global_parameters()
        for client in clients:
            client.set_parameters(global_params)
    
    def collect_updates(
        self,
        clients: List[FLClient]
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, float]]]:
        updates, metrics = [], []
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
        if weights is None:
            weights = [1.0 / len(updates)] * len(updates)
        else:
            total = sum(weights)
            weights = [w / total for w in weights]
        
        aggregated_update = {}
        for name in updates[0].keys():
            aggregated_param = torch.zeros_like(updates[0][name])
            for update, weight in zip(updates, weights):
                aggregated_param += weight * update[name]
            aggregated_update[name] = aggregated_param
        return aggregated_update
    
    def apply_update(self, aggregated_update: Dict[str, torch.Tensor]):
        current_params = self.get_global_parameters()
        for name in current_params.keys():
            current_params[name] += aggregated_update[name]
        self.set_global_parameters(current_params)
    
    def evaluate_global_model(self, test_loader) -> Dict[str, float]:
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        test_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        return {
            'loss': test_loss / len(test_loader),
            'accuracy': correct / total if total > 0 else 0.0
        }
    
    def train_round(self, round_num: int, test_loader=None) -> Dict[str, float]:
        self.current_round = round_num
        selected_clients = self.select_clients(round_num)
        self.distribute_model(selected_clients)
        updates, client_metrics = self.collect_updates(selected_clients)
        aggregated_update = self.aggregate(updates)
        self.apply_update(aggregated_update)
        
        if test_loader is not None:
            global_metrics = self.evaluate_global_model(test_loader)
            self.history['global_loss'].append(global_metrics['loss'])
            self.history['global_accuracy'].append(global_metrics['accuracy'])
        else:
            global_metrics = {'loss': 0.0, 'accuracy': 0.0}
        
        return global_metrics
    
    def get_history(self) -> Dict:
        return self.history