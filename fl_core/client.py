import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
from typing import Dict, Tuple, Optional


class FLClient:
    """
    Federated Learning Client
    Handles local training and gradient computation
    """
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_data: DataLoader,
        test_data: Optional[DataLoader] = None,
        device: str = "cpu",
        lr: float = 0.01,
        local_epochs: int = 5,
        is_malicious: bool = False
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.train_data = train_data
        self.test_data = test_data
        self.device = device
        self.lr = lr
        self.local_epochs = local_epochs
        self.is_malicious = is_malicious
        
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        
        # Track client history
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'gradient_norms': []
        }
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get model parameters"""
        return {name: param.data.clone() for name, param in self.model.named_parameters()}
    
    def set_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Set model parameters"""
        for name, param in self.model.named_parameters():
            param.data = parameters[name].clone()
    
    def train(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Perform local training
        
        Returns:
            model_update: Dictionary of parameter updates (delta)
            metrics: Training metrics (loss, accuracy, gradient_norm)
        """
        # Store initial parameters
        initial_params = self.get_parameters()
        
        self.model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for epoch in range(self.local_epochs):
            batch_loss = 0.0
            batch_correct = 0
            batch_total = 0
            
            for batch_idx, (data, target) in enumerate(self.train_data):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                batch_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                batch_correct += pred.eq(target.view_as(pred)).sum().item()
                batch_total += target.size(0)
            
            epoch_loss += batch_loss / len(self.train_data)
            epoch_correct += batch_correct
            epoch_total += batch_total
        
        # Calculate average metrics
        avg_loss = epoch_loss / self.local_epochs
        avg_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0
        
        # Compute model update (delta = new_params - initial_params)
        current_params = self.get_parameters()
        model_update = {}
        gradient_norm = 0.0
        
        for name in initial_params.keys():
            delta = current_params[name] - initial_params[name]
            model_update[name] = delta
            gradient_norm += torch.norm(delta).item() ** 2
        
        gradient_norm = gradient_norm ** 0.5
        
        # Store metrics
        metrics = {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'gradient_norm': gradient_norm
        }
        
        self.training_history['loss'].append(avg_loss)
        self.training_history['accuracy'].append(avg_accuracy)
        self.training_history['gradient_norms'].append(gradient_norm)
        
        return model_update, metrics
    
    def evaluate(self, data_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Evaluate model on test data
        
        Args:
            data_loader: DataLoader to evaluate on (uses self.test_data if None)
            
        Returns:
            metrics: Dictionary with loss and accuracy
        """
        if data_loader is None:
            data_loader = self.test_data
        
        if data_loader is None:
            return {'loss': 0.0, 'accuracy': 0.0}
        
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = test_loss / len(data_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def compute_gradient(self, batch_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute gradient on a single batch (used for gradient similarity)
        
        Args:
            batch_data: Tuple of (data, target)
            
        Returns:
            gradients: Dictionary of gradients per parameter
        """
        self.model.train()
        data, target = batch_data
        data, target = data.to(self.device), target.to(self.device)
        
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)
        loss.backward()
        
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        
        return gradients
    
    def get_history(self) -> Dict:
        """Get training history"""
        return self.training_history
    
    def reset_history(self):
        """Reset training history"""
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'gradient_norms': []
        }