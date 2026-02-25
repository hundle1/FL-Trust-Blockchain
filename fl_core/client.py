import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
        
        self.training_history = {'loss': [], 'accuracy': [], 'gradient_norms': []}
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        return {name: param.data.clone() for name, param in self.model.named_parameters()}
    
    def set_parameters(self, parameters: Dict[str, torch.Tensor]):
        for name, param in self.model.named_parameters():
            param.data = parameters[name].clone()
    
    def train(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Perform local training.

        Returns:
            model_update: Parameter delta (new - initial)
            metrics:      loss, accuracy, gradient_norm
        """
        initial_params = self.get_parameters()
        
        self.model.train()
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0
        
        for epoch in range(self.local_epochs):
            batch_loss, batch_correct, batch_total = 0.0, 0, 0
            
            for data, target in self.train_data:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                batch_loss += loss.item()
                batch_correct += output.argmax(dim=1).eq(target).sum().item()
                batch_total += target.size(0)
            
            epoch_loss += batch_loss / len(self.train_data)
            epoch_correct += batch_correct
            epoch_total += batch_total
        
        avg_loss = epoch_loss / self.local_epochs
        avg_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0.0
        
        current_params = self.get_parameters()
        model_update = {}
        gradient_norm_sq = 0.0
        
        for name in initial_params:
            delta = current_params[name] - initial_params[name]
            model_update[name] = delta
            gradient_norm_sq += torch.norm(delta).item() ** 2
        
        gradient_norm = gradient_norm_sq ** 0.5
        
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
        loader = data_loader or self.test_data
        if loader is None:
            return {'loss': 0.0, 'accuracy': 0.0}
        
        self.model.eval()
        test_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                correct += output.argmax(dim=1).eq(target).sum().item()
                total += target.size(0)
        
        return {
            'loss': test_loss / len(loader),
            'accuracy': correct / total if total > 0 else 0.0
        }
    
    def get_history(self) -> Dict:
        return self.training_history
    
    def reset_history(self):
        self.training_history = {'loss': [], 'accuracy': [], 'gradient_norms': []}