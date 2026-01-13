"""
Smart Contract Simulation
Mock smart contract for automated actions based on trust
"""

from typing import Dict, Callable, Any


class SmartContract:
    """
    Simulated smart contract for FL trust management
    Automatically executes actions based on trust levels
    """
    
    def __init__(
        self,
        penalty_threshold: float = 0.2,
        reward_threshold: float = 0.8,
        enable_auto_execute: bool = False
    ):
        """
        Args:
            penalty_threshold: Trust below this triggers penalty
            reward_threshold: Trust above this triggers reward
            enable_auto_execute: Whether to auto-execute actions
        """
        self.penalty_threshold = penalty_threshold
        self.reward_threshold = reward_threshold
        self.enable_auto_execute = enable_auto_execute
        
        # Contract state
        self.client_penalties = {}
        self.client_rewards = {}
        self.action_log = []
        
        # Registered actions
        self.penalty_actions = []
        self.reward_actions = []
    
    def register_penalty_action(self, action: Callable):
        """
        Register an action to execute when client is penalized
        
        Args:
            action: Callable that takes (client_id, trust_score)
        """
        self.penalty_actions.append(action)
    
    def register_reward_action(self, action: Callable):
        """
        Register an action to execute when client is rewarded
        
        Args:
            action: Callable that takes (client_id, trust_score)
        """
        self.reward_actions.append(action)
    
    def evaluate(self, client_id: int, trust_score: float, round_num: int) -> Dict[str, Any]:
        """
        Evaluate trust and trigger actions if needed
        
        Args:
            client_id: Client to evaluate
            trust_score: Current trust score
            round_num: Current round number
            
        Returns:
            action_result: Dictionary of actions taken
        """
        result = {
            'client_id': client_id,
            'trust_score': trust_score,
            'round': round_num,
            'penalty_triggered': False,
            'reward_triggered': False,
            'actions_executed': []
        }
        
        # Check for penalty
        if trust_score < self.penalty_threshold:
            result['penalty_triggered'] = True
            
            if client_id not in self.client_penalties:
                self.client_penalties[client_id] = 0
            self.client_penalties[client_id] += 1
            
            # Execute penalty actions
            if self.enable_auto_execute:
                for action in self.penalty_actions:
                    action_result = action(client_id, trust_score)
                    result['actions_executed'].append({
                        'type': 'penalty',
                        'result': action_result
                    })
        
        # Check for reward
        elif trust_score > self.reward_threshold:
            result['reward_triggered'] = True
            
            if client_id not in self.client_rewards:
                self.client_rewards[client_id] = 0
            self.client_rewards[client_id] += 1
            
            # Execute reward actions
            if self.enable_auto_execute:
                for action in self.reward_actions:
                    action_result = action(client_id, trust_score)
                    result['actions_executed'].append({
                        'type': 'reward',
                        'result': action_result
                    })
        
        # Log action
        self.action_log.append(result)
        
        return result
    
    def get_client_penalty_count(self, client_id: int) -> int:
        """Get number of penalties for client"""
        return self.client_penalties.get(client_id, 0)
    
    def get_client_reward_count(self, client_id: int) -> int:
        """Get number of rewards for client"""
        return self.client_rewards.get(client_id, 0)
    
    def get_action_log(self) -> list:
        """Get log of all actions"""
        return self.action_log
    
    def get_statistics(self) -> Dict:
        """Get contract statistics"""
        total_penalties = sum(self.client_penalties.values())
        total_rewards = sum(self.client_rewards.values())
        
        return {
            'total_evaluations': len(self.action_log),
            'total_penalties': total_penalties,
            'total_rewards': total_rewards,
            'clients_penalized': len(self.client_penalties),
            'clients_rewarded': len(self.client_rewards),
            'penalty_threshold': self.penalty_threshold,
            'reward_threshold': self.reward_threshold
        }


# Example penalty/reward actions
def ban_client_action(client_id: int, trust_score: float) -> str:
    """Example: Ban client from participation"""
    return f"Client {client_id} banned due to low trust ({trust_score:.2f})"


def reduce_weight_action(client_id: int, trust_score: float) -> str:
    """Example: Reduce aggregation weight"""
    return f"Client {client_id} weight reduced (trust: {trust_score:.2f})"


def increase_weight_action(client_id: int, trust_score: float) -> str:
    """Example: Increase aggregation weight"""
    return f"Client {client_id} weight increased (trust: {trust_score:.2f})"


def send_notification_action(client_id: int, trust_score: float) -> str:
    """Example: Send notification"""
    return f"Notification sent to client {client_id} (trust: {trust_score:.2f})"