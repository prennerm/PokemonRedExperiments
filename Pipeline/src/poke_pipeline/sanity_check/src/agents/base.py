"""
Base agent interface and common functionality.

This module provides the BaseAgent abstract class that defines the
interface for all agents in the Multi-Agent Benchmark Suite.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the benchmark suite.
    
    This class defines the interface that all agents must implement,
    ensuring consistent behavior across different agent types.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base agent.
        
        Args:
            config: Agent configuration dictionary
        """
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
        self.description = config.get('description', 'No description provided')
        self.model = None
        
        logger.info(f"🤖 Initialized {self.name}: {self.description}")
    
    @abstractmethod
    def create_model(self, env, hyperparams: Dict[str, Any], 
                    tensorboard_log: Optional[str] = None, **kwargs):
        """
        Create the model for this agent.
        
        Args:
            env: The environment to train on
            hyperparams: Hyperparameters for the model
            tensorboard_log: Path for TensorBoard logging
            **kwargs: Additional arguments
            
        Returns:
            Created model instance
        """
        pass
    
    def train(self, model, total_timesteps: int, **kwargs):
        """
        Train the agent's model.
        
        Args:
            model: The model to train
            total_timesteps: Number of timesteps to train for
            **kwargs: Additional training arguments
            
        Returns:
            Trained model
        """
        try:
            logger.info(f"🚀 Starting training for {self.name} ({total_timesteps:,} timesteps)")
            
            # Store reference to model
            self.model = model
            
            # Train the model
            trained_model = model.learn(total_timesteps=total_timesteps, **kwargs)
            
            logger.info(f"✅ Training completed for {self.name}")
            return trained_model
            
        except Exception as e:
            logger.error(f"❌ Training failed for {self.name}: {e}")
            raise
    
    def evaluate(self, model, env, n_episodes: int = 20, **kwargs) -> Dict[str, Any]:
        """
        Evaluate the agent's performance.
        
        Args:
            model: The trained model to evaluate
            env: The environment to evaluate on
            n_episodes: Number of episodes to evaluate
            **kwargs: Additional evaluation arguments
            
        Returns:
            Evaluation results dictionary
        """
        try:
            from stable_baselines3.common.evaluation import evaluate_policy
            
            logger.info(f"🔍 Evaluating {self.name} ({n_episodes} episodes)")
            
            # Evaluate the model
            mean_reward, std_reward = evaluate_policy(
                model, env, n_eval_episodes=n_episodes, **kwargs
            )
            
            results = {
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'n_episodes': n_episodes,
                'agent_name': self.name
            }
            
            logger.info(f"📊 {self.name} evaluation: {mean_reward:.2f} ± {std_reward:.2f}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed for {self.name}: {e}")
            raise
    
    def save_model(self, model, path: Union[str, Path]) -> Path:
        """
        Save the agent's model.
        
        Args:
            model: The model to save
            path: Path where to save the model
            
        Returns:
            Path to the saved model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            model.save(str(path))
            logger.info(f"💾 Saved {self.name} model to {path}")
            return path
            
        except Exception as e:
            logger.error(f"❌ Failed to save {self.name} model: {e}")
            raise
    
    def load_model(self, path: Union[str, Path], env=None):
        """
        Load a saved model.
        
        Args:
            path: Path to the saved model
            env: Environment (required for some model types)
            
        Returns:
            Loaded model instance
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        try:
            # This will be implemented by specific agent subclasses
            # since different agents use different model types
            raise NotImplementedError("load_model must be implemented by subclasses")
            
        except Exception as e:
            logger.error(f"❌ Failed to load {self.name} model: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the agent's model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'agent_name': self.name,
            'agent_type': self.__class__.__name__,
            'description': self.description,
            'config': self.config
        }
        
        if self.model is not None:
            info.update({
                'model_type': type(self.model).__name__,
                'policy_type': type(self.model.policy).__name__ if hasattr(self.model, 'policy') else 'unknown'
            })
            
            # Try to get model parameters count
            try:
                if hasattr(self.model, 'policy'):
                    total_params = sum(p.numel() for p in self.model.policy.parameters())
                    info['total_parameters'] = total_params
            except:
                info['total_parameters'] = 'unknown'
        
        return info
    
    def get_hyperparams(self) -> Dict[str, Any]:
        """
        Get the hyperparameters used by this agent.
        
        Returns:
            Dictionary of hyperparameters
        """
        return self.config.get('hyperparams', {})
    
    def set_hyperparams(self, hyperparams: Dict[str, Any]):
        """
        Set hyperparameters for this agent.
        
        Args:
            hyperparams: Dictionary of hyperparameters to set
        """
        if 'hyperparams' not in self.config:
            self.config['hyperparams'] = {}
        
        self.config['hyperparams'].update(hyperparams)
        logger.info(f"🔧 Updated hyperparameters for {self.name}")
    
    def validate_hyperparams(self, hyperparams: Dict[str, Any]) -> bool:
        """
        Validate hyperparameters for this agent.
        
        Args:
            hyperparams: Hyperparameters to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Basic validation - subclasses can override for specific validation
        required_params = ['learning_rate', 'gamma']
        
        for param in required_params:
            if param not in hyperparams:
                logger.warning(f"⚠️ Missing required parameter '{param}' for {self.name}")
                return False
        
        return True
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.name} ({self.__class__.__name__})"
    
    def __repr__(self) -> str:
        """Detailed representation of the agent."""
        return f"{self.__class__.__name__}(name='{self.name}', config={self.config})"