"""
Standard PPO Agent (Baseline).

This module implements the standard PPO agent using stable-baselines3,
serving as the baseline for comparison in the Multi-Agent Benchmark Suite.
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

from .base import BaseAgent

logger = logging.getLogger(__name__)


class StandardPPOAgent(BaseAgent):
    """
    Standard PPO agent implementation using stable-baselines3.
    
    This agent serves as the baseline for comparison against more advanced
    variants (RecurrentPPO, LSTM, Lambda-Discrepancy).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Standard PPO agent.
        
        Args:
            config: Agent configuration dictionary
        """
        super().__init__(config)
        self.model_class = PPO
        
        # Default hyperparameters for Standard PPO
        self.default_hyperparams = {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'clip_range_vf': None,
            'normalize_advantage': True,
            'ent_coef': 0.0,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'target_kl': None,
            'stats_window_size': 100,
            'verbose': 1
        }
        
        logger.info(f"🤖 Standard PPO Agent initialized")
    
    def create_model(self, env: VecEnv, hyperparams: Dict[str, Any], 
                    tensorboard_log: Optional[str] = None, **kwargs) -> PPO:
        """
        Create PPO model with specified hyperparameters.
        
        Args:
            env: Vectorized environment
            hyperparams: Hyperparameters for the model
            tensorboard_log: Path for TensorBoard logging
            **kwargs: Additional arguments
            
        Returns:
            Configured PPO model
        """
        # Merge default hyperparameters with provided ones
        model_params = self.default_hyperparams.copy()
        model_params.update(hyperparams)
        
        # Add tensorboard logging
        if tensorboard_log:
            model_params['tensorboard_log'] = tensorboard_log
        
        # Create model
        try:
            model = self.model_class(
                policy="MultiInputPolicy",
                env=env,
                **model_params
            )
            
            logger.info(f"✅ Created Standard PPO model with policy: MultiInputPolicy")
            logger.info(f"📊 Key hyperparameters: lr={model_params['learning_rate']}, "
                       f"n_steps={model_params['n_steps']}, batch_size={model_params['batch_size']}")
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to create Standard PPO model: {e}")
            raise
    
    def load_model(self, path: Union[str, Path], env: Optional[VecEnv] = None) -> PPO:
        """
        Load a saved Standard PPO model.
        
        Args:
            path: Path to the saved model
            env: Environment (optional, for compatibility)
            
        Returns:
            Loaded PPO model
        """
        try:
            model = self.model_class.load(str(path), env=env)
            logger.info(f"✅ Loaded Standard PPO model from {path}")
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to load Standard PPO model: {e}")
            raise
    
    def validate_hyperparams(self, hyperparams: Dict[str, Any]) -> bool:
        """
        Validate hyperparameters for Standard PPO.
        
        Args:
            hyperparams: Hyperparameters to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check required parameters
        required_params = ['learning_rate', 'n_steps', 'batch_size', 'gamma']
        for param in required_params:
            if param not in hyperparams:
                logger.warning(f"⚠️ Missing required parameter '{param}' for Standard PPO")
                return False
        
        # Check parameter ranges
        if hyperparams.get('learning_rate', 0) <= 0:
            logger.warning("⚠️ learning_rate must be positive")
            return False
        
        if hyperparams.get('n_steps', 0) <= 0:
            logger.warning("⚠️ n_steps must be positive")
            return False
        
        if hyperparams.get('batch_size', 0) <= 0:
            logger.warning("⚠️ batch_size must be positive")
            return False
        
        if not 0 <= hyperparams.get('gamma', 0) <= 1:
            logger.warning("⚠️ gamma must be between 0 and 1")
            return False
        
        # Check that batch_size divides n_steps evenly
        n_steps = hyperparams.get('n_steps', 2048)
        batch_size = hyperparams.get('batch_size', 64)
        
        if n_steps % batch_size != 0:
            logger.warning(f"⚠️ n_steps ({n_steps}) should be divisible by batch_size ({batch_size})")
            return False
        
        logger.info("✅ Standard PPO hyperparameters validated successfully")
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the Standard PPO model.
        
        Returns:
            Dictionary with model information
        """
        info = super().get_model_info()
        
        # Add Standard PPO specific information
        info.update({
            'algorithm': 'PPO',
            'policy_type': 'MultiInputPolicy',
            'framework': 'stable-baselines3',
            'supports_discrete': True,
            'supports_continuous': True,
            'supports_multidiscrete': True,
            'supports_multibox': True,
            'default_hyperparams': self.default_hyperparams
        })
        
        return info
    
    def get_environment_specific_hyperparams(self, env_name: str) -> Dict[str, Any]:
        """
        Get environment-specific hyperparameters for Standard PPO.
        
        Args:
            env_name: Name of the environment
            
        Returns:
            Environment-specific hyperparameters
        """
        # Environment-specific tuning based on common practices
        env_hyperparams = {
            'CartPole-v1': {
                'n_steps': 512,
                'batch_size': 256,
                'n_epochs': 4,
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01
            },
            'MountainCar-v0': {
                'n_steps': 2048,
                'batch_size': 256,
                'n_epochs': 3,
                'learning_rate': 1e-3,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.1
            },
            'LunarLander-v3': {
                'n_steps': 2048,
                'batch_size': 512,
                'n_epochs': 4,
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.1,
                'max_grad_norm': 0.5
            },
            'FrozenLake-v1': {
                'n_steps': 128,
                'batch_size': 64,
                'n_epochs': 4,
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.1
            }
        }
        
        return env_hyperparams.get(env_name, self.default_hyperparams)
    
    def __str__(self) -> str:
        """String representation of the Standard PPO agent."""
        return f"Standard PPO Agent (Baseline)"
    
    def __repr__(self) -> str:
        """Detailed representation of the Standard PPO agent."""
        return f"StandardPPOAgent(config={self.config})"