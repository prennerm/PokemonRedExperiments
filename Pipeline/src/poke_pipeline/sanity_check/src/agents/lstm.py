"""
LSTM Agent (v2).

This module implements the LSTM-enhanced PPO agent using sb3-contrib,
adding LSTM capabilities for environments that benefit from memory.
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecEnv

from .base import BaseAgent

logger = logging.getLogger(__name__)


class LSTMAgent(BaseAgent):
    """
    LSTM-enhanced PPO agent implementation using sb3-contrib.
    
    This agent uses LSTM layers to handle environments that require
    memory or partial observability better than standard PPO.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LSTM Agent.
        
        Args:
            config: Agent configuration dictionary
        """
        super().__init__(config)
        self.model_class = RecurrentPPO
        
        # Default hyperparameters for LSTM Agent
        self.default_hyperparams = {
            'learning_rate': 2.5e-4,
            'n_steps': 128,
            'batch_size': 256,
            'n_epochs': 4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'clip_range_vf': None,
            'normalize_advantage': True,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'target_kl': None,
            'stats_window_size': 100,
            'verbose': 1,
            # Recurrent-specific parameters
            'lstm_hidden_size': 64,
            'n_lstm_layers': 1,
            'shared_lstm': False,
            'enable_critic_lstm': True,
            'lstm_kwargs': None
        }
        
        logger.info(f"🧠 LSTM Agent initialized with LSTM support")
    
    def create_model(self, env: VecEnv, hyperparams: Dict[str, Any], 
                    tensorboard_log: Optional[str] = None, **kwargs) -> RecurrentPPO:
        """
        Create LSTM-enhanced PPO model with specified hyperparameters.
        
        Args:
            env: Vectorized environment
            hyperparams: Hyperparameters for the model
            tensorboard_log: Path for TensorBoard logging
            **kwargs: Additional arguments
            
        Returns:
            Configured RecurrentPPO model
        """
        # Merge default hyperparameters with provided ones
        model_params = self.default_hyperparams.copy()
        model_params.update(hyperparams)
        
        # Add tensorboard logging
        if tensorboard_log:
            model_params['tensorboard_log'] = tensorboard_log
        
        # Extract LSTM-specific parameters
        lstm_hidden_size = model_params.pop('lstm_hidden_size', 64)
        n_lstm_layers = model_params.pop('n_lstm_layers', 1)
        shared_lstm = model_params.pop('shared_lstm', False)
        enable_critic_lstm = model_params.pop('enable_critic_lstm', True)
        lstm_kwargs = model_params.pop('lstm_kwargs', None)
        
        # Create model
        try:
            # Determine the correct policy based on observation space
            obs_space = env.observation_space
            if hasattr(obs_space, 'spaces'):  # Dict observation space
                policy_name = "MultiInputLstmPolicy"
            else:  # Box observation space
                policy_name = "MlpLstmPolicy"
            
            model = self.model_class(
                policy=policy_name,
                env=env,
                policy_kwargs={
                    'lstm_hidden_size': lstm_hidden_size,
                    'n_lstm_layers': n_lstm_layers,
                    'shared_lstm': shared_lstm,
                    'enable_critic_lstm': enable_critic_lstm,
                    'lstm_kwargs': lstm_kwargs or {}
                },
                **model_params
            )
            
            logger.info(f"✅ Created LSTM-enhanced PPO model with policy: {policy_name}")
            logger.info(f"🧠 LSTM config: hidden_size={lstm_hidden_size}, layers={n_lstm_layers}")
            logger.info(f"📊 Key hyperparameters: lr={model_params['learning_rate']}, "
                       f"n_steps={model_params['n_steps']}, batch_size={model_params['batch_size']}")
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to create LSTM-enhanced PPO model: {e}")
            raise
    
    def load_model(self, path: Union[str, Path], env: Optional[VecEnv] = None) -> RecurrentPPO:
        """
        Load a saved LSTM-enhanced PPO model.
        
        Args:
            path: Path to the saved model
            env: Environment (optional, for compatibility)
            
        Returns:
            Loaded RecurrentPPO model
        """
        try:
            model = self.model_class.load(str(path), env=env)
            logger.info(f"✅ Loaded LSTM-enhanced PPO model from {path}")
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to load LSTM-enhanced PPO model: {e}")
            raise
    
    def validate_hyperparams(self, hyperparams: Dict[str, Any]) -> bool:
        """
        Validate hyperparameters for LSTM Agent.
        
        Args:
            hyperparams: Hyperparameters to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check required parameters
        required_params = ['learning_rate', 'n_steps', 'batch_size', 'gamma']
        for param in required_params:
            if param not in hyperparams:
                logger.warning(f"⚠️ Missing required parameter '{param}' for LSTM Agent")
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
        
        # Check LSTM-specific parameters
        if hyperparams.get('lstm_hidden_size', 0) <= 0:
            logger.warning("⚠️ lstm_hidden_size must be positive")
            return False
        
        if hyperparams.get('n_lstm_layers', 0) <= 0:
            logger.warning("⚠️ n_lstm_layers must be positive")
            return False
        
        # Check that batch_size is compatible with LSTM requirements
        batch_size = hyperparams.get('batch_size', 256)
        if batch_size < 64:
            logger.warning(f"⚠️ batch_size ({batch_size}) might be too small for LSTM training")
        
        logger.info("✅ LSTM Agent hyperparameters validated successfully")
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the LSTM Agent model.
        
        Returns:
            Dictionary with model information
        """
        info = super().get_model_info()
        
        # Add LSTM Agent specific information
        info.update({
            'algorithm': 'RecurrentPPO',
            'policy_type': 'MlpLstmPolicy',
            'framework': 'sb3-contrib',
            'supports_discrete': True,
            'supports_continuous': True,
            'supports_multidiscrete': True,
            'supports_multibox': True,
            'has_lstm': True,
            'handles_partial_observability': True,
            'memory_capable': True,
            'default_hyperparams': self.default_hyperparams
        })
        
        return info
    
    def get_environment_specific_hyperparams(self, env_name: str) -> Dict[str, Any]:
        """
        Get environment-specific hyperparameters for LSTM Agent.
        
        Args:
            env_name: Name of the environment
            
        Returns:
            Environment-specific hyperparameters
        """
        # Environment-specific tuning for LSTM Agent
        env_hyperparams = {
            'CartPole-v1': {
                'n_steps': 128,
                'batch_size': 256,
                'n_epochs': 4,
                'learning_rate': 2.5e-4,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'lstm_hidden_size': 64,
                'n_lstm_layers': 1,
                'enable_critic_lstm': True
            },
            'LunarLander-v3': {
                'n_steps': 256,
                'batch_size': 256,
                'n_epochs': 4,
                'learning_rate': 2.5e-4,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'max_grad_norm': 0.5,
                'lstm_hidden_size': 64,
                'n_lstm_layers': 1,
                'enable_critic_lstm': True
            },
            'FrozenLake-v1': {
                'n_steps': 128,
                'batch_size': 128,
                'n_epochs': 4,
                'learning_rate': 1e-4,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.1,
                'lstm_hidden_size': 32,
                'n_lstm_layers': 1,
                'enable_critic_lstm': True
            },
            'Acrobot-v1': {
                'n_steps': 256,
                'batch_size': 256,
                'n_epochs': 4,
                'learning_rate': 2.5e-4,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'max_grad_norm': 0.5,
                'lstm_hidden_size': 64,
                'n_lstm_layers': 1,
                'enable_critic_lstm': True
            },
            'Acrobot-v1-Partial': {
                'n_steps': 256,
                'batch_size': 256,
                'n_epochs': 4,
                'learning_rate': 2.5e-4,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'max_grad_norm': 0.5,
                'lstm_hidden_size': 128,  # Larger hidden size for partial observability
                'n_lstm_layers': 2,       # More layers for memory requirements
                'enable_critic_lstm': True
            }
        }
        
        return env_hyperparams.get(env_name, self.default_hyperparams)
 
    def _get_lstm_states(self, model: RecurrentPPO, env: VecEnv) -> Dict[str, Any]:
        """
        Get LSTM states for debugging or analysis.
        
        Args:
            model: Trained RecurrentPPO model
            env: Environment
            
        Returns:
            Dictionary with LSTM state information
        """
        try:
            # Get initial LSTM states
            lstm_states = model.policy.get_initial_state(env.num_envs)
            
            return {
                'lstm_states_shape': [state.shape for state in lstm_states],
                'num_lstm_layers': len(lstm_states) // 2,  # h and c states
                'lstm_hidden_size': lstm_states[0].shape[-1] if lstm_states else 0,
                'batch_size': lstm_states[0].shape[0] if lstm_states else 0
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Could not retrieve LSTM states: {e}")
            return {}
    
    def __str__(self) -> str:
        """String representation of the LSTM Agent."""
        return f"LSTM Agent (v2) with Memory"
    
    def __repr__(self) -> str:
        """Detailed representation of the LSTM Agent."""
        return f"LSTMAgent(config={self.config})"
