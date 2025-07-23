"""
Lambda-Discrepancy Agent (v3).

This module implements the Lambda-Discrepancy PPO agent based on the paper
"Mitigating Partial Observability in Sequential Decision Processes via the Lambda Discrepancy".
Uses RecurrentPPO with an additional value head for λ-discrepancy auxiliary loss.
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MultiInputLstmPolicy
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import explained_variance

from .base import BaseAgent

logger = logging.getLogger(__name__)


class MultiInputLstmPolicyLD(MultiInputLstmPolicy):
    """
    Extends the standard recurrent policy by adding a second value head for λ-discrepancy.
    
    This policy has two value heads:
    1. Standard value head for Monte Carlo returns
    2. Lambda-discrepancy head for TD(0) discrepancy targets
    """
    
    def _build_mlp_extractor(self) -> None:
        """Build the MLP extractor and add the lambda-discrepancy value head."""
        super()._build_mlp_extractor()
        latent_dim = self.mlp_extractor.latent_dim_vf
        
        # Lambda-discrepancy value head
        self.value_net_ld = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1)
        )
        
        # Initialize weights orthogonally
        for m in self.value_net_ld.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def evaluate_actions(self, obs, actions, lstm_states, episode_starts):
        """
        Evaluate actions and return both main value and lambda-discrepancy value.
        
        Returns:
            tuple: (main_values, ld_values, log_prob, entropy)
        """
        # 1) Extract features
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_feat, vf_feat = features, features
        else:
            pi_feat, vf_feat = features

        # 2) Actor sequence (needed for LSTM state advancement)
        latent_pi, lstm_states_pi = self._process_sequence(
            pi_feat, lstm_states.pi, episode_starts, self.lstm_actor
        )

        # 3) Critic sequence
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(
                vf_feat, lstm_states.vf, episode_starts, self.lstm_critic
            )
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
            lstm_states_vf = (lstm_states_pi[0].detach(),
                              lstm_states_pi[1].detach())
        else:
            # Fallback for feed-forward critic
            latent_vf = self.critic(vf_feat)
            lstm_states_vf = lstm_states_pi

        # 4) MLP heads
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        # 5) Main value and policy
        mc_values = self.value_net(latent_vf)
        dist = self._get_action_dist_from_latent(latent_pi)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        # 6) Lambda-discrepancy value head
        ld_values = self.value_net_ld(latent_vf)

        return mc_values, ld_values, log_prob, entropy


class RecurrentPPOLD(RecurrentPPO):
    """
    PPO variant with λ-discrepancy auxiliary loss.
    
    Implements the lambda-discrepancy method from the paper:
    "Mitigating Partial Observability in Sequential Decision Processes via the Lambda Discrepancy"
    """
    
    def __init__(self, *args, ld_coef=0.1, **kwargs):
        """
        Initialize RecurrentPPO with lambda-discrepancy.
        
        Args:
            ld_coef: Coefficient for lambda-discrepancy loss
        """
        super().__init__(*args, **kwargs)
        self.ld_coef = ld_coef
        logger.info(f"🔬 RecurrentPPOLD initialized with ld_coef={ld_coef}")

    def train(self) -> None:
        """
        Train the policy with lambda-discrepancy auxiliary loss.
        
        The lambda-discrepancy target is computed as:
        D_t = TD(0)_t - V(s_t) where TD(0)_t = r_t + γ·V(s_{t+1})
        """
        # Set training mode and update learning rate
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)

        # Containers for logging
        entropy_losses, pg_losses, mc_losses, ld_losses = [], [], [], []

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                # --- Current Policy Evaluation ---
                actions = rollout_data.actions.long().flatten()
                mc_values, ld_values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )
                mc_values = mc_values.flatten()
                ld_values = ld_values.flatten()

                # --- Lambda-Discrepancy Target Calculation ---
                old_values = rollout_data.old_values.flatten()      # V(s_t)
                returns = rollout_data.returns.flatten()           # Monte Carlo Returns G_t
                eps = rollout_data.episode_starts.float().flatten()  # 1.0 at episode start

                # Mask for non-episode-start and shift for next step
                not_start = 1.0 - eps
                next_not_start = th.cat([
                    not_start[1:],
                    th.zeros(1, device=not_start.device)
                ], dim=0)

                # G_{t+1} with zero at episode end
                next_returns = th.cat([
                    returns[1:],
                    th.zeros(1, device=returns.device)
                ], dim=0)

                # Immediate reward r_t
                rewards = returns - self.gamma * next_returns * next_not_start

                # V(s_{t+1}) with zero at episode end
                next_values = th.cat([
                    old_values[1:],
                    th.zeros(1, device=old_values.device)
                ], dim=0)

                # TD(0) target: r_t + γ·V(s_{t+1})
                td0_targets = rewards + self.gamma * next_values * next_not_start

                # Lambda-discrepancy target: D_t = TD(0)_t - V(s_t)
                ld_targets = (td0_targets - old_values).detach()

                # --- Standard PPO Policy Loss ---
                advantages = rollout_data.advantages.flatten()
                ratio = th.exp(log_prob - rollout_data.old_log_prob.flatten())
                policy_loss = -th.mean(th.min(
                    advantages * ratio,
                    advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                ))

                # --- MC and Lambda-Discrepancy Loss & Entropy ---
                mc_loss = F.mse_loss(mc_values, returns)
                ld_loss = F.l1_loss(ld_values, ld_targets)
                ent_loss = -th.mean(entropy)

                # Total loss
                loss = (
                    policy_loss
                    + self.ent_coef * ent_loss
                    + self.vf_coef * mc_loss
                    + self.ld_coef * ld_loss
                )

                # --- Optimization Step ---
                self.policy.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                # --- Logging per Batch ---
                pg_losses.append(policy_loss.item())
                mc_losses.append(mc_loss.item())
                ld_losses.append(ld_loss.item())
                entropy_losses.append(ent_loss.item())

        # --- Final Logging Metrics ---
        self.logger.record("train/policy_loss", np.mean(pg_losses))
        self.logger.record("train/mc_loss", np.mean(mc_losses))
        self.logger.record("train/ld_loss", np.mean(ld_losses))
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))

        if len(ld_losses) > 0:
            with th.no_grad():
                # Average absolute lambda-discrepancy
                avg_ld = th.abs(ld_targets).mean().item()
                self.logger.record("train/ld_target_abs_mean", avg_ld)
                # Bootstrap vs MC Returns comparison
                td0_mean = td0_targets.mean().item()
                mc_mean = returns.mean().item()
                self.logger.record("train/bootstrap_value_mean", td0_mean)
                self.logger.record("train/mc_value_mean", mc_mean)
                self.logger.record("train/bootstrap_vs_mc_ratio", td0_mean / (mc_mean + 1e-8))

        # Update the update count
        self._n_updates += self.n_epochs


class LDAgent(BaseAgent):
    """
    Lambda-Discrepancy Agent implementation.
    
    This agent uses RecurrentPPO with an additional lambda-discrepancy auxiliary loss
    to better handle partial observability in sequential decision processes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Lambda-Discrepancy Agent.
        
        Args:
            config: Agent configuration dictionary
        """
        super().__init__(config)
        self.model_class = RecurrentPPOLD
        
        # Default hyperparameters for Lambda-Discrepancy Agent
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
            # LSTM-specific parameters
            'lstm_hidden_size': 64,
            'n_lstm_layers': 1,
            'shared_lstm': False,
            'enable_critic_lstm': True,
            'lstm_kwargs': None,
            # Lambda-discrepancy specific parameter
            'ld_coef': 0.1
        }
        
        logger.info(f"🔬 Lambda-Discrepancy Agent initialized with LD support")
    
    def create_model(self, env: VecEnv, hyperparams: Dict[str, Any], 
                    tensorboard_log: Optional[str] = None, **kwargs) -> RecurrentPPOLD:
        """
        Create Lambda-Discrepancy PPO model with specified hyperparameters.
        
        Args:
            env: Vectorized environment
            hyperparams: Hyperparameters for the model
            tensorboard_log: Path for TensorBoard logging
            **kwargs: Additional arguments
            
        Returns:
            Configured RecurrentPPOLD model
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
            model = self.model_class(
                policy=MultiInputLstmPolicyLD,
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
            
            logger.info(f"✅ Created Lambda-Discrepancy PPO model with policy: MultiInputLstmPolicyLD")
            logger.info(f"🧠 LSTM config: hidden_size={lstm_hidden_size}, layers={n_lstm_layers}")
            logger.info(f"🔬 Lambda-Discrepancy coef: {model_params.get('ld_coef', 0.1)}")
            logger.info(f"📊 Key hyperparameters: lr={model_params['learning_rate']}, "
                       f"n_steps={model_params['n_steps']}, batch_size={model_params['batch_size']}")
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to create Lambda-Discrepancy PPO model: {e}")
            raise
    
    def load_model(self, path: Union[str, Path], env: Optional[VecEnv] = None) -> RecurrentPPOLD:
        """
        Load a saved Lambda-Discrepancy PPO model.
        
        Args:
            path: Path to the saved model
            env: Environment (optional, for compatibility)
            
        Returns:
            Loaded RecurrentPPOLD model
        """
        try:
            model = self.model_class.load(str(path), env=env)
            logger.info(f"✅ Loaded Lambda-Discrepancy PPO model from {path}")
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to load Lambda-Discrepancy PPO model: {e}")
            raise
    
    def validate_hyperparams(self, hyperparams: Dict[str, Any]) -> bool:
        """
        Validate hyperparameters for Lambda-Discrepancy Agent.
        
        Args:
            hyperparams: Hyperparameters to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check required parameters
        required_params = ['learning_rate', 'n_steps', 'batch_size', 'gamma', 'ld_coef']
        for param in required_params:
            if param not in hyperparams:
                logger.warning(f"⚠️ Missing required parameter '{param}' for Lambda-Discrepancy Agent")
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
        
        # Check lambda-discrepancy coefficient
        if hyperparams.get('ld_coef', 0) < 0:
            logger.warning("⚠️ ld_coef must be non-negative")
            return False
        
        # Check that batch_size is compatible with LSTM requirements
        batch_size = hyperparams.get('batch_size', 256)
        if batch_size < 64:
            logger.warning(f"⚠️ batch_size ({batch_size}) might be too small for LSTM training")
        
        logger.info("✅ Lambda-Discrepancy Agent hyperparameters validated successfully")
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the Lambda-Discrepancy Agent model.
        
        Returns:
            Dictionary with model information
        """
        info = super().get_model_info()
        
        # Add Lambda-Discrepancy Agent specific information
        info.update({
            'algorithm': 'RecurrentPPO-LD',
            'policy_type': 'MultiInputLstmPolicyLD',
            'framework': 'sb3-contrib + custom',
            'supports_discrete': True,
            'supports_continuous': True,
            'supports_multidiscrete': True,
            'supports_multibox': True,
            'has_lstm': True,
            'handles_partial_observability': True,
            'memory_capable': True,
            'has_lambda_discrepancy': True,
            'auxiliary_loss': 'Lambda-Discrepancy',
            'default_hyperparams': self.default_hyperparams
        })
        
        return info
    
    def get_environment_specific_hyperparams(self, env_name: str) -> Dict[str, Any]:
        """
        Get environment-specific hyperparameters for Lambda-Discrepancy Agent.
        
        Args:
            env_name: Name of the environment
            
        Returns:
            Environment-specific hyperparameters
        """
        # Environment-specific tuning for Lambda-Discrepancy Agent
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
                'enable_critic_lstm': True,
                'ld_coef': 0.05  # Lower for simple environment
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
                'enable_critic_lstm': True,
                'ld_coef': 0.1  # Standard for complex environment
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
                'enable_critic_lstm': True,
                'ld_coef': 0.2  # Higher for stochastic environment
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
                'enable_critic_lstm': True,
                'ld_coef': 0.1  # Standard for complex dynamics
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
                'enable_critic_lstm': True,
                'ld_coef': 0.3  # Higher coefficient for partial observability challenges
            }
        }
        
        return env_hyperparams.get(env_name, self.default_hyperparams)
    
    def __str__(self) -> str:
        """String representation of the Lambda-Discrepancy Agent."""
        return f"Lambda-Discrepancy Agent (v3) with LSTM + LD-Loss"
    
    def __repr__(self) -> str:
        """Detailed representation of the Lambda-Discrepancy Agent."""
        return f"LDAgent(config={self.config})"
