#!/usr/bin/env python3
"""
test_run_ld.py

Beispielhaftes Trainingsskript für einen PPO-Agenten, der
eine zusätzliche λ‑Discrepancy (LD) als Auxiliary Loss verwendet.
Die Implementierung basiert auf sb3_contrib und nutzt rekurrente Policies.
"""

import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from sb3_contrib.ppo_recurrent.ppo_recurrent import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MultiInputLstmPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback  # Dein eigener Tensorboard-Callback
from red_gym_env_lstm import RedGymEnvLSTM  # Dein Environment

##########################################################
# 1. Custom Policy: MultiInputLstmPolicyLD
##########################################################

class MultiInputLstmPolicyLD(MultiInputLstmPolicy):
    """
    Custom recurrent policy, die einen zusätzlichen Value-Head
    (value_net_ld) implementiert, um die λ-Discrepancy zu berechnen.
    
    Die latente Dimension wird dabei über den LSTM-Layer bestimmt,
    d.h. über `self.lstm_actor.hidden_size`.
    """
    def __init__(self, *args, **kwargs):
        super(MultiInputLstmPolicyLD, self).__init__(*args, **kwargs)
        # Keine direkte Initialisierung von value_net_ld hier!
    
    def _build_mlp_extractor(self) -> None:
        # Zuerst den MLP-Extractor der Basisklasse bauen, der latent_dim_vf setzt:
        super()._build_mlp_extractor()
        # Nun die latente Dimension aus dem MLP-Extractor verwenden:
        latent_dim = self.mlp_extractor.latent_dim_vf
        self.value_net_ld = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1)
        )
        # Optional: Initialisierung der Gewichte des zusätzlichen Value-Heads:
        for m in self.value_net_ld.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor, lstm_states, episode_starts: th.Tensor):
        """
        Überschreibt evaluate_actions, um neben dem Standardwert
        auch einen zusätzlichen Value (value_ld) zu berechnen.
        
        :return: values, value_ld, log_prob, entropy
        """
        # Extrahiere Features (über den Features-Extractor)
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features

        # LSTM-Pass für die Policy (Actor)
        latent_pi, _ = self._process_sequence(pi_features, lstm_states.pi, episode_starts, self.lstm_actor)
        # LSTM-Pass für den Critic:
        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(vf_features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(vf_features)

        # Weiterleitung über den MLP-Extractor
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        value_ld = self.value_net_ld(latent_vf)
        entropy = distribution.entropy()
        return values, value_ld, log_prob, entropy

##########################################################
# 2. Custom PPO: RecurrentPPOLD
##########################################################

class RecurrentPPOLD(RecurrentPPO):
    """
    Erweiterte Version von RecurrentPPO, die einen
    zusätzlichen λ‑Discrepancy Loss (MSE zwischen Standardwert und extra Value)
    in den Gesamtverlust integriert.
    
    Parameter:
      - ld_coef: Gewichtung des λ‑Discrepancy Loss.
    """
    def __init__(self, *args, ld_coef=0.1, **kwargs):
        super(RecurrentPPOLD, self).__init__(*args, **kwargs)
        self.ld_coef = ld_coef

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses = []
        value_losses = []
        ld_losses = []
        clip_fractions = []
        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                # Bei diskreten Aktionen sicherstellen, dass sie vom richtigen Typ sind:
                if self.action_space.__class__.__name__ == "Discrete":
                    actions = actions.long().flatten()
                mask = rollout_data.mask > 1e-8

                # Verwende die custom evaluate_actions-Methode, die (values, value_ld, log_prob, entropy) liefert
                values, value_ld, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.mean(th.min(policy_loss_1, policy_loss_2)[mask])
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()[mask]).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = th.mean(((rollout_data.returns - values_pred) ** 2)[mask])
                value_losses.append(value_loss.item())

                # Berechne den λ‑Discrepancy Loss (MSE zwischen Standardwert und extra Value)
                ld_loss = F.mse_loss(values, value_ld.squeeze(-1))
                ld_losses.append(ld_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob[mask])
                else:
                    entropy_loss = -th.mean(entropy[mask])
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.ld_coef * ld_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(((th.exp(log_ratio) - 1) - log_ratio)[mask]).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at epoch {epoch} due to KL divergence: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", float(np.mean(entropy_losses)))
        self.logger.record("train/policy_gradient_loss", float(np.mean(pg_losses)))
        self.logger.record("train/value_loss", float(np.mean(value_losses)))
        self.logger.record("train/ld_loss", float(np.mean(ld_losses)))
        self.logger.record("train/approx_kl", float(np.mean(approx_kl_divs)))
        self.logger.record("train/clip_fraction", float(np.mean(clip_fractions)))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

##########################################################
# 3. Hilfsfunktion zur Erstellung der Umgebung
##########################################################

def make_env(rank, env_conf, seed=0):
    def _init():
        env = RedGymEnvLSTM(env_conf)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

##########################################################
# 4. Hauptprogramm: Setup, Modelldefinition und Training
##########################################################

if __name__ == "__main__":
    use_wandb_logging = False
    ep_length = 2048 * 8  # Beispiel: Anzahl der Steps pro Episode
    sess_id = "test_sessions/session_ld"
    sess_path = Path(sess_id)
    sess_path.mkdir(parents=True, exist_ok=True)

    env_config = {
        'headless': True,
        'save_final_state': False,
        'early_stop': False,
        'action_freq': 24,
        'init_state': '../has_pokedex_nballs.state',
        'max_steps': ep_length,
        'print_rewards': True,
        'save_video': False,
        'fast_video': True,
        'session_path': sess_path,
        'gb_path': '../PokemonRed.gb',
        'debug': False,
        'reward_scale': 0.5,
        'explore_weight': 0.25
    }

    num_cpu = 8
    env = DummyVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path, name_prefix="poke_ld")
    callbacks = [checkpoint_callback, TensorboardCallback(str(sess_path))]

    if use_wandb_logging:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        wandb.tensorboard.patch(root_logdir=str(sess_path))
        run = wandb.init(
            project="pokemon-train",
            id=sess_id,
            name="v2-ld",
            config=env_config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        callbacks.append(WandbCallback())

    file_name = ""  # Pfad zu einem Checkpoint, falls gewünscht
    train_steps_batch = ep_length // 64

    if file_name and os.path.exists(file_name + ".zip"):
        print("\nLade Checkpoint...")
        model = RecurrentPPOLD.load(file_name, env=env, custom_objects={"policy": MultiInputLstmPolicyLD})
        model.n_steps = train_steps_batch
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = train_steps_batch
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        model = RecurrentPPOLD(
            policy=MultiInputLstmPolicyLD,
            env=env,
            verbose=1,
            n_steps=train_steps_batch,
            batch_size=512,
            n_epochs=1,
            gamma=0.997,
            ent_coef=0.01,
            tensorboard_log=str(sess_path),
            ld_coef=0.1  # Hyperparameter: Gewichtung des λ-Discrepancy Loss
        )

    total_timesteps = ep_length * num_cpu * 10000
    model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks), tb_log_name="poke_ppo_ld")

    if use_wandb_logging:
        run.finish()
