'''\
Module: poke_pipeline/ppo_lambda_discrepancy.py

Implementation of the λ‑Discrepancy policy and PPO extension, adapted from test_run_ld_v2.py
''' 
import os
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from sb3_contrib.ppo_recurrent.ppo_recurrent import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MultiInputLstmPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed, explained_variance
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from poke_pipeline.tensorboard_callback import TensorboardCallback
from poke_pipeline.stream_agent_wrapper import StreamWrapper
from poke_pipeline.red_gym_env_lstm import RedGymEnvLSTM

class MultiInputLstmPolicyLD(MultiInputLstmPolicy):
    """
    Extends the standard recurrent policy by adding a second value head for λ‑discrepancy.
    """
    def _build_mlp_extractor(self) -> None:
        super()._build_mlp_extractor()
        latent_dim = self.mlp_extractor.latent_dim_vf
        self.value_net_ld = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1)
        )
        for m in self.value_net_ld.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def evaluate_actions(self, obs, actions, lstm_states, episode_starts):
        # 1) run exactly the same forward that super().evaluate_actions does under the hood,
        #    but capture the *critic* latent *before* the final value head:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_feat, vf_feat = features, features
        else:
            pi_feat, vf_feat = features

        # 2) Actor sequence (we only need critic, but actor must also advance LSTM states)
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
            # feed-forward critic LSTM fallback
            latent_vf = self.critic(vf_feat)
            lstm_states_vf = lstm_states_pi

        # 4) MLP heads
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)

        # 5) main value + policy
        values = self.value_net(latent_vf)
        dist   = self._get_action_dist_from_latent(latent_pi)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()

        # 6) λ-discrepancy head
        value_ld = self.value_net_ld(latent_vf)

        # 7) return exactly what RecurrentPPOLD.train() expects:
        #    (main_value, ld_value, log_prob, entropy)
        return values, value_ld, log_prob, entropy


class RecurrentPPOLD(RecurrentPPO):
    """
    PPO variant with λ‑discrepancy auxiliary loss.
    """
    def __init__(self, *args, ld_coef=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.ld_coef = ld_coef

    def train(self) -> None:
        # Set training mode and update learning rate
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)

        # Containers for logging
        entropy_losses, pg_losses, value_losses, ld_losses = [], [], [], []

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                # --- Aktuelle Policy-Evaluation ---
                actions = rollout_data.actions.long().flatten()
                values, value_ld, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )
                values = values.flatten()
                value_ld = value_ld.flatten()

                # --- λ-Diskrepanz Targets berechnen ---
                old_values = rollout_data.old_values.flatten()      # V(s_t)
                returns    = rollout_data.returns.flatten()         # Monte Carlo Returns G_t
                eps        = rollout_data.episode_starts.float().flatten()  # 1.0 am Episodenstart

                # Maske für Nicht-Episodenstart und Shift für nächsten Schritt
                not_start      = 1.0 - eps
                next_not_start = th.cat([
                    not_start[1:],
                    th.zeros(1, device=not_start.device)
                ], dim=0)

                # G_{t+1} mit Null am Episode-Ende
                next_returns = th.cat([
                    returns[1:],
                    th.zeros(1, device=returns.device)
                ], dim=0)

                # Sofortiger Reward r_t
                rewards = returns - self.gamma * next_returns * next_not_start

                # V(s_{t+1}) mit Null am Episode-Ende
                next_values = th.cat([
                    old_values[1:],
                    th.zeros(1, device=old_values.device)
                ], dim=0)

                # TD(0)-Target: r_t + γ·V(s_{t+1})
                td0_targets = rewards + self.gamma * next_values * next_not_start

                # λ-Diskrepanz-Ziel: D_t = TD(0)_t - V(s_t)
                ld_targets = (td0_targets - old_values).detach()

                # --- Standard PPO Policy Loss ---
                advantages = rollout_data.advantages.flatten()
                ratio = th.exp(log_prob - rollout_data.old_log_prob.flatten())
                policy_loss = -th.mean(th.min(
                    advantages * ratio,
                    advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                ))

                # --- Value- und λ-Diskrepanz-Loss & Entropie ---
                vf_loss  = F.mse_loss(values, returns)
                ld_loss  = F.mse_loss(value_ld, ld_targets)
                ent_loss = -th.mean(entropy)

                # Gesamter Loss
                loss = (
                    policy_loss
                    + self.ent_coef * ent_loss
                    + self.vf_coef    * vf_loss
                    + self.ld_coef    * ld_loss
                )

                # --- Optimierungsschritt ---
                self.policy.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                # --- Logging pro Batch ---
                pg_losses.append(policy_loss.item())
                value_losses.append(vf_loss.item())
                ld_losses.append(ld_loss.item())
                entropy_losses.append(ent_loss.item())

        # --- Finale Logging-Metriken ---
        self.logger.record("train/policy_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/ld_loss", np.mean(ld_losses))
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))

        if len(ld_losses) > 0:
            with th.no_grad():
                # Durchschnittliche absolute λ-Diskrepanz
                avg_ld = th.abs(ld_targets).mean().item()
                self.logger.record("train/ld_target_abs_mean", avg_ld)
                # Vergleich TD(0) vs MC Returns
                td0_mean = td0_targets.mean().item()
                mc_mean  = returns.mean().item()
                self.logger.record("train/td0_mean", td0_mean)
                self.logger.record("train/mc_mean", mc_mean)
                self.logger.record("train/td0_vs_mc_ratio", td0_mean / (mc_mean + 1e-8))

        # Update der Update-Zählung
        self._n_updates += self.n_epochs



# Helper to create the VecEnv

def make_env(rank, env_conf, seed=0):
    def _init():
        base_env = RedGymEnvLSTM(env_conf)
        env = StreamWrapper(base_env, stream_metadata={
            "user": f"ld-v4",
            "env_id": rank,
            "color": "#447799",
            "extra": "",
        })
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

# Usage in a training script:
# env_fns = [make_env(i, cfg['env'], seed) for i in range(cfg['num_cpu'])]
# vec_env = DummyVecEnv(env_fns)
# model = RecurrentPPOLD(
#     policy=MultiInputLstmPolicyLD,
#     env=vec_env,
#     verbose=1,
#     n_steps=cfg['model']['n_steps'],
#     batch_size=cfg['model']['batch_size'],
#     n_epochs=cfg['model']['n_epochs'],
#     gamma=cfg['model']['gamma'],
#     ent_coef=cfg['model']['ent_coef'],
#     vf_coef=cfg['model']['vf_coef'],
#     ld_coef=cfg['model']['ld_coef'],
#     tensorboard_log=str(cfg['paths']['tb_log']),
# )
