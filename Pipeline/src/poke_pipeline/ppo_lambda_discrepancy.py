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
        values, _, log_prob, entropy = super().evaluate_actions(obs, actions, lstm_states, episode_starts)
        # compute the auxiliary value head
        features = self.extract_features(obs)[0] if self.share_features_extractor else self.extract_features(obs)[1]
        value_ld = self.value_net_ld(features.flatten(start_dim=1))
        return values, value_ld, log_prob, entropy

class RecurrentPPOLD(RecurrentPPO):
    """
    PPO variant with λ‑discrepancy auxiliary loss.
    """
    def __init__(self, *args, ld_coef=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.ld_coef = ld_coef

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        entropy_losses, pg_losses, value_losses, ld_losses = [], [], [], []

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions.long().flatten()
                values, value_ld, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss = -th.mean(th.min(advantages * ratio,
                                              advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range))
                                       )
                value_pred = rollout_data.returns if self.clip_range_vf is None else (
                    rollout_data.old_values + th.clamp(values - rollout_data.old_values,
                                                       -self.clip_range_vf(self._current_progress_remaining),
                                                       self.clip_range_vf(self._current_progress_remaining)))
                vf_loss = F.mse_loss(values, rollout_data.returns)
                ld_loss = F.mse_loss(values, value_ld.squeeze(-1))
                ent_loss = -th.mean(entropy)
                loss = policy_loss + self.ent_coef * ent_loss + self.vf_coef * vf_loss + self.ld_coef * ld_loss

                self.policy.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                pg_losses.append(policy_loss.item())
                value_losses.append(vf_loss.item())
                ld_losses.append(ld_loss.item())
                entropy_losses.append(ent_loss.item())

        # logging
        self.logger.record("train/policy_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/ld_loss", np.mean(ld_losses))
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))

        self._n_updates += self.n_epochs

        return

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
