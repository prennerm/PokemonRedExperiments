#!/usr/bin/env python3
"""
test_run_ld.py

Beispielhaftes Trainingsskript für einen PPO-Agenten, der neben einem LSTM‐Modul
zwei Value‑Heads (mit unterschiedlichen TD(λ) Schätzungen) nutzt. Der
zusätzliche „Lambda Discrepancy“-Loss (MSE zwischen den beiden Value-Schätzungen)
wird als Auxilliary Loss in die Gesamtverlustfunktion integriert.
"""

import sys
import uuid
from os.path import exists
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from red_gym_env_lstm import RedGymEnvLSTM
from stream_agent_wrapper import StreamWrapper  # falls benötigt
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MultiInputLstmPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback


#############################################
# 1. Benutzerdefinierte Policy mit zwei Value-Heads
#############################################

class MultiInputLstmPolicyLD(MultiInputLstmPolicy):
    """
    Diese Policy erweitert die Standard-RecurrentActorCriticPolicy,
    indem sie einen zweiten Value-Head hinzufügt, der für die λ-Discrepancy
    Berechnung verwendet wird.
    """
    def __init__(self, *args, **kwargs):
        super(MultiInputLstmPolicyLD, self).__init__(*args, **kwargs)
        # Annahme: self.lstm_hidden_dim wird in der Basisklasse definiert.
        self.value_net_ld = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, self.lstm_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.lstm_hidden_dim, 1)
        )
        # Initialisiere die Gewichte des zusätzlichen Value-Head
        for m in self.value_net_ld.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, obs, lstm_states, dones):
        """
        Gibt Aktion, den Standard-Value und den zusätzlichen λ-Value sowie
        die aktualisierten LSTM-Zustände zurück.
        """
        # Berechne latente Features (wie in der Basisklasse)
        features, lstm_states = self._get_latent(obs, lstm_states, dones)
        distribution = self._get_action_dist_from_latent(features)
        value = self.value_net(features)         # Standard TD(λ) (z. B. λ = 0)
        value_ld = self.value_net_ld(features)     # Alternativer TD(λ)-Schätzer (z. B. λ = 1)
        return distribution, value, value_ld, lstm_states

    def evaluate_actions(self, obs, lstm_states, actions, dones):
        """
        Wird im Trainingsschritt aufgerufen, um die Log-Wahrscheinlichkeiten,
        den Value, den zusätzlichen Value (für LD) und die Entropie zu berechnen.
        """
        features, lstm_states = self._get_latent(obs, lstm_states, dones)
        distribution = self._get_action_dist_from_latent(features)
        value = self.value_net(features)
        value_ld = self.value_net_ld(features)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return value, value_ld, log_prob, entropy, lstm_states

    def _predict(self, observation, deterministic=False):
        # Verwende die neue forward-Methode, um eine Aktion vorherzusagen.
        lstm_states = self.initial_state
        # Beachte: 'dones' wird hier als Tensor mit Nullen übergeben.
        dones = torch.zeros((observation.shape[0],), device=observation.device)
        distribution, value, value_ld, _ = self.forward(observation, lstm_states, dones)
        action = distribution.get_actions(deterministic=deterministic)
        return action


#############################################
# 2. Angepasste PPO-Klasse, die den λ-Discrepancy Loss integriert
#############################################

class RecurrentPPOLD(RecurrentPPO):
    """
    Diese Klasse erweitert RecurrentPPO, indem sie im Trainingsschritt einen
    zusätzlichen Loss-Term (den λ-Discrepancy Loss) hinzufügt.
    """
    def __init__(self, *args, ld_coef=1.0, **kwargs):
        super(RecurrentPPOLD, self).__init__(*args, **kwargs)
        self.ld_coef = ld_coef

    def train(self):
        """
        Überschreibe die train()-Methode, um neben dem üblichen PPO-Loss
        auch den MSE-Loss zwischen den beiden Value-Schätzungen (λ-Discrepancy) zu minimieren.
        Diese Implementierung basiert – vereinfacht – auf der Standard-Trainingsschleife.
        """
        self.policy.set_training_mode(True)
        # Hole die Daten aus dem Rollout-Buffer (Annahme: Buffer liefert Mini-Batches)
        rollout_data = self.rollout_buffer.get(self.batch_size)
        # Initialisiere Verlustakkumulatoren
        policy_loss = 0
        value_loss = 0
        entropy_loss = 0
        ld_loss = 0

        for rollout in rollout_data:
            # Entpacke Rollout-Daten (Beispielhafte Namen; passe diese ggf. an Deine Buffer-Struktur an)
            obs = rollout.observations
            actions = rollout.actions
            old_values = rollout.old_values
            old_log_probs = rollout.old_log_probs
            advantages = rollout.advantages
            returns = rollout.returns
            lstm_states = rollout.lstm_states
            dones = rollout.dones

            # Berechne die aktuelle Policy-Ausgabe
            value, value_ld, log_prob, entropy, _ = self.policy.evaluate_actions(obs, lstm_states, actions, dones)
            # PPO-Ratio
            ratio = torch.exp(log_prob - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
            curr_policy_loss = -torch.min(surr1, surr2).mean()
            curr_value_loss = F.mse_loss(value, returns)
            # Lambda Discrepancy Loss: MSE zwischen den beiden Value-Schätzungen
            curr_ld_loss = F.mse_loss(value, value_ld)
            curr_entropy_loss = -entropy.mean()

            policy_loss += curr_policy_loss
            value_loss += curr_value_loss
            ld_loss += curr_ld_loss
            entropy_loss += curr_entropy_loss

        # Durchschnitt über alle Mini-Batches
        n_batches = len(rollout_data)
        policy_loss /= n_batches
        value_loss /= n_batches
        ld_loss /= n_batches
        entropy_loss /= n_batches

        # Gesamtverlust: Neben dem Standardwertverlust fließt der λ-Discrepancy Loss mit Faktor ld_coef ein
        loss = policy_loss + self.vf_coef * value_loss + self.ld_coef * ld_loss + self.ent_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Logging der Verlustwerte
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        self.logger.record("train/ld_loss", ld_loss.item())
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/total_loss", loss.item())


#############################################
# 3. Hilfsfunktion zur Erstellung der Umgebung
#############################################

def make_env(rank, env_conf, seed=0):
    """
    Erzeugt eine Umgebung – ähnlich wie in Deinem bisherigen Setup.
    """
    def _init():
        env = RedGymEnvLSTM(env_conf)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init


#############################################
# 4. Hauptprogramm: Umgebung, Modelldefinition und Training
#############################################

if __name__ == "__main__":

    use_wandb_logging = False
    ep_length = 2048 * 8  # wie bisher
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
    print("Erstelle Umgebungen...")
    env = DummyVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    print("DummyVecEnv erfolgreich initialisiert.")

    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path,
                                               name_prefix="poke_ld")
    callbacks = [checkpoint_callback, TensorboardCallback(sess_path)]

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

    # Falls ein Checkpoint vorhanden ist, laden – ansonsten neues Modell instanziieren.
    file_name = ""  # Hier den Pfad zu einem Checkpoint eintragen, falls gewünscht.
    train_steps_batch = ep_length // 64

    if exists(file_name + ".zip"):
        print("\nloading checkpoint")
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
            tensorboard_log=sess_path,
            ld_coef=0.1  # Hyperparameter: Gewichtung des λ-Discrepancy Loss
        )

    print(model.policy)

    total_timesteps = ep_length * num_cpu * 10000
    model.learn(total_timesteps=total_timesteps,
                callback=CallbackList(callbacks),
                tb_log_name="poke_ppo_ld")

    if use_wandb_logging:
        run.finish()
