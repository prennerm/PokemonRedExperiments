import sys
import uuid
import numpy as np
from os.path import exists
from pathlib import Path
from red_gym_env_v2_adapted import RedGymEnv
from stream_agent_wrapper import StreamWrapper
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from tensorboard_callback import TensorboardCallback
import json
import time

class StatsCallback(BaseCallback):
    """
    Callback zum Sammeln und Speichern von Trainingsstatistiken.
    Er fragt am Ende eines Rollouts (oder jeder Episode) die in den Environments
    gesammelten agent_stats ab und speichert sie in einer JSON-Datei.
    """
    def __init__(self, save_freq: int, save_path: str, verbose=0):
        """
        :param save_freq: Anzahl der gesammelten Einträge, ab denen gespeichert wird.
        :param save_path: Ordner, in dem die JSON-Datei(en) gespeichert werden.
        """
        super(StatsCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.all_stats = []  # Aggregierte Statistiken

    def _on_step(self) -> bool:
        return True
        

    def _on_rollout_end(self) -> None:
        # Hole die agent_stats von allen Sub-Umgebungen
        stats_lists = self.training_env.get_attr("agent_stats")
        # stats_lists ist eine Liste – ein Element pro Subenv (jede ist eine Liste von Dictionaires)
        for env_stats in stats_lists:
            if env_stats:  # wenn nicht leer
                self.all_stats.extend(env_stats)
                # Optional: die Statistik im Environment zurücksetzen, um keine Duplikate zu sammeln
                self.training_env.set_attr("agent_stats", [])
        
        # Falls genügend Einträge vorhanden sind, speichere in eine Datei:
        if len(self.all_stats) >= self.save_freq:
            file_name = self.save_path / f"all_runs_{int(time.time())}.json"
            with open(file_name, "w") as f:
                json.dump(self.all_stats, f, default=lambda o: int(o) if isinstance(o, np.int64) else o)
            if self.verbose:
                print(f"Saved stats to {file_name} (total entries: {len(self.all_stats)})")
            self.all_stats = []  # Zurücksetzen der gesammelten Daten

    def _on_training_end(self) -> None:
        # Am Ende des Trainings – falls noch Daten vorhanden sind, diese speichern
        if self.all_stats:
            file_name = self.save_path / f"all_runs_{int(time.time())}.json"
            with open(file_name, "w") as f:
                json.dump(self.all_stats, f, default=lambda o: int(o) if isinstance(o, np.int64) else o)
            if self.verbose:
                print(f"Final stats saved to {file_name}")



def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = RedGymEnv(env_conf)
            #stream_metadata = { # All of this is part is optional
            #    "user": "v2-default", # choose your own username
            #    "env_id": rank, # environment identifier
            #    "color": "#447799", # choose your color :)
            #    "extra": "", # any extra text you put here will be displayed
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":

    use_wandb_logging = False
    ep_length = 2048 * 80 #von 80 auf 8 geändert
    sess_id = "test_sessions/session_v2"
    sess_path = Path(sess_id)
    sess_path.mkdir(parents=True, exist_ok=True)

    stats_dir = sess_path / "json_logs"
    stats_dir.mkdir(parents=True, exist_ok=True)

    env_config = {
                'headless': True, 'save_final_state': False, 'early_stop': False,
                'action_freq': 24, 'init_state': '../has_pokedex_nballs.state', 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': '../PokemonRed.gb', 'debug': False, 'reward_scale': 0.5, 'explore_weight': 0.25
            }
    
    print(env_config)
    
    num_cpu = 32 # Also sets the number of episodes per training iteration, default 64
    print("Erstelle Umgebungen...")
    envs = [make_env(i, env_config) for i in range(num_cpu)]
    print("Umgebungen erstellt.")
    env = SubprocVecEnv(envs)
    print("SubprocVecEnv erfolgreich initialisiert.")

    
    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path,
                                     name_prefix="poke")
    tensorboard_callback = TensorboardCallback(sess_path)
    stats_callback = StatsCallback(save_freq=100, save_path=str(stats_dir), verbose=1)
    
    callbacks = [checkpoint_callback, TensorboardCallback(sess_path), stats_callback]

    if use_wandb_logging:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        wandb.tensorboard.patch(root_logdir=str(sess_path))
        run = wandb.init(
            project="pokemon-train",
            id=sess_id,
            name="v2-a",
            config=env_config,
            sync_tensorboard=True,  
            monitor_gym=True,  
            save_code=True,
        )
        callbacks.append(WandbCallback())

    #env_checker.check_env(env)

    # put a checkpoint here you want to start from
    file_name = "" #"runs/poke_26214400_steps"

    train_steps_batch = ep_length // 64
    
    if exists(file_name + ".zip"):
        print("\nloading checkpoint")
        model = PPO.load(file_name, env=env)
        model.n_steps = train_steps_batch
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = train_steps_batch
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        model = PPO("MultiInputPolicy", env, verbose=1, n_steps=train_steps_batch, batch_size=512, n_epochs=1, gamma=0.997, ent_coef=0.01, tensorboard_log=sess_path)
    
    #print(model.policy)

    try:
    #model.learn(total_timesteps=(ep_length)*num_cpu*1000, callback=CallbackList(callbacks), tb_log_name="poke_ppo")
        model.learn(total_timesteps=100_000_000, callback=CallbackList(callbacks), tb_log_name="poke_ppo")
    except KeyboardInterrupt:
        print("Training interrupted. Saving remaining stats...")
        for cb in callbacks:
            if hasattr(cb, "on_training_end"):
                cb._on_training_end()
        raise

    if use_wandb_logging:
        run.finish()
