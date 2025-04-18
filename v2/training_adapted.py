import sys
import uuid
from datetime import datetime
from os.path import exists
from pathlib import Path
from red_gym_env_v2_adapted import RedGymEnv
from stream_agent_wrapper import StreamWrapper
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback
from callbacks import StatsCallback

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

    #anpassen damit die einzelnen trainings und die jeweiligen json logs sauber getrennt werden

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_path = sess_path / run_name
    json_dir = run_path / "json_logs"
    zip_dir  = run_path / "checkpoints"
    tb_dir   = run_path / "tensorboard"
    # erstelle: run_path selbst und alle drei Sub‑Ordner
    for d in (run_path, json_dir, zip_dir, tb_dir):
        d.mkdir(parents=True, exist_ok=True)

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

    
    checkpoint_callback = CheckpointCallback(
        save_freq=ep_length,
        save_path=str(zip_dir),
        name_prefix="poke"
    )
    tensorboard_callback = TensorboardCallback(str(tb_dir))
    stats_callback = StatsCallback(
        save_freq=100,
        save_path=str(json_dir),
        verbose=1
    )
    
    callbacks = [checkpoint_callback, tensorboard_callback, stats_callback]

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

    #model.learn(total_timesteps=(ep_length)*num_cpu*1000, callback=CallbackList(callbacks), tb_log_name="poke_ppo")
    try:
        model.learn(total_timesteps=100_000_000, callback=CallbackList(callbacks), tb_log_name="poke_ppo")
    except KeyboardInterrupt:
        print("Training interrupted. Saving remaining stats...")
        for cb in callbacks:
            if hasattr(cb, "_on_training_end"):
                cb._on_training_end()
        raise

    if use_wandb_logging:
        run.finish()
