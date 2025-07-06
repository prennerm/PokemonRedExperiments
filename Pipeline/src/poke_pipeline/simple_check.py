#!/usr/bin/env python3
"""
simple_check.py

Sanity Check Runner für λ-Discrepancy PPO mit dem einfachen 1×4-Grid-Environment.
Erzeugt beim Start unter `experiments/sanity_checks/simple_check/<TIMESTAMP>/tensorboard`
einen TensorBoard-Ordner und loggt dort alle Metriken.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

from poke_pipeline.simple_grid import OneByFourGrid
from poke_pipeline.ppo_lambda_discrepancy import RecurrentPPOLD, MultiInputLstmPolicyLD

def make_sanity_env(rank: int, seed: int = 0):
    """
    Erzeugt eine Funktion, die beim Aufruf genau eine Instanz
    unseres OneByFourGrid-Env zurückgibt.
    """
    def _init():
        env = OneByFourGrid()
        env.reset()
        return env
    set_random_seed(seed)
    return _init

def run_sanity_check():
    # 1) Experiments-Ordner anlegen wie bei v1–v4 (nur TensorBoard)
    base = Path("experiments") / "sanity_checks" / "simple_check"
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_root = base / ts
    dirs = {
        "root":        session_root,
        "tensorboard": session_root / "tensorboard",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print(f" Sanity Check → TensorBoard-Logs in {dirs['tensorboard']}")
    print("=" * 50)

    # 2) Vektorisierte Umgebung (hier nur 1 Env für LSTM)
    n_envs = 1
    env_fns = [make_sanity_env(i, seed=42) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)

    # 3) Modell initialisieren
    model = RecurrentPPOLD(
        policy=MultiInputLstmPolicyLD,
        env=vec_env,
        verbose=1,
        n_steps=16,           # etwas längere Rollouts für besseres Learning
        batch_size=16,
        n_epochs=4,           # mehr Epochs für besseres Learning
        gamma=0.99,
        ent_coef=0.01,        # kleiner Entropy-Loss für Exploration
        vf_coef=0.5,          # MC-Value-Loss
        ld_coef=0.1,          # λ-Discrepancy-Loss
        tensorboard_log=str(dirs["tensorboard"]),
    )

    print(f" Modell initialisiert: {type(model).__name__}")
    print(f" Environment: {type(vec_env.envs[0]).__name__}")

    # 4) Teste zuerst random policy
    print("\n🎲 Testing random policy...")
    obs = vec_env.reset()
    total_reward = 0
    episodes = 0
    
    for step in range(100):
        actions = [vec_env.action_space.sample() for _ in range(n_envs)]
        obs, rewards, dones, infos = vec_env.step(actions)
        total_reward += sum(rewards)
        
        if any(dones):
            episodes += sum(dones)
            if episodes >= 5:  # Stop after 5 episodes
                break
    
    avg_random_reward = total_reward / max(episodes, 1)
    print(f"Random policy average reward: {avg_random_reward:.2f}")

    # 5) Training
    total_timesteps = 1000  # Mehr timesteps für besseres Learning
    print(f"\n Training für {total_timesteps} timesteps...")
    
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=False,  # Progress bar deaktiviert um Dependency-Problem zu vermeiden
    )

    # 6) Teste trainierte Policy
    print("\n Testing trained policy...")
    obs = vec_env.reset()
    episode_rewards = []
    episode_lengths = []
    current_reward = 0
    current_length = 0

    for step in range(200):
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(actions)
        
        current_reward += rewards[0]
        current_length += 1
        
        if dones[0]:
            episode_rewards.append(current_reward)
            episode_lengths.append(current_length)
            current_reward = 0
            current_length = 0
            
            if len(episode_rewards) >= 10:  # Stop after 10 episodes
                break

    # 7) Analyse der Ergebnisse
    print("\n Results Analysis:")
    print("=" * 50)
    
    if episode_rewards:
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        avg_length = sum(episode_lengths) / len(episode_lengths)
        success_rate = sum(1 for r in episode_rewards if r > 0) / len(episode_rewards)
        
        print(f"Average Episode Reward: {avg_reward:.2f}")
        print(f"Average Episode Length: {avg_length:.1f}")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Best Episode Reward: {max(episode_rewards):.2f}")
        
        # Sanity check criteria
        print("\n Sanity Check Results:")
        improvement = avg_reward - avg_random_reward
        
        if avg_reward > 0.5:
            print(" PASS: Average reward > 0.5 (agent learned to reach goal)")
        else:
            print(" FAIL: Average reward <= 0.5 (agent didn't learn well)")
            
        if improvement > 0.1:
            print(f" PASS: Improvement over random: +{improvement:.2f}")
        else:
            print(f" FAIL: Little improvement over random: +{improvement:.2f}")
            
        if success_rate > 0.7:
            print(" PASS: Success rate > 70% (good consistency)")
        else:
            print(" FAIL: Success rate <= 70% (inconsistent)")
    else:
        print(" FAIL: No episodes completed during testing")

    # 8) Einzelne Episode mit Debug-Output
    print("\n🎮 Sample Episode (step-by-step):")
    print("-" * 40)
    
    env = OneByFourGrid()
    obs, _ = env.reset()
    total_reward = 0
    
    print(f"Initial state: position {obs['state'].argmax()}")
    
    for step in range(10):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        pos = obs['state'].argmax()
        action_name = 'LEFT' if action == 0 else 'RIGHT'
        print(f"Step {step+1}: Action={action_name}, Position={pos}, Reward={reward:.1f}")
        
        if done:
            print(f"Episode finished! Total reward: {total_reward:.1f}")
            break

    vec_env.close()
    print(f"\n Sanity check completed!")
    print(f" TensorBoard logs: tensorboard --logdir {dirs['tensorboard']}")

if __name__ == "__main__":
    try:
        run_sanity_check()
    except Exception as e:
        print(f" Sanity check failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)