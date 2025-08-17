#!/usr/bin/env python3
"""
Minimal Test: Nur Environment ohne Training
"""
import sys
sys.path.append('src')

from poke_pipeline.red_gym_env_lstm import RedGymEnvLSTM
import time

# Minimal config
config = {
    "session_path": "debug_test",
    "save_final_state": False,
    "print_rewards": True,
    "headless": True,
    "init_state": "data/init.state",
    "action_freq": 1,
    "max_steps": 1000,  # Sehr kurz
    "save_video": False,
    "fast_video": True,
    "gb_path": "data/PokemonRed.gb",
    "debug": False,
    "reward_scale": 0.5,
    "explore_weight": 0.25,
    "worker_rank": 0,
    "num_cpu": 1
}

print("Creating environment...")
env = RedGymEnvLSTM(config)

print("Resetting environment...")
obs = env.reset()
print(f"Reset successful, obs shape: {obs['screen'].shape if isinstance(obs, dict) else len(obs)}")

print("Testing 10 steps...")
for i in range(10):
    print(f"Step {i+1}: Starting...")
    action = env.action_space.sample()
    print(f"Step {i+1}: Action={action}, calling env.step()...")
    
    try:
        result = env.step(action)
        if len(result) == 5:  # Gymnasium API
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:  # OpenAI Gym API
            obs, reward, done, info = result
        print(f"Step {i+1}: SUCCESS - reward={reward:.3f}, done={done}")
    except Exception as e:
        print(f"Step {i+1}: ERROR - {e}")
        break
    
    if done:
        print("Episode finished, resetting...")
        obs = env.reset()
        break

print("Environment test completed successfully!")
