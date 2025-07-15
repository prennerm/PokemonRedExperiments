#!/usr/bin/env python3
"""
Advanced Sanity Check Suite für λ-Discrepancy PPO

Testet verschiedene Aspekte in aufsteigender Komplexität:
1. FrozenLake-4x4: Discrete, stochastic, sparse rewards
2. CartPole-v1: Continuous observations, dense rewards, control task
3. LunarLander-v2: Complex dynamics, shaped rewards, landing task
4. MountainCar-v0: Sparse rewards, momentum-based, exploration challenge

Jedes Environment testet verschiedene Aspekte:
- Memory requirements (LSTM)
- λ-Discrepancy effectiveness
- Exploration vs Exploitation
- Reward structure handling
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from poke_pipeline.ppo_lambda_discrepancy import RecurrentPPOLD, MultiInputLstmPolicyLD

class AdvancedSanityCheck:
    """
    Comprehensive sanity check suite for λ-Discrepancy PPO
    """
    
    def __init__(self, base_dir: str = "experiments/sanity_checks/advanced"):
        self.base_dir = Path(base_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_root = self.base_dir / self.timestamp
        
        # Test environments with expected performance thresholds
        self.test_configs = {
            "FrozenLake-v1": {
                "env_id": "FrozenLake-v1",
                "env_kwargs": {"is_slippery": False, "render_mode": None},
                "expected_random_reward": 0.1,  # Random policy success rate
                "success_threshold": 0.8,       # Minimum success rate to pass
                "timesteps": 15000,
                "wrapper": "discrete_dict",     # Convert to dict observation
                "description": "Tests sparse rewards + stochastic transitions"
            },
            "CartPole-v1": {
                "env_id": "CartPole-v1",
                "env_kwargs": {},
                "expected_random_reward": 20,   # Random policy average
                "success_threshold": 120,       # Near-optimal performance
                "timesteps": 200000,
                "wrapper": "continuous_dict",   # Convert to dict observation  
                "description": "Tests continuous obs + dense rewards + control"
            },
            "MountainCar-v0": {
                "env_id": "MountainCar-v0", 
                "env_kwargs": {},
                "expected_random_reward": -200, # Random policy (timeout)
                "success_threshold": -110,
                "timesteps": 300000,
                "wrapper": "continuous_dict",
                "description": "Tests sparse rewards + exploration + momentum"
            },
            "LunarLander-v3": {
                "env_id": "LunarLander-v3",
                "env_kwargs": {},
                "expected_random_reward": -150, # Random policy crashes
                "success_threshold": 150,       # Successful landing
                "timesteps": 400000,
                "wrapper": "continuous_dict", 
                "description": "Tests complex dynamics + shaped rewards"
            }
        }
        
        self.results = {}
        self.setup_directories()
    
    def get_baseline_configs(self):
        """Get configurations for standard PPO baseline"""
        baseline_configs = {}
        
        for env_name, config in self.test_configs.items():
            # Kopiere Original-Config
            baseline_config = config.copy()
            baseline_config['description'] = f"Standard PPO baseline for {env_name}"
            baseline_config['variant'] = 'baseline'
            baseline_configs[f"{env_name}_baseline"] = baseline_config
        
        return baseline_configs

    def setup_directories(self):
        """Create directory structure for results"""
        dirs = ["logs", "tensorboard", "models", "plots"]
        for d in dirs:
            (self.session_root / d).mkdir(parents=True, exist_ok=True)
    
    def create_dict_wrapper(self, env, wrapper_type: str):
        """Wrap environment to provide dict observations for our policy"""
        class DictObsWrapper(gym.ObservationWrapper):
            def __init__(self, env):
                super().__init__(env)
                if wrapper_type == "discrete_dict":
                    # For discrete environments like FrozenLake
                    obs_dim = env.observation_space.n
                    self.observation_space = gym.spaces.Dict({
                        "state": gym.spaces.Box(0, 1, shape=(obs_dim,), dtype=np.float32)
                    })
                else:  # continuous_dict
                    # For continuous environments
                    self.observation_space = gym.spaces.Dict({
                        "state": env.observation_space
                    })
            
            def observation(self, obs):
                if wrapper_type == "discrete_dict":
                    # Convert discrete state to one-hot
                    one_hot = np.zeros(self.env.observation_space.n, dtype=np.float32)
                    one_hot[obs] = 1.0
                    return {"state": one_hot}
                else:
                    # Wrap continuous observation
                    return {"state": obs}
        
            def step(self, action):
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                # MountainCar Erfolgs-Bonus
                if self.env.spec.id == "MountainCar-v0":
                    position, velocity = obs[0], obs[1]
                    
                    # Erfolgs-Bonus (massiv erhöht)
                    if terminated and position >= 0.5:
                        reward += 500.0  # Großer Erfolgs-Bonus
                        print(f"SUCCESS! Position: {position:.3f}, Velocity: {velocity:.3f}, Bonus: +500.0")

                    # Initialisiere max_position falls nicht vorhanden
                    if not hasattr(self, 'max_position'):
                        self.max_position = -1.2  # Startposition
                        self.episode_step = 0
                    
                    # Episode Step Counter
                    self.episode_step += 1
                    
                    # Höchster Punkt Bonus
                    if position > self.max_position:
                        height_improvement = position - self.max_position
                        high_point_bonus = 50.0 * height_improvement  # Skaliert mit Verbesserung
                        self.max_position = position
                        reward += high_point_bonus
                        #print(f"NEW HIGH! Position: {position:.3f}, Previous: {self.max_position - height_improvement:.3f}, Bonus: +{high_point_bonus:.1f}")

                    # Standard Reward Shaping
                    height_bonus = 0.1 * (position + 1.2)
                    momentum_bonus = 0.05 * abs(velocity)

                    if velocity > 0:
                        momentum_bonus *= 2
                    
                    reward += height_bonus + momentum_bonus
                    
                    # Reset bei Episode Ende
                    if terminated or truncated:
                        self.max_position = -1.2
                        self.episode_step = 0

                elif self.env.spec.id == "LunarLander-v3":
                    # LunarLander Reward Shaping
                    x, y, vx, vy, angle, angular_vel, leg1, leg2 = obs
                    
                    # Landing bonus (näher zum Landeplatz = besser)
                    distance_to_target = abs(x)  # x=0 ist optimal
                    proximity_bonus = max(0, 0.5 - distance_to_target)  # Bonus für Nähe
                    
                    # Stability bonus (geringere Geschwindigkeit = besser)
                    stability_bonus = max(0, 0.2 - abs(vx) - abs(vy))
                    
                    # Angle bonus (aufrecht = besser)
                    angle_bonus = max(0, 0.1 - abs(angle))
                    
                    # Successful landing (beide Beine am Boden)
                    if leg1 and leg2 and abs(vx) < 0.1 and abs(vy) < 0.1:
                        reward += 100.0  # Großer Landing-Bonus
                        print(f"SUCCESSFUL LANDING! Position: ({x:.3f}, {y:.3f}), Bonus: +100.0")
                    
                    reward += proximity_bonus + stability_bonus + angle_bonus

                return self.observation(obs), reward, terminated, truncated, info

        return DictObsWrapper(env)
    
    def make_env(self, env_config: dict, rank: int = 0, seed: int = 42):
        """Create a single environment with proper wrapping"""
        def _init():
            env = gym.make(env_config["env_id"], **env_config["env_kwargs"])
            env = self.create_dict_wrapper(env, env_config["wrapper"])
            env = Monitor(env)  # For episode statistics
            env.reset(seed=seed + rank)
            return env
        set_random_seed(seed)
        return _init
    
    def evaluate_random_policy(self, env_config: dict, n_episodes: int = 20) -> float:
        """Evaluate random policy baseline"""
        env_fn = self.make_env(env_config)
        env = DummyVecEnv([env_fn])
        
        episode_rewards = []
        
        # Robuste reset() Behandlung
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result
        
        for _ in range(n_episodes * 200):  # Max steps per episode
            actions = [env.action_space.sample()]
            
            # Robuste step() Behandlung
            step_result = env.step(actions)
            if len(step_result) == 5:
                obs, rewards, terminateds, truncateds, infos = step_result
                done = bool(terminateds[0] or truncateds[0])
            else:
                obs, rewards, dones, infos = step_result
                done = bool(dones[0])
            
            if done:
                if 'episode' in infos[0]:
                    episode_rewards.append(infos[0]['episode']['r'])
                if len(episode_rewards) >= n_episodes:
                    break
        
        env.close()
        return np.mean(episode_rewards) if episode_rewards else env_config["expected_random_reward"]
    
    def train_and_evaluate(self, env_name: str, env_config: dict, use_baseline: bool = False) -> Dict:
        """Train agent (either baseline or LD) on environment and evaluate performance"""

        variant = "Standard PPO" if use_baseline else "PPO+LSTM+λD"
        print(f"\n Testing {env_name} ({variant})")
        print(f" {env_config['description']}")
        print("=" * 60)
        
        # Create vectorized environment
        n_envs = 4
        env_fns = [self.make_env(env_config, i) for i in range(n_envs)]
        vec_env = DummyVecEnv(env_fns)
        # store vecnormalize reference for consistent evaluation
        self.vec_normalize = None
        # Für CartPole: Reward-Normalisierung und Monitoring aktivieren
        if env_name == "CartPole-v1":
            from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
            # sammle Episodendaten
            vec_env = VecMonitor(vec_env)
            # skaliere Rewards und Beobachtungen
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_reward=10.0)
            self.vec_normalize = vec_env  # Speichere für spätere Nutzung

        # Baseline evaluation
        print(" Evaluating random policy...")
        random_performance = self.evaluate_random_policy(env_config)
        print(f"Random policy average reward: {random_performance:.2f}")
        

        if use_baseline:
            # Standard PPO ohne LSTM, ohne λ-Discrepancy
            from stable_baselines3 import PPO
            
            # Vereinfachte Configs für Standard PPO
            if env_name == "FrozenLake-v1":
                config = {
                    "n_steps": 128, "batch_size": 64, "n_epochs": 4,
                    "gamma": 0.99, "ent_coef": 0.1, "vf_coef": 0.5,
                    "learning_rate": 3e-4
                }
            elif env_name == "CartPole-v1":
                config = {
                    "n_steps": 512, "batch_size": 256, "n_epochs": 4,
                    "gamma": 0.99, "ent_coef": 0.015, "vf_coef": 0.5,
                    "learning_rate": 3e-4
                }
            elif env_name == "LunarLander-v3":
                config = {
                    "n_steps": 2048, "batch_size": 512, "n_epochs": 4,
                    "gamma": 0.99, "ent_coef": 0.1, "vf_coef": 0.5,
                    "learning_rate": 3e-4, "clip_range": 0.2, "max_grad_norm": 0.5
                }
            elif env_name == "MountainCar-v0":
                config = {
                    "n_steps": 2048, "batch_size": 256, "n_epochs": 3,
                    "gamma": 0.99, "ent_coef": 0.3, "vf_coef": 0.25,
                    "learning_rate": 1e-3, "clip_range": 0.2, "max_grad_norm": 0.5
                }
            
            # Standard PPO Model
            model = PPO(
                policy="MultiInputPolicy",
                env=vec_env,
                verbose=1,
                tensorboard_log=str(self.session_root / "tensorboard"),
                **config
            )
            
        else:

            # Model configuration based on environment complexity
            if env_name == "FrozenLake-v1":
                config = {
                    "n_steps": 128, "batch_size": 64, "n_epochs": 4,
                    "gamma": 0.99, "ent_coef": 0.1, "vf_coef": 0.5, "ld_coef": 0.1,
                    "learning_rate": 3e-4
                }
            elif env_name == "CartPole-v1":
                config = {
                    "n_steps": 512, "batch_size": 256, "n_epochs": 4,
                    "gamma": 0.99, "ent_coef": 0.015, "vf_coef": 0.5, "ld_coef": 0.05,
                    "learning_rate": 3e-4
                }
            elif env_name == "LunarLander-v3":
                config = {
                    "n_steps": 2048,
                    "batch_size": 512, 
                    "n_epochs": 4,
                    "gamma": 0.99,
                    "ent_coef": 0.1,        
                    "vf_coef": 0.5,       
                    "ld_coef": 0.005,        
                    "learning_rate": 3e-4,
                    "clip_range": 0.2,     
                    "max_grad_norm": 0.5   
                }
            elif env_name == "MountainCar-v0":
                config = {
                    "n_steps": 2048,    
                    "batch_size": 256,   
                    "n_epochs": 3,          
                    "gamma": 0.99,       
                    "ent_coef": 0.3,       
                    "vf_coef": 0.25,    
                    "ld_coef": 0.01,    
                    "learning_rate": 1e-3,  
                    "clip_range": 0.2,      
                    "max_grad_norm": 0.5    
                }
            else: 
                config = {
                    "n_steps": 2048, "batch_size": 512, "n_epochs": 4,
                    "gamma": 0.999, "ent_coef": 0.1, "vf_coef": 0.25, "ld_coef": 0.01,
                    "learning_rate": 3e-4, "clip_range": 0.2, "max_grad_norm": 0.5
                }
            
            # Initialize model
            model = RecurrentPPOLD(
                policy=MultiInputLstmPolicyLD,
                env=vec_env,
                verbose=1,
                tensorboard_log=str(self.session_root / "tensorboard"),
                **config
            )
        
        print(f" Training for {env_config['timesteps']} timesteps...")
        
        # Training
        model.learn(
            total_timesteps=env_config['timesteps'],
            progress_bar=False,
            tb_log_name=f"{env_name}_{self.timestamp}"
        )
        
        # Save model
        model_path = self.session_root / "models" / f"{env_name}_model"
        model.save(model_path)
        
        # Evaluation
        print(" Evaluating trained policy...")
        if env_name == "CartPole-v1":
            eval_env = DummyVecEnv([self.make_env(env_config)])
        else:
            eval_env = DummyVecEnv([self.make_env(env_config)])
    
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=20, deterministic=True
        )
        
        # Detailed episode analysis
        episode_rewards, episode_lengths, actual_successes = self.detailed_evaluation(model, env_config)
        
        vec_env.close()
        eval_env.close()
        
        # Calculate improvement and success metrics
        improvement = mean_reward - random_performance
        if env_config.get('env_id') == 'MountainCar-v0':
            actual_success_rate = actual_successes / len(episode_rewards)
            reward_success_rate = np.mean(np.array(episode_rewards) >= env_config['success_threshold'])
            success_rate = actual_success_rate  # Use actual success for MountainCar
        else:
            # For other environments, both rates are the same
            success_rate = np.mean(np.array(episode_rewards) >= env_config['success_threshold'])
            actual_success_rate = success_rate  # Same as reward-based
            reward_success_rate = success_rate  # Same as reward-based

        result = {
            "environment": env_name,
            "description": env_config['description'],
            "random_performance": random_performance,
            "trained_performance": mean_reward,
            "std_reward": std_reward,
            "improvement": improvement,
            "success_threshold": env_config['success_threshold'],
            "success_rate": success_rate,
            "actual_success_rate": actual_success_rate,
            "reward_success_rate": reward_success_rate,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "passed": mean_reward >= env_config['success_threshold']
        }
        
        self.results[env_name] = result
        return result
    
    def detailed_evaluation(
        self,
        model,
        env_config: dict,
        n_episodes: int = 20
    ) -> Tuple[List[float], List[int]]:
        """
        Detaillierte Bewertung mit Action-Debugging
        """
        import torch
        
        # Debug Information
        print(f"DEBUG: Environment = {env_config['env_id']}")
        print(f"DEBUG: VecNormalize = {hasattr(self, 'vec_normalize') and self.vec_normalize is not None}")
        
        # WICHTIG: Model in Evaluation Mode
        model.policy.set_training_mode(False)

        env = DummyVecEnv([self.make_env(env_config)])

        episode_rewards: List[float] = []
        episode_lengths: List[int] = []
        
        # DEBUG: Action tracking
        action_counts = {}
        actual_successes = 0

        for episode in range(n_episodes):
            # LSTM State Reset für jede Episode
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)
            
            # Debug: Erste 3 Episoden detailliert loggen
            debug_mode = episode < 3
            if debug_mode:
                print(f"Episode {episode}: LSTM reset, episode_starts = {episode_starts}")
            
            # Robuste Behandlung für verschiedene Gymnasium Versionen
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, _ = reset_result
            else:
                obs = reset_result
                
            if debug_mode:
                print(f"  Initial obs: {obs}")
                
            total_reward = 0.0
            length = 0
            episode_actions = []  # DEBUG: Track actions per episode

            # Maximal 1000 Schritte pro Episode
            for step in range(1000):
                # Debug: Erste paar Steps loggen
                if debug_mode and step < 5:
                    print(f"  Step {step}: obs={obs}, episode_starts={episode_starts}")
                
                # Verwende LSTM states korrekt
                action, lstm_states = model.predict(
                    obs, 
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True
                )
                
                # DEBUG: Action Probabilities (nur für erste Episode)
                if debug_mode and step < 5 and episode == 0:
                    try:
                        with torch.no_grad():
                            # Get action distribution
                            obs_tensor = model.policy.obs_to_tensor(obs)[0]
                            features = model.policy.extract_features(obs_tensor)
                            latent_pi = model.policy.mlp_extractor.forward_actor(features)
                            action_dist = model.policy.action_dist.proba_distribution(latent_pi)
                            probs = action_dist.distribution.probs
                            entropy = action_dist.entropy()
                            
                            print(f"  Step {step}: Action probs = {probs.cpu().numpy()}")
                            print(f"  Step {step}: Entropy = {entropy.cpu().numpy()}")
                    except Exception as e:
                        print(f"  Step {step}: Could not get action probs: {e}")
                
                # DEBUG: Track actions
                action_int = int(action[0])
                episode_actions.append(action_int)
                action_counts[action_int] = action_counts.get(action_int, 0) + 1
                
                if debug_mode and step < 5:
                    print(f"  Step {step}: action={action}, lstm_states shape={lstm_states[0].shape if lstm_states else 'None'}")
                
                episode_starts = np.zeros((1,), dtype=bool)
                
                # Robuste Behandlung für step() Rückgabewerte
                step_result = env.step(action)
                if len(step_result) == 5:
                    obs, rewards, terminateds, truncateds, infos = step_result
                    done = bool(terminateds[0] or truncateds[0])
                else:
                    obs, rewards, dones, infos = step_result
                    done = bool(dones[0])
                
                r = float(rewards[0])
                total_reward += r
                length += 1

                if done and env_config.get('env_id') == 'MountainCar-v0':
                    # Success detection via reward (episodes with +500 bonus)
                    if total_reward > 300:  
                        actual_successes += 1

                elif done and env_config.get('env_id') == 'LunarLander-v3':
                    # Success detection for LunarLander (successful landing)
                    if total_reward > 200:
                        actual_successes += 1

                if debug_mode and step < 5:
                    print(f"  Step {step}: reward={r}, done={done}, total_reward={total_reward}")

                if done:
                    if debug_mode:
                        print(f"Episode {episode} finished: total_reward={total_reward}, length={length}")
                        print(f"  Actions this episode: {episode_actions[:10]}...")  # First 10 actions
                    break
                

            episode_rewards.append(total_reward)
            episode_lengths.append(length)

        # DEBUG: Final results
        print(f"DEBUG: Final episode_rewards = {episode_rewards}")
        print(f"DEBUG: Mean reward = {np.mean(episode_rewards):.3f}")
        print(f"DEBUG: Action distribution = {action_counts}")
        
        # DEBUG: Action diversity check
        total_actions = sum(action_counts.values())
        if total_actions > 0:
            action_diversity = len(action_counts) / max(1, len(action_counts))
            most_common_action = max(action_counts, key=action_counts.get)
            most_common_pct = action_counts[most_common_action] / total_actions
            print(f"DEBUG: Most common action: {most_common_action} ({most_common_pct:.1%})")
            if most_common_pct > 0.9:
                print(f"DEBUG: WARNING - Policy is very deterministic!")
        
        env.close()
        return episode_rewards, episode_lengths, actual_successes
    
    def run_comprehensive_check(self, environments: Optional[List[str]] = None):
        """Run the complete sanity check suite"""
        if environments is None:
            environments = list(self.test_configs.keys())
        
        print(" Starting Advanced Lambda-Discrepancy PPO Sanity Check")
        print(f" Results will be saved to: {self.session_root}")
        print("=" * 70)
        
        passed_tests = 0
        total_tests = len(environments)
        
        for env_name in environments:
            if env_name not in self.test_configs:
                print(f" Unknown environment: {env_name}")
                continue
                
            try:
                result = self.train_and_evaluate(env_name, self.test_configs[env_name])
                
                # Print results
                print(f"\n Results for {env_name}:")
                print(f"  Random baseline: {result['random_performance']:.2f}")
                print(f"  Trained performance: {result['trained_performance']:.2f} ± {result['std_reward']:.2f}")
                print(f"  Improvement: {result['improvement']:.2f}")
                if self.test_configs[env_name].get('env_id') == 'MountainCar-v0':
                    print(f"  Actual success rate: {result['actual_success_rate']:.1%}")
                    print(f"  Reward success rate: {result['reward_success_rate']:.1%}")
                else:
                    print(f"  Success rate: {result['success_rate']:.1%}")
                print(f"  Status: {' PASSED' if result['passed'] else ' FAILED'}")
                
                if result['passed']:
                    passed_tests += 1
                    
            except Exception as e:
                print(f" Failed to test {env_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Generate final report
        self.generate_report(passed_tests, total_tests)
        self.create_plots()
        
        return self.results
    
    def run_comparison_check(self, environments: Optional[List[str]] = None):
        """Run comparison between Standard PPO and PPO+LSTM+λD"""
        if environments is None:
            environments = list(self.test_configs.keys())
        
        print(" Starting PPO vs PPO+LSTM+λD Comparison")
        print(f" Results will be saved to: {self.session_root}")
        print("=" * 70)
        
        comparison_results = {}
        
        for env_name in environments:
            if env_name not in self.test_configs:
                print(f" Unknown environment: {env_name}")
                continue
            
            try:
                print(f"\n{'='*60}")
                print(f" TESTING ENVIRONMENT: {env_name}")
                print(f"{'='*60}")
                
                # Test Standard PPO
                print("\n [1/2] Testing Standard PPO...")
                baseline_result = self.train_and_evaluate(
                    env_name, self.test_configs[env_name], use_baseline=True
                )
                
                # Test PPO+LSTM+λD
                print("\n [2/2] Testing PPO+LSTM+λD...")
                advanced_result = self.train_and_evaluate(
                    env_name, self.test_configs[env_name], use_baseline=False
                )
                
                comparison_results[env_name] = {
                    'baseline': baseline_result,
                    'advanced': advanced_result
                }
                
                # Print comparison
                print(f"\n🎯 COMPARISON RESULTS for {env_name}:")
                print(f"  Standard PPO:     {baseline_result['trained_performance']:.2f}")
                print(f"  PPO+LSTM+λD:      {advanced_result['trained_performance']:.2f}")
                print(f"  Improvement:      {advanced_result['trained_performance'] - baseline_result['trained_performance']:.2f}")
                print(f"  Baseline passed:  {baseline_result['passed']}")
                print(f"  Advanced passed:  {advanced_result['passed']}")
                
            except Exception as e:
                print(f" Failed to compare {env_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Generate comparison report and plots
        self.results = comparison_results
        self.generate_comparison_report()
        self.create_comparison_plots()
        
        return comparison_results

    def generate_comparison_report(self):
        """Generate comparison report"""
        report_path = self.session_root / "comparison_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("PPO vs PPO+LSTM+λD Comparison Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {self.timestamp}\n\n")
            
            for env_name, results in self.results.items():
                baseline = results['baseline']
                advanced = results['advanced']
                
                f.write(f"\n{env_name}:\n")
                f.write(f"  Description: {baseline['description']}\n")
                f.write(f"  Random baseline: {baseline['random_performance']:.2f}\n")
                f.write(f"  Standard PPO: {baseline['trained_performance']:.2f} ± {baseline['std_reward']:.2f}\n")
                f.write(f"  PPO+LSTM+λD: {advanced['trained_performance']:.2f} ± {advanced['std_reward']:.2f}\n")
                f.write(f"  Improvement: {advanced['trained_performance'] - baseline['trained_performance']:.2f}\n")
                f.write(f"  Baseline passed: {baseline['passed']}\n")
                f.write(f"  Advanced passed: {advanced['passed']}\n")
                f.write(f"  Success improvement: {advanced['success_rate'] - baseline['success_rate']:.1%}\n")
        
        print(f"\n Comparison report saved to: {report_path}")

    def generate_report(self, passed_tests: int, total_tests: int):
        """Generate comprehensive text report"""
        report_path = self.session_root / "sanity_check_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Lambda-Discrepancy PPO Advanced Sanity Check Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Tests passed: {passed_tests}/{total_tests}\n")
            f.write(f"Overall success rate: {passed_tests/total_tests:.1%}\n\n")
            
            for env_name, result in self.results.items():
                f.write(f"\n{env_name}:\n")
                f.write(f"  Description: {result['description']}\n")
                f.write(f"  Random baseline: {result['random_performance']:.2f}\n")
                f.write(f"  Trained performance: {result['trained_performance']:.2f} ± {result['std_reward']:.2f}\n")
                f.write(f"  Improvement: {result['improvement']:.2f}\n")
                f.write(f"  Success threshold: {result['success_threshold']:.2f}\n")
                f.write(f"  Success rate: {result['success_rate']:.1%}\n")
                f.write(f"  Status: {'PASSED' if result['passed'] else 'FAILED'}\n")
                f.write(f"  Avg episode length: {np.mean(result['episode_lengths']):.1f}\n")
        
        print(f"\n Detailed report saved to: {report_path}")
    
    def create_paper_style_plots(self):
        """Create publication-ready plots like in the lambda discrepancy paper"""
        if not self.results:
            return
            
        # Create separate plot for each environment
        for env_name, result in self.results.items():
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Extract learning curve data from TensorBoard logs
            tb_log_dir = self.session_root / "tensorboard" / f"{env_name}_{self.timestamp}_1"
            
            if tb_log_dir.exists():
                # Read TensorBoard data
                rewards, timesteps = self.extract_tensorboard_data(tb_log_dir)
                
                # Plot learning curve
                ax.plot(timesteps, rewards, 
                    label=f'PPO+LSTM+λD (ld_coef={self.get_ld_coef(env_name)})',
                    linewidth=2, alpha=0.8)
                
                # Add smoothed version
                if len(rewards) > 10:
                    from scipy.ndimage import uniform_filter1d
                    smoothed = uniform_filter1d(rewards, size=min(50, len(rewards)//10))
                    ax.plot(timesteps, smoothed, 
                        label=f'PPO+LSTM+λD (smoothed)',
                        linewidth=3, alpha=0.9)
            
            # Add baseline comparison
            random_perf = result['random_performance']
            ax.axhline(y=random_perf, color='red', linestyle='--', 
                    label=f'Random Policy ({random_perf:.1f})', alpha=0.7)
            
            # Add success threshold
            threshold = result['success_threshold']
            ax.axhline(y=threshold, color='green', linestyle='--', 
                    label=f'Success Threshold ({threshold:.1f})', alpha=0.7)
            
            # Formatting like in the paper
            ax.set_xlabel('Timesteps', fontsize=12)
            ax.set_ylabel('Episode Reward', fontsize=12)
            ax.set_title(f'{env_name} - Learning Curve', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Scientific notation for large timesteps
            ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
            
            plt.tight_layout()
            
            # Save individual plot
            plot_path = self.session_root / "plots" / f"{env_name}_learning_curve.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f" Learning curve saved: {plot_path}")

    def extract_tensorboard_data(self, tb_log_dir):
        """Extract episode rewards and timesteps from TensorBoard logs"""
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            
            # Initialize event accumulator
            ea = EventAccumulator(str(tb_log_dir))
            ea.Reload()
            
            # Get episode reward data
            if 'rollout/ep_rew_mean' in ea.Tags()['scalars']:
                reward_events = ea.Scalars('rollout/ep_rew_mean')
                timesteps = [event.step for event in reward_events]
                rewards = [event.value for event in reward_events]
                return rewards, timesteps
            else:
                print(f" Warning: No episode reward data found in {tb_log_dir}")
                return [], []
                
        except ImportError:
            print(" Warning: TensorBoard not available for data extraction")
            return [], []
        except Exception as e:
            print(f" Warning: Could not extract TensorBoard data: {e}")
            return [], []

    def get_ld_coef(self, env_name):
        """Get lambda discrepancy coefficient for environment"""
        configs = {
            "FrozenLake-v1": 0.1,
            "CartPole-v1": 0.05,
            "LunarLander-v3": 0.005,
            "MountainCar-v0": 0.01
        }
        return configs.get(env_name, 0.01)

    def create_plots(self):
        """Create both overview and paper-style plots"""
        if not self.results:
            return
        
        # Create paper-style individual plots
        self.create_paper_style_plots()
        
        # Keep existing overview plots
        self.create_overview_plots()

    def create_overview_plots(self):
        """Create visualization plots, now including a learning curve Reward vs. Timesteps."""
        if not self.results:
            return

        envs = list(self.results.keys())
        n_envs = len(envs)

        # 3×2 Subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

        # 1) Performance comparison
        random_perfs  = [self.results[env]['random_performance']  for env in envs]
        trained_perfs = [self.results[env]['trained_performance'] for env in envs]
        x = np.arange(n_envs)
        width = 0.35

        ax1.bar(x - width/2, random_perfs,  width, label='Random',    alpha=0.7)
        ax1.bar(x + width/2, trained_perfs, width, label='Trained',   alpha=0.7)
        ax1.set_xticks(x)
        ax1.set_xticklabels([env.replace('-v', '\nv') for env in envs], rotation=45)
        ax1.set_ylabel('Avg Reward')
        ax1.set_title('Policy Performance Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2) Success rates
        success_rates = [self.results[env]['success_rate'] for env in envs]
        colors = ['green' if r >= 0.7 else 'orange' if r >= 0.5 else 'red'
                for r in success_rates]
        ax2.bar(envs, success_rates, color=colors, alpha=0.7)
        ax2.set_xticklabels([env.replace('-v', '\nv') for env in envs], rotation=45)
        ax2.set_ylabel('Success Rate')
        ax2.set_title('Success Rates by Environment')
        ax2.axhline(0.7, color='green',  linestyle='--', alpha=0.5, label='Good (70%)')
        ax2.axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='OK (50%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3) Episode length distribution (erstes Env)
        lengths = self.results[envs[0]]['episode_lengths']
        ax3.hist(lengths, bins=20, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Episode Length')
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'Episode Length Distribution ({envs[0]})')
        ax3.grid(True, alpha=0.3)

        # 4) Reward distribution (erstes Env)
        rewards = self.results[envs[0]]['episode_rewards']
        thr     = self.results[envs[0]]['success_threshold']
        ax4.hist(rewards, bins=20, alpha=0.7, edgecolor='black')
        ax4.axvline(thr, color='red', linestyle='--', label='Threshold')
        ax4.set_xlabel('Episode Reward')
        ax4.set_ylabel('Frequency')
        ax4.set_title(f'Reward Distribution ({envs[0]})')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5) Learning Curve: Reward vs. cumulative Timesteps
        cum_steps   = np.cumsum(lengths)
        ax5.plot(cum_steps, rewards, marker='o', alpha=0.7)
        ax5.set_xlabel('Timesteps')
        ax5.set_ylabel('Episode Reward')
        ax5.set_title(f'Learning Curve ({envs[0]})')
        ax5.grid(True, alpha=0.3)

        # 6) Leerer Platz (kann für späteren Use-Case entfallen)
        ax6.axis('off')

        plt.tight_layout()
        plot_path = self.session_root / "plots" / "sanity_check_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        print(f" Plots saved to: {plot_path}")

    def create_comparison_plots(self):
        """Create comparison plots between Standard PPO and PPO+LSTM+λD"""
        if not self.results:
            return
        
        # Create separate comparison plot for each environment
        for env_name, results in self.results.items():
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Extract TensorBoard data for both variants
            baseline_rewards, baseline_timesteps = self.extract_tensorboard_data_for_variant(env_name, 'baseline')
            advanced_rewards, advanced_timesteps = self.extract_tensorboard_data_for_variant(env_name, 'advanced')
            
            # Plot both learning curves
            if baseline_rewards and len(baseline_rewards) > 0:
                ax.plot(baseline_timesteps, baseline_rewards, 
                    label='Standard PPO', color='red', linewidth=2, alpha=0.8)
                
                # Add smoothed version for baseline
                if len(baseline_rewards) > 10:
                    from scipy.ndimage import uniform_filter1d
                    smoothed_baseline = uniform_filter1d(baseline_rewards, size=min(50, len(baseline_rewards)//10))
                    ax.plot(baseline_timesteps, smoothed_baseline, 
                        label='Standard PPO (smoothed)', color='darkred', linewidth=3, alpha=0.9)
            
            if advanced_rewards and len(advanced_rewards) > 0:
                ax.plot(advanced_timesteps, advanced_rewards, 
                    label='PPO+LSTM+λD', color='blue', linewidth=2, alpha=0.8)
                
                # Add smoothed version for advanced
                if len(advanced_rewards) > 10:
                    from scipy.ndimage import uniform_filter1d
                    smoothed_advanced = uniform_filter1d(advanced_rewards, size=min(50, len(advanced_rewards)//10))
                    ax.plot(advanced_timesteps, smoothed_advanced, 
                        label='PPO+LSTM+λD (smoothed)', color='darkblue', linewidth=3, alpha=0.9)
            
            # Add baseline comparison
            random_perf = results['baseline']['random_performance']
            ax.axhline(y=random_perf, color='gray', linestyle='--', 
                    label=f'Random Policy ({random_perf:.1f})', alpha=0.7)
            
            # Add success threshold
            threshold = results['baseline']['success_threshold']
            ax.axhline(y=threshold, color='green', linestyle='--', 
                    label=f'Success Threshold ({threshold:.1f})', alpha=0.7)
            
            # Add final performance annotations
            baseline_final = results['baseline']['trained_performance']
            advanced_final = results['advanced']['trained_performance']
            improvement = advanced_final - baseline_final
            
            ax.text(0.02, 0.98, f'Final Performance:\nStandard PPO: {baseline_final:.1f}\nPPO+LSTM+λD: {advanced_final:.1f}\nImprovement: {improvement:.1f}',
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Formatting
            ax.set_xlabel('Timesteps', fontsize=12)
            ax.set_ylabel('Episode Reward', fontsize=12)
            ax.set_title(f'{env_name} - Learning Curve Comparison', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10, loc='lower right')
            ax.grid(True, alpha=0.3)
            ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
            
            plt.tight_layout()
            
            # Save comparison plot
            plot_path = self.session_root / "plots" / f"{env_name}_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f" Comparison plot saved: {plot_path}")

    def extract_tensorboard_data_for_variant(self, env_name, variant):
        """Extract TensorBoard data for specific variant (baseline or advanced)"""
        if variant == 'baseline':
            # For baseline runs, the tensorboard log has a different naming
            tb_log_dir = self.session_root / "tensorboard" / f"{env_name}_{self.timestamp}_1"
        else:
            # For advanced runs
            tb_log_dir = self.session_root / "tensorboard" / f"{env_name}_{self.timestamp}_2"
        
        return self.extract_tensorboard_data(tb_log_dir)

def debug_single_env(env_name: str):
    """Debug a single environment - perfect for debugging"""
    checker = AdvancedSanityCheck()
    
    if env_name not in checker.test_configs:
        print(f"Unknown environment: {env_name}")
        return
    
    # DEBUG: Zeige geladene Config
    config = checker.test_configs[env_name]
    print(f" DEBUG: Loaded timesteps = {config['timesteps']}")
    print(f" DEBUG: Full config = {config}")
    
    print(f" DEBUG MODE: Testing {env_name} only")
    print("=" * 50)
    
    # DEBUG: Action Space Test
    print(" DEBUG: Testing Action Space...")
    env = gym.make(config["env_id"], **config["env_kwargs"])
    print(f"  Action space: {env.action_space}")
    print(f"  Action space type: {type(env.action_space)}")
    if hasattr(env.action_space, 'n'):
        print(f"  Action space.n: {env.action_space.n}")
        
        # Test random actions
        print("  Random actions sample:")
        for i in range(5):
            action = env.action_space.sample()
            print(f"    Random action {i}: {action}")
    env.close()
    
    try:
        result = checker.train_and_evaluate(env_name, checker.test_configs[env_name])
        
        print(f"\n DEBUG RESULTS for {env_name}:")
        print(f"  Random baseline: {result['random_performance']:.2f}")
        print(f"  Trained performance: {result['trained_performance']:.2f} ± {result['std_reward']:.2f}")
        print(f"  Improvement: {result['improvement']:.2f}")
        print(f"  Success rate: {result['success_rate']:.1%}")
        print(f"  Status: {' PASSED' if result['passed'] else ' FAILED'}")
        
        return result
        
    except Exception as e:
        print(f" DEBUG FAILED for {env_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main entry point for advanced sanity check"""
    import sys

    # Comparison mode
    if len(sys.argv) > 1 and sys.argv[1] == '--compare':
        if len(sys.argv) > 2:
            env_list = sys.argv[2].split(',')
            print(f" Running comparison for: {env_list}")
        else:
            env_list = None
            print(" Running comparison for all environments")
        
        checker = AdvancedSanityCheck()
        results = checker.run_comparison_check(env_list)
        
        # Summary
        print("\n" + "=" * 70)
        print(" PPO vs PPO+LSTM+λD COMPARISON COMPLETE")
        print("=" * 70)
        
        for env_name, result in results.items():
            baseline_perf = result['baseline']['trained_performance']
            advanced_perf = result['advanced']['trained_performance']
            improvement = advanced_perf - baseline_perf
            
            print(f" {env_name}:")
            print(f"   Standard PPO: {baseline_perf:.2f}")
            print(f"   PPO+LSTM+λD:  {advanced_perf:.2f}")
            print(f"   Improvement:  {improvement:.2f}")
            print(f"   Winner: {'PPO+LSTM+λD' if improvement > 0 else 'Standard PPO'}")
        
        print(f"\n Detailed results in: {checker.session_root}")
        return
    
    # Quick debug mode
    if len(sys.argv) > 1 and sys.argv[1] == '--debug':
        if len(sys.argv) > 2:
            debug_single_env(sys.argv[2])
        else:
            debug_single_env('FrozenLake-v1')  # Default to FrozenLake
        return
    
    if len(sys.argv) > 1 and sys.argv[1] == '--envs':
        if len(sys.argv) > 2:
            # Comma-separated environment list
            env_list = sys.argv[2].split(',')
            print(f" Running custom environments: {env_list}")
        else:
            env_list = None
        
        checker = AdvancedSanityCheck()
        results = checker.run_comprehensive_check(env_list)
        
        # Summary
        passed = sum(1 for r in results.values() if r['passed'])
        total = len(results)
        print(f"\n Custom run: {passed}/{total} tests passed")
        return
    

    checker = AdvancedSanityCheck()
    
    # You can specify which environments to test
    # environments = ["FrozenLake-v1", "CartPole-v1"]  # Subset for quick testing
    environments = None  # All environments
    
    results = checker.run_comprehensive_check(environments)
    
    # Final summary
    passed = sum(1 for r in results.values() if r['passed'])
    total = len(results)
    
    print("\n" + "=" * 70)
    print(" ADVANCED SANITY CHECK COMPLETE")
    print("=" * 70)
    print(f" Overall Results: {passed}/{total} tests passed ({passed/total:.1%})")
    
    if passed == total:
        print(" All tests passed! The Lambda-Discrepancy PPO implementation looks solid.")
    elif passed >= total * 0.75:
        print(" Most tests passed. Implementation is likely working correctly.")
    elif passed >= total * 0.5:
        print("  Some tests failed. Check the failed environments for issues.")
    else:
        print(" Multiple test failures. Implementation may need debugging.")
    
    print(f" Detailed results in: {checker.session_root}")
    print(f" TensorBoard: tensorboard --logdir {checker.session_root}/tensorboard")


if __name__ == "__main__":
    main()