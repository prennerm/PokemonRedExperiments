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
                "success_threshold": -120,      # Should solve in <120 steps
                "timesteps": 250000,
                "wrapper": "continuous_dict",
                "description": "Tests sparse rewards + exploration + momentum"
            },
            "LunarLander-v3": {
                "env_id": "LunarLander-v3",
                "env_kwargs": {},
                "expected_random_reward": -150, # Random policy crashes
                "success_threshold": 100,       # Successful landing
                "timesteps": 150000,
                "wrapper": "continuous_dict", 
                "description": "Tests complex dynamics + shaped rewards"
            }
        }
        
        self.results = {}
        self.setup_directories()
    
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
    
    def train_and_evaluate(self, env_name: str, env_config: dict) -> Dict:
        """Train agent on environment and evaluate performance"""
        print(f"\n Testing {env_name}")
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
        elif env_name == "MountainCar-v0":
            config = {
                "n_steps": 512, "batch_size": 256, "n_epochs": 8,
                "gamma": 0.99, "ent_coef": 0.1, "vf_coef": 0.5, "ld_coef": 0.1,  # 0.01 -> 0.1
                "learning_rate": 1e-4
            }
        else:  # Complex environments
            config = {
                "n_steps": 512, "batch_size": 256, "n_epochs": 8,
                "gamma": 0.99, "ent_coef": 0.01, "vf_coef": 0.5, "ld_coef": 0.1,
                "learning_rate": 1e-4
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
        if self.vec_normalize is not None:
            # Verwende das gleiche VecNormalize Setup
            eval_env_fns = [self.make_env(env_config)]
            eval_env = DummyVecEnv(eval_env_fns)
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, training=False)
            # Kopiere die Normalisierungs-Parameter vom Training
            eval_env.obs_rms = self.vec_normalize.obs_rms
            eval_env.ret_rms = self.vec_normalize.ret_rms
        else:
            eval_env = DummyVecEnv([self.make_env(env_config)])
    
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=20, deterministic=True
        )
        
        # Detailed episode analysis
        episode_rewards, episode_lengths = self.detailed_evaluation(model, env_config)
        
        vec_env.close()
        eval_env.close()
        
        # Calculate improvement and success metrics
        improvement = mean_reward - random_performance
        success_rate = np.mean(np.array(episode_rewards) >= env_config['success_threshold'])
        
        result = {
            "environment": env_name,
            "description": env_config['description'],
            "random_performance": random_performance,
            "trained_performance": mean_reward,
            "std_reward": std_reward,
            "improvement": improvement,
            "success_threshold": env_config['success_threshold'],
            "success_rate": success_rate,
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
        Detaillierte Bewertung: Man führt n_episodes Episoden durch,
        sammelt pro Episode die kumulierte Belohnung und Länge.
        """
        # WICHTIG: Model in Evaluation Mode
        model.policy.set_training_mode(False)
        
        # Environment Setup (gleiche Logik wie in train_and_evaluate)
        if hasattr(self, 'vec_normalize') and self.vec_normalize is not None:
            # CartPole: Verwende VecNormalize auch für Evaluation
            env = DummyVecEnv([self.make_env(env_config)])
            from stable_baselines3.common.vec_env import VecNormalize
            env = VecNormalize(env, norm_obs=True, norm_reward=True, training=False)
            # Kopiere Normalisierungsparameter vom Training
            env.obs_rms = self.vec_normalize.obs_rms
            env.ret_rms = self.vec_normalize.ret_rms
        else:
            # Andere Environments: Kein VecNormalize
            env = DummyVecEnv([self.make_env(env_config)])

        episode_rewards: List[float] = []
        episode_lengths: List[int] = []

        for _ in range(n_episodes):
            # LSTM State Reset für jede Episode
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)
            
            # Robuste Behandlung für verschiedene Gymnasium Versionen
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, _ = reset_result  # Neue Gymnasium Version
            else:
                obs = reset_result     # Alte Gym Version
                
            total_reward = 0.0
            length = 0

            # Maximal 1000 Schritte pro Episode
            for step in range(1000):
                # Verwende LSTM states korrekt
                action, lstm_states = model.predict(
                    obs, 
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True
                )
                episode_starts = np.zeros((1,), dtype=bool)  # Nur beim ersten Step True
                
                # Robuste Behandlung für step() Rückgabewerte
                step_result = env.step(action)
                if len(step_result) == 5:
                    # Neue Gymnasium Version: obs, rewards, terminateds, truncateds, infos
                    obs, rewards, terminateds, truncateds, infos = step_result
                    done = bool(terminateds[0] or truncateds[0])
                else:
                    # Alte Gym Version: obs, rewards, dones, infos
                    obs, rewards, dones, infos = step_result
                    done = bool(dones[0])
                
                r = float(rewards[0])
                total_reward += r
                length += 1

                if done:
                    break

            episode_rewards.append(total_reward)
            episode_lengths.append(length)

        env.close()
        return episode_rewards, episode_lengths
    
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
    
    def generate_report(self, passed_tests: int, total_tests: int):
        """Generate comprehensive text report"""
        report_path = self.session_root / "sanity_check_report.txt"
        
        with open(report_path, 'w') as f:
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
    

    def create_plots(self):
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


def main():
    """Main entry point for advanced sanity check"""
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