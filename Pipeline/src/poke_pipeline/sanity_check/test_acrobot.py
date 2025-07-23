#!/usr/bin/env python3
"""
Simple test script for Acrobot implementations.
Tests both standard Acrobot-v1 and Acrobot-v1 with partial observability.
"""

import sys
from pathlib import Path
import numpy as np

# Add the src directory to Python path for imports
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from core.environment import EnvironmentFactory


def test_acrobot_environments():
    """Test both Acrobot environment configurations."""
    
    print("🤖 Testing Acrobot Environment Implementations")
    print("=" * 50)
    
    factory = EnvironmentFactory()
    
    # Test 1: Standard Acrobot-v1
    print("\n1️⃣ Testing Acrobot-v1 (Standard)")
    print("-" * 30)
    
    try:
        env = factory.make_env('Acrobot-v1', seed=42)
        print(f"✅ Environment created successfully")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")
        
        # Test reset
        obs, info = env.reset()
        print(f"   Reset successful - obs shape: {obs['state'].shape}")
        print(f"   Initial observation: {obs['state'][:4]}... (first 4 values)")
        
        # Test a few random steps
        total_reward = 0
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"   Step {step+1}: action={action}, reward={reward:.3f}, done={terminated or truncated}")
            
        print(f"   Total reward after 5 steps: {total_reward:.3f}")
        env.close()
        
    except Exception as e:
        print(f"❌ Error testing Acrobot-v1: {e}")
        return False
    
    # Test 2: Acrobot-v1 with Partial Observability
    print("\n2️⃣ Testing Acrobot-v1-Partial (Partial Observability)")
    print("-" * 45)
    
    try:
        env = factory.make_env('Acrobot-v1-Partial', seed=42)
        print(f"✅ Environment created successfully")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")
        
        # Test reset
        obs, info = env.reset()
        print(f"   Reset successful - obs shape: {obs['state'].shape}")
        print(f"   Partial observation: {obs['state']} (should be 4D instead of 6D)")
        
        # Verify observation dimension reduction
        expected_shape = (4,)  # Should be 4D after hiding velocity components
        if obs['state'].shape == expected_shape:
            print(f"   ✅ Correct observation reduction: {obs['state'].shape}")
        else:
            print(f"   ❌ Unexpected observation shape: {obs['state'].shape}, expected: {expected_shape}")
        
        # Test a few random steps
        total_reward = 0
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"   Step {step+1}: action={action}, reward={reward:.3f}, done={terminated or truncated}")
            
        print(f"   Total reward after 5 steps: {total_reward:.3f}")
        env.close()
        
    except Exception as e:
        print(f"❌ Error testing Acrobot-v1-Partial: {e}")
        return False
    
    # Test 3: Compare observation spaces
    print("\n3️⃣ Comparing Observation Spaces")
    print("-" * 30)
    
    try:
        env_standard = factory.make_env('Acrobot-v1', seed=42)
        env_partial = factory.make_env('Acrobot-v1-Partial', seed=42)
        
        obs_std, _ = env_standard.reset()
        obs_part, _ = env_partial.reset()
        
        print(f"   Standard Acrobot obs shape: {obs_std['state'].shape}")
        print(f"   Partial Acrobot obs shape: {obs_part['state'].shape}")
        print(f"   Difference: {obs_std['state'].shape[0] - obs_part['state'].shape[0]} dimensions hidden")
        
        # Show what components are visible/hidden
        if obs_std['state'].shape[0] == 6 and obs_part['state'].shape[0] == 4:
            print(f"   ✅ Correct: 2 velocity components (θ̇₁, θ̇₂) are hidden")
            print(f"   Standard: [cos(θ₁), sin(θ₁), cos(θ₂), sin(θ₂), θ̇₁, θ̇₂]")
            print(f"   Partial:  [cos(θ₁), sin(θ₁), cos(θ₂), sin(θ₂)]")
        else:
            print(f"   ❌ Unexpected observation dimensions")
        
        env_standard.close()
        env_partial.close()
        
    except Exception as e:
        print(f"❌ Error comparing environments: {e}")
        return False
    
    # Test 4: Available environments
    print("\n4️⃣ Available Environments")
    print("-" * 25)
    
    available_envs = factory.get_available_environments()
    acrobot_envs = [env for env in available_envs if 'Acrobot' in env]
    
    print(f"   Total environments available: {len(available_envs)}")
    print(f"   Acrobot environments: {acrobot_envs}")
    
    expected_acrobot_envs = ['Acrobot-v1', 'Acrobot-v1-Partial']
    for env_name in expected_acrobot_envs:
        if env_name in available_envs:
            print(f"   ✅ {env_name} is available")
        else:
            print(f"   ❌ {env_name} is missing")
    
    print("\n🎉 Acrobot environment tests completed!")
    return True


def test_random_policy_performance():
    """Test random policy performance on both environments."""
    
    print("\n📊 Random Policy Performance Test")
    print("=" * 35)
    
    factory = EnvironmentFactory()
    
    for env_name in ['Acrobot-v1', 'Acrobot-v1-Partial']:
        print(f"\n🎲 Testing random policy on {env_name}")
        print("-" * 40)
        
        try:
            mean_reward, std_reward = factory.evaluate_random_policy(env_name, n_episodes=5, seed=42)
            print(f"   Mean reward: {mean_reward:.3f} ± {std_reward:.3f}")
            print(f"   Expected range: around -500 to -50 (Acrobot typically)")
            
            if -600 <= mean_reward <= 0:
                print(f"   ✅ Reward in expected range")
            else:
                print(f"   ⚠️  Reward outside typical range")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")


if __name__ == "__main__":
    print("🚀 Starting Acrobot Environment Tests")
    
    try:
        # Run basic environment tests
        if test_acrobot_environments():
            # Run performance tests
            test_random_policy_performance()
        else:
            print("❌ Basic tests failed, skipping performance tests")
            
    except Exception as e:
        print(f"❌ Test script failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✨ Test script completed!")
