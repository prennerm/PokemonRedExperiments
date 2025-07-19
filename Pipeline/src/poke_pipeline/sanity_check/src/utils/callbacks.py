"""
Training callbacks for monitoring and progress tracking.

This module provides callbacks for enhanced statistics collection and 
progress visualization during agent training.
"""

import time
import numpy as np
from typing import Dict, Any, Optional
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger


class EnhancedStatsCallback(BaseCallback):
    """
    Enhanced statistics callback for detailed training monitoring.
    
    Collects additional metrics beyond the standard SB3 logging:
    - Episode success rates
    - Reward statistics
    - Training efficiency metrics
    """
    
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.start_time = None
        self.last_log_time = None
        
    def _on_training_start(self) -> None:
        """Called when training starts."""
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
        if self.verbose > 0:
            print("🚀 Training started - Enhanced statistics enabled")
    
    def _on_step(self) -> bool:
        """Called at each training step."""
        # Check if episode ended
        if self.locals.get('done', False):
            # Get episode info
            info = self.locals.get('info', {})
            if isinstance(info, list) and len(info) > 0:
                info = info[0]
            
            # Log episode reward if available
            if 'episode' in info:
                episode_reward = info['episode']['r']
                episode_length = info['episode']['l']
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.episode_count += 1
        
        # Log enhanced statistics periodically
        if self.num_timesteps % self.log_freq == 0:
            self._log_enhanced_stats()
        
        return True
    
    def _log_enhanced_stats(self) -> None:
        """Log enhanced statistics to TensorBoard."""
        if len(self.episode_rewards) == 0:
            return
        
        current_time = time.time()
        
        # Calculate statistics
        recent_rewards = self.episode_rewards[-100:]  # Last 100 episodes
        recent_lengths = self.episode_lengths[-100:]
        
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)
        mean_length = np.mean(recent_lengths)
        
        # Training efficiency
        time_elapsed = current_time - self.start_time
        steps_per_second = self.num_timesteps / time_elapsed if time_elapsed > 0 else 0
        
        # Log to TensorBoard
        self.logger.record("enhanced/mean_episode_reward", mean_reward)
        self.logger.record("enhanced/std_episode_reward", std_reward)
        self.logger.record("enhanced/mean_episode_length", mean_length)
        self.logger.record("enhanced/total_episodes", self.episode_count)
        self.logger.record("enhanced/steps_per_second", steps_per_second)
        
        # Success rate (environment-specific)
        if len(recent_rewards) > 0:
            # Generic success threshold - can be customized per environment
            success_threshold = self._get_success_threshold()
            if success_threshold is not None:
                success_rate = np.mean(np.array(recent_rewards) >= success_threshold)
                self.logger.record("enhanced/success_rate", success_rate)
        
        self.last_log_time = current_time
    
    def _get_success_threshold(self) -> Optional[float]:
        """Get success threshold based on environment."""
        # This could be made configurable or environment-specific
        # For now, return None to disable success rate logging
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current training statistics."""
        if len(self.episode_rewards) == 0:
            return {}
        
        return {
            'total_episodes': self.episode_count,
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'mean_length': np.mean(self.episode_lengths),
            'total_timesteps': self.num_timesteps,
            'training_time': time.time() - self.start_time if self.start_time else 0
        }


class ProgressCallback(BaseCallback):
    """
    Progress callback for console output during training.
    
    Provides regular progress updates with key metrics displayed
    in a user-friendly format.
    """
    
    def __init__(self, log_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.start_time = None
        self.last_mean_reward = None
        
    def _on_training_start(self) -> None:
        """Called when training starts."""
        self.start_time = time.time()
        
        if self.verbose > 0:
            print("🎯 Training Progress Monitor Started")
            print("-" * 50)
    
    def _on_step(self) -> bool:
        """Called at each training step."""
        if self.num_timesteps % self.log_freq == 0:
            self._print_progress()
        
        return True
    
    def _print_progress(self) -> None:
        """Print training progress to console."""
        if self.verbose == 0:
            return
        
        current_time = time.time()
        elapsed_time = current_time - self.start_time if self.start_time else 0
        
        # Get recent episode rewards if available
        recent_reward = "N/A"
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            recent_rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
            if recent_rewards:
                recent_reward = f"{np.mean(recent_rewards):.2f}"
        
        # Calculate steps per second
        steps_per_sec = self.num_timesteps / elapsed_time if elapsed_time > 0 else 0
        
        # Progress percentage (if total_timesteps is known)
        progress = "N/A"
        if hasattr(self.model, '_total_timesteps') and self.model._total_timesteps:
            progress_pct = (self.num_timesteps / self.model._total_timesteps) * 100
            progress = f"{progress_pct:.1f}%"
        
        # Format time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        print(f"📊 Step: {self.num_timesteps:,} | "
              f"Progress: {progress} | "
              f"Reward: {recent_reward} | "
              f"Time: {time_str} | "
              f"Speed: {steps_per_sec:.0f} steps/s")
    
    def _on_training_end(self) -> None:
        """Called when training ends."""
        if self.verbose > 0:
            total_time = time.time() - self.start_time if self.start_time else 0
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            
            print("-" * 50)
            print(f"✅ Training completed in {hours:02d}:{minutes:02d}:{seconds:02d}")
            print(f"📈 Total timesteps: {self.num_timesteps:,}")


class EpisodeStatsCallback(BaseCallback):
    """
    Simple callback to collect episode statistics.
    
    Lightweight alternative to EnhancedStatsCallback for basic monitoring.
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
    
    def _on_step(self) -> bool:
        """Called at each training step."""
        # Check if episode ended
        if self.locals.get('done', False):
            info = self.locals.get('info', {})
            if isinstance(info, list) and len(info) > 0:
                info = info[0]
            
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
                self.episode_count += 1
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get episode statistics."""
        if len(self.episode_rewards) == 0:
            return {'episode_count': 0}
        
        return {
            'episode_count': self.episode_count,
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'mean_length': np.mean(self.episode_lengths),
            'total_timesteps': self.num_timesteps
        }