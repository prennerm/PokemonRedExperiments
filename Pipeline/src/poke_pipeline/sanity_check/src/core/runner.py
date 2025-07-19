"""
Main runner for multi-agent benchmarks.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np

from .environment import EnvironmentFactory
from .evaluator import PolicyEvaluator
from .reporter import ComparisonReporter, MarkdownReporter
from agents.base import BaseAgent
from agents import get_agent_class, list_available_agents, get_agent_info
from utils.tensorboard import extract_tensorboard_data
from utils.paths import create_session_directory


class SanityCheckRunner:
    """Main runner for multi-agent benchmarks."""
    
    def __init__(self, base_dir: Path = Path("results")):
        self.base_dir = Path(base_dir)
        
        # Create session directory with proper parameters
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_root = create_session_directory(None, self.timestamp)
        
        # Initialize components (no config_dir needed anymore)
        self.env_factory = EnvironmentFactory()
        self.evaluator = PolicyEvaluator(self.env_factory)
        self.reporter = ComparisonReporter(self.session_root / "reports")
        self.md_reporter = MarkdownReporter(self.session_root / "reports")
        
        # Get available agents from registry (no YAML needed)
        self.available_agents = list_available_agents()
        self.agent_info = get_agent_info()
        
        # Storage for results
        self.results = {}
        self.training_times = {}
        self.tensorboard_data = {}
        
        print(f"🎯 Session initialized: {self.session_root}")
        print(f"📦 Available agents: {', '.join(self.available_agents)}")
        print(f"🏗️ Available environments: {', '.join(self.env_factory.get_available_environments())}")
    
    def run_single(self, env_name: str, agent_name: str, seed: Optional[int] = None) -> Dict[str, Any]:
        """Run single agent on single environment."""
        print(f"\n🔄 Running {agent_name} on {env_name}")
        
        # Load agent
        agent = self._load_agent(agent_name)
        
        # Create environment
        env_config = self.env_factory.get_env_config(env_name)
        
        # Create vectorized environment for training
        vec_env = self.env_factory.make_vec_env(
            env_name, 
            n_envs=1, 
            seed=seed,
            monitor_dir=self.session_root / "logs" / f"{env_name}_{agent_name}"
        )
        
        # Train model
        print(f"  Training for {env_config['timesteps']} timesteps...")
        start_time = time.time()
        
        model = agent.create_model(vec_env, {}, 
                                  tensorboard_log=str(self.session_root / "tensorboard"))
        trained_model = agent.train(model, env_config['timesteps'])
        
        training_time = time.time() - start_time
        
        # Save model
        model_path = self.session_root / "models" / f"{env_name}_{agent_name}.zip"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        trained_model.save(str(model_path))
        
        # Evaluate
        print(f"  Evaluating...")
        results = self.evaluator.detailed_evaluation(
            trained_model, env_name, agent_name, n_episodes=20, seed=seed
        )
        
        # Add training time
        results['training_time'] = training_time
        
        # Extract TensorBoard data
        # Try different naming patterns for tensorboard logs
        tb_log_patterns = [
            f"{env_name}_{agent_name}_{self.timestamp}_1",
            f"{agent_name}_1", 
            "PPO_1",
            f"{env_name}_{agent_name}_1"
        ]
        
        tb_log_dir = None
        for pattern in tb_log_patterns:
            candidate_dir = self.session_root / "tensorboard" / pattern
            if candidate_dir.exists():
                tb_log_dir = candidate_dir
                break
        
        if tb_log_dir and tb_log_dir.exists():
            try:
                rewards, timesteps = extract_tensorboard_data(tb_log_dir)
                self.tensorboard_data[f"{env_name}_{agent_name}"] = {
                    'rewards': rewards,
                    'timesteps': timesteps
                }
                print(f"  ✅ Extracted TensorBoard data from {tb_log_dir.name}")
            except Exception as e:
                print(f"  ⚠️ Warning: Could not extract TensorBoard data: {e}")
        else:
            print(f"  ⚠️ Warning: No TensorBoard logs found for {env_name}_{agent_name}")
        
        vec_env.close()
        
        return results
    
    def run_comparison(self, env_name: str, agent_names: List[str], 
                      seed: Optional[int] = None) -> Dict[str, Any]:
        """Compare multiple agents on single environment."""
        print(f"\n🎯 Comparing agents on {env_name}")
        print(f"Agents: {', '.join(agent_names)}")
        
        models = {}
        individual_results = {}
        
        for i, agent_name in enumerate(agent_names):
            print(f"\n[{i+1}/{len(agent_names)}] Training {agent_name}...")
            
            # Run single agent
            result = self.run_single(env_name, agent_name, seed)
            individual_results[agent_name] = result
            
            # Load trained model for comparison
            model_path = self.session_root / "models" / f"{env_name}_{agent_name}.zip"
            agent = self._load_agent(agent_name)
            models[agent_name] = agent.load_model(str(model_path))
        
        # Generate comparison
        comparison_results = self.evaluator.compare_agents(models, env_name, n_episodes=20, seed=seed)
        
        # Combine results
        final_results = {
            'individual_results': individual_results,
            'comparison_summary': comparison_results['comparison_summary'],
            'env_name': env_name,
            'agent_names': agent_names,
            'n_episodes': 20,
            'tensorboard_data': self.tensorboard_data
        }
        
        return final_results
    
    def run_comprehensive(self, env_names: Optional[List[str]] = None, 
                         agent_names: Optional[List[str]] = None,
                         seed: Optional[int] = None) -> Dict[str, Any]:
        """Run comprehensive benchmark across all environments and agents."""
        # Use all environments if none specified
        if env_names is None:
            env_names = self.env_factory.get_available_environments()
        
        # Use all agents if none specified
        if agent_names is None:
            agent_names = self.available_agents
        
        print(f"\n🚀 Running comprehensive benchmark")
        print(f"Environments: {', '.join(env_names)}")
        print(f"Agents: {', '.join(agent_names)}")
        
        results = {}
        total_start_time = time.time()
        
        for env_name in env_names:
            print(f"\n{'='*60}")
            print(f"TESTING ENVIRONMENT: {env_name}")
            print(f"{'='*60}")
            
            try:
                env_results = self.run_comparison(env_name, agent_names, seed)
                results[env_name] = env_results
                
                # Print summary
                if 'comparison_summary' in env_results:
                    summary = env_results['comparison_summary']
                    print(f"\n🎯 RESULTS for {env_name}:")
                    for agent, reward in summary['agent_ranking']:
                        print(f"  {agent}: {reward:.2f}")
                    print(f"  Winner: {summary['best_agent']}")
                
            except Exception as e:
                print(f"❌ Failed to test {env_name}: {e}")
                import traceback
                traceback.print_exc()
        
        total_time = time.time() - total_start_time
        
        # Generate summary
        results['summary'] = self._generate_summary(results, total_time, len(env_names), len(agent_names))
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _load_agent(self, agent_name: str) -> BaseAgent:
        """Load agent instance."""
        # Get agent class from registry
        agent_class = get_agent_class(agent_name)
        
        # Create basic config for agent
        agent_config = {
            'name': agent_name,
            'description': self.agent_info.get(agent_name, {}).get('description', f'{agent_name} agent')
        }
        
        return agent_class(agent_config)
    
    def _generate_summary(self, results: Dict[str, Any], total_time: float, 
                         n_envs: int, n_agents: int) -> Dict[str, Any]:
        """Generate summary statistics."""
        total_episodes = 0
        total_training_time = 0
        
        for env_name, env_results in results.items():
            if env_name == 'summary':
                continue
            
            if 'individual_results' in env_results:
                for agent_name, result in env_results['individual_results'].items():
                    total_episodes += result.get('n_episodes', 0)
                    total_training_time += result.get('training_time', 0)
        
        return {
            'total_environments': n_envs,
            'total_agents': n_agents,
            'total_time': total_time,
            'total_training_time': total_training_time,
            'total_episodes': total_episodes,
            'timestamp': self.timestamp,
            'session_root': str(self.session_root)
        }
    
    def _save_results(self, results: Dict[str, Any]):
        """Save results in multiple formats."""
        print(f"\n📊 Saving results to {self.session_root}")
        
        # Generate reports
        self.reporter.generate_comparison_report(results)
        self.reporter.generate_json_report(results)
        self.reporter.generate_summary_table(results)
        self.md_reporter.generate_markdown_report(results)
        
        # Save raw results
        with open(self.session_root / "raw_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Reports generated:")
        print(f"  - comparison_report.txt")
        print(f"  - results.json")
        print(f"  - summary_table.csv")
        print(f"  - README.md")
    
    def print_final_summary(self, results: Dict[str, Any]):
        """Print final summary to console."""
        print(f"\n🎯 MULTI-AGENT BENCHMARK COMPLETE")
        print(f"{'='*70}")
        
        for env_name, env_results in results.items():
            if env_name == 'summary':
                continue
            
            print(f"\n {env_name}:")
            if 'comparison_summary' in env_results:
                summary = env_results['comparison_summary']
                for agent, reward in summary['agent_ranking']:
                    print(f"   {agent}: {reward:.2f}")
                print(f"   Winner: {summary['best_agent']}")
        
        if 'summary' in results:
            summary = results['summary']
            print(f"\n📊 Overall Statistics:")
            print(f"   Total Training Time: {summary['total_training_time']:.2f}s")
            print(f"   Total Episodes: {summary['total_episodes']}")
            print(f"   Session: {summary['session_root']}")
    
    def generate_plots(self, results: Dict[str, Any]) -> None:
        """Generate visualization plots using the visualization module."""
        from visualization.plotter import LearningCurvePlotter, ComparisonPlotter
        
        # Create plots directory
        plots_dir = self.session_root / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        print(f"📊 Generating plots in {plots_dir}")
        
        try:
            # Initialize plotters
            learning_plotter = LearningCurvePlotter(plots_dir)
            comparison_plotter = ComparisonPlotter(plots_dir)
            
            # Generate learning curves for each environment
            for env_name, env_results in results.items():
                if env_name == 'summary':
                    continue
                    
                if 'tensorboard_data' in env_results:
                    # Convert tensorboard_data to the expected format for the plotter
                    learning_data = {}
                    for key, tb_data in env_results['tensorboard_data'].items():
                        # Extract agent name from key (format: "CartPole-v1_standard_ppo")
                        if '_' in key:
                            agent_name = key.split('_', 1)[1]  # Get everything after first underscore
                        else:
                            agent_name = key
                        
                        # Handle both dictionary format and tuple format
                        if isinstance(tb_data, dict):
                            rewards = tb_data.get('rewards', [])
                            timesteps = tb_data.get('timesteps', [])
                            learning_data[agent_name] = (rewards, timesteps)
                        else:
                            # Assume it's already in tuple format
                            learning_data[agent_name] = tb_data
                    
                    if learning_data:
                        learning_plotter.plot_learning_curves(
                            learning_data,
                            env_name
                        )
                
                if 'individual_results' in env_results:
                    # For single environment, create a result structure for comparison plotter
                    single_env_results = {env_name: env_results['individual_results']}
                    comparison_plotter.plot_comparison_matrix(
                        single_env_results
                    )
            
            # Generate overall comparison if multiple environments
            env_names = [k for k in results.keys() if k != 'summary']
            if len(env_names) > 1:
                comparison_plotter.create_publication_plots(results)
                
            print(f"✅ Plots generated successfully")
            
        except Exception as e:
            print(f"⚠️ Error generating plots: {e}")
    
    def generate_reports(self, results: Dict[str, Any]) -> None:
        """Generate reports using the reporter modules."""
        print(f"📝 Generating reports in {self.session_root}")
        
        try:
            # Generate comparison report
            comparison_path = self.reporter.generate_comparison_report(results)
            print(f"   ✅ Comparison report: {comparison_path}")
            
            # Generate JSON report
            json_path = self.reporter.generate_json_report(results)
            print(f"   ✅ JSON report: {json_path}")
            
            # Generate CSV summary
            csv_path = self.reporter.generate_summary_table(results)
            print(f"   ✅ CSV summary: {csv_path}")
            
            # Generate markdown report
            md_path = self.md_reporter.generate_markdown_report(results)
            print(f"   ✅ Markdown report: {md_path}")
            
            print(f"✅ Reports generated successfully")
            
        except Exception as e:
            print(f"⚠️ Error generating reports: {e}")