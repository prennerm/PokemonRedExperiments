"""
Report generation utilities.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np


class ComparisonReporter:
    """Generate comparison reports for multi-agent benchmarks."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_comparison_report(self, results: Dict[str, Any], 
                                  title: str = "Multi-Agent Benchmark Results") -> Path:
        """Generate comprehensive comparison report."""
        report_path = self.output_dir / "comparison_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"{title}\n")
            f.write("=" * len(title) + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary statistics
            if 'summary' in results:
                f.write("SUMMARY\n")
                f.write("-" * 40 + "\n")
                summary = results['summary']
                f.write(f"Total environments tested: {summary.get('total_environments', 0)}\n")
                f.write(f"Total agents tested: {summary.get('total_agents', 0)}\n")
                f.write(f"Total training time: {summary.get('total_training_time', 0):.2f} seconds\n")
                f.write(f"Total evaluation episodes: {summary.get('total_episodes', 0)}\n\n")
            
            # Environment-by-environment results
            for env_name, env_results in results.items():
                if env_name == 'summary':
                    continue
                
                f.write(f"ENVIRONMENT: {env_name}\n")
                f.write("-" * 60 + "\n")
                
                if 'individual_results' in env_results:
                    individual = env_results['individual_results']
                    
                    # Performance table
                    f.write("Performance Results:\n")
                    f.write(f"{'Agent':<20} {'Mean Reward':<15} {'Std Reward':<15} {'Success Rate':<15}\n")
                    f.write("-" * 65 + "\n")
                    
                    for agent_name, result in individual.items():
                        f.write(f"{agent_name:<20} {result['mean_reward']:<15.2f} "
                               f"{result['std_reward']:<15.2f} {result['success_rate']:<15.2f}\n")
                    
                    f.write("\n")
                    
                    # Comparison summary
                    if 'comparison_summary' in env_results:
                        summary = env_results['comparison_summary']
                        f.write("Comparison Summary:\n")
                        f.write(f"Best Agent: {summary['best_agent']} ({summary['best_reward']:.2f})\n")
                        f.write(f"Worst Agent: {summary['worst_agent']} ({summary['worst_reward']:.2f})\n")
                        f.write(f"Performance Gap: {summary['performance_gap']:.2f}\n")
                        
                        f.write("\nAgent Ranking:\n")
                        for rank, (agent, reward) in enumerate(summary['agent_ranking'], 1):
                            f.write(f"{rank}. {agent}: {reward:.2f}\n")
                    
                    f.write("\n")
                    
                    # Detailed statistics
                    f.write("Detailed Statistics:\n")
                    for agent_name, result in individual.items():
                        f.write(f"\n{agent_name}:\n")
                        f.write(f"  Mean Reward: {result['mean_reward']:.4f} ± {result['std_reward']:.4f}\n")
                        f.write(f"  Mean Episode Length: {result['mean_length']:.2f} ± {result['std_length']:.2f}\n")
                        f.write(f"  Success Rate: {result['success_rate']:.2%}\n")
                        f.write(f"  Evaluation Time: {result.get('evaluation_time', 0):.2f}s\n")
                        f.write(f"  Model Type: {result.get('model_type', 'unknown')}\n")
                        f.write(f"  Model Size: {result.get('model_size', 0)} parameters\n")
                        f.write(f"  Baseline Passed: {result.get('baseline_passed', False)}\n")
                        f.write(f"  Improvement over Random: {result.get('improvement_over_random', 0):.2f}\n")
                
                f.write("\n" + "=" * 60 + "\n\n")
        
        return report_path
    
    def generate_json_report(self, results: Dict[str, Any]) -> Path:
        """Generate JSON report for programmatic access."""
        report_path = self.output_dir / "results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, (bool, np.bool_)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            else:
                return obj
        
        clean_results = convert_numpy(results)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False)
        
        return report_path
    
    def generate_summary_table(self, results: Dict[str, Any]) -> Path:
        """Generate CSV summary table."""
        report_path = self.output_dir / "summary_table.csv"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("Environment,Agent,Mean_Reward,Std_Reward,Success_Rate,Baseline_Passed,Model_Type\n")
            
            # Data rows
            for env_name, env_results in results.items():
                if env_name == 'summary':
                    continue
                
                if 'individual_results' in env_results:
                    for agent_name, result in env_results['individual_results'].items():
                        f.write(f"{env_name},{agent_name},{result['mean_reward']:.4f},"
                               f"{result['std_reward']:.4f},{result['success_rate']:.4f},"
                               f"{result.get('baseline_passed', False)},{result.get('model_type', 'unknown')}\n")
        
        return report_path


class MarkdownReporter:
    """Generate markdown reports for documentation."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_markdown_report(self, results: Dict[str, Any], 
                                title: str = "Multi-Agent Benchmark Results") -> Path:
        """Generate markdown report with embedded plots."""
        report_path = self.output_dir / "README.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# {title}\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            if 'summary' in results:
                f.write("## Summary\n\n")
                summary = results['summary']
                f.write(f"- **Total environments tested**: {summary.get('total_environments', 0)}\n")
                f.write(f"- **Total agents tested**: {summary.get('total_agents', 0)}\n")
                f.write(f"- **Total training time**: {summary.get('total_training_time', 0):.2f} seconds\n")
                f.write(f"- **Total evaluation episodes**: {summary.get('total_episodes', 0)}\n\n")
            
            # Results by environment
            f.write("## Results by Environment\n\n")
            
            for env_name, env_results in results.items():
                if env_name == 'summary':
                    continue
                
                f.write(f"### {env_name}\n\n")
                
                if 'individual_results' in env_results:
                    individual = env_results['individual_results']
                    
                    # Performance table
                    f.write("| Agent | Mean Reward | Std Reward | Success Rate | Baseline Passed |\n")
                    f.write("|-------|-------------|------------|--------------|----------------|\n")
                    
                    for agent_name, result in individual.items():
                        baseline_passed = "✅" if result.get('baseline_passed', False) else "❌"
                        f.write(f"| {agent_name} | {result['mean_reward']:.2f} | "
                               f"{result['std_reward']:.2f} | {result['success_rate']:.2%} | {baseline_passed} |\n")
                    
                    f.write("\n")
                    
                    # Plot reference
                    plot_path = f"plots/{env_name}_comparison.png"
                    f.write(f"![{env_name} Learning Curves]({plot_path})\n\n")
                    
                    # Winner
                    if 'comparison_summary' in env_results:
                        summary = env_results['comparison_summary']
                        f.write(f"**Winner**: {summary['best_agent']} ({summary['best_reward']:.2f})\n\n")
            
            # Methodology
            f.write("## Methodology\n\n")
            f.write("- **Evaluation**: 20 episodes per agent per environment\n")
            f.write("- **Reproducibility**: Fixed seeds for deterministic evaluation\n")
            f.write("- **Success Criteria**: Environment-specific thresholds\n")
            f.write("- **No Reward Shaping**: Environments used as-is for scientific validity\n\n")
            
            # Files
            f.write("## Generated Files\n\n")
            f.write("- `comparison_report.txt`: Detailed text report\n")
            f.write("- `results.json`: Machine-readable results\n")
            f.write("- `summary_table.csv`: CSV summary for analysis\n")
            f.write("- `plots/`: Learning curve visualizations\n")
            f.write("- `models/`: Trained model files\n")
            f.write("- `tensorboard/`: TensorBoard logs\n")
        
        return report_path