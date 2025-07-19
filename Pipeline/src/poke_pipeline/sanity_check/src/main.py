#!/usr/bin/env python3
"""
Multi-Agent Benchmark Suite - Main Entry Point

CLI interface for running systematic comparisons of three different 
Reinforcement Learning agents on standard Gym environments.

Usage examples:
    python main.py --agent standard_ppo --env CartPole-v1
    python main.py --compare --env LunarLander-v3
    python main.py --comprehensive
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional, List

# Add the src directory to Python path for imports
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

from agents import get_agent_class, list_available_agents
from core.runner import SanityCheckRunner
from core.environment import EnvironmentFactory
from utils.paths import create_session_directory, get_project_root

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Multi-Agent Benchmark Suite for RL algorithm comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train single agent on single environment
  python main.py --agent standard_ppo --env CartPole-v1
  
  # Compare all agents on one environment
  python main.py --compare --env LunarLander-v3
  
  # Run comprehensive benchmark (all agents, all environments)
  python main.py --comprehensive
  
  # Run with custom output directory
  python main.py --agent lstm --env MountainCar-v0 --output ./results
  
  # List available agents and environments
  python main.py --list
        """
    )
    
    # Execution modes (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--agent', 
        type=str,
        help='Train single agent (standard_ppo, lstm, ld)'
    )
    mode_group.add_argument(
        '--compare', 
        action='store_true',
        help='Compare all agents on single environment'
    )
    mode_group.add_argument(
        '--comprehensive', 
        action='store_true',
        help='Run comprehensive benchmark (all agents, all environments)'
    )
    mode_group.add_argument(
        '--list', 
        action='store_true',
        help='List available agents and environments'
    )
    
    # Environment selection
    parser.add_argument(
        '--env', 
        type=str,
        help='Environment name (CartPole-v1, MountainCar-v0, LunarLander-v3, FrozenLake-v1)'
    )
    
    # Configuration and paths
    parser.add_argument(
        '--output', 
        type=str,
        help='Output directory for results (default: timestamped session dir)'
    )
    
    # Training parameters
    parser.add_argument(
        '--timesteps', 
        type=int,
        help='Number of training timesteps (overrides config)'
    )
    parser.add_argument(
        '--seed', 
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    # Evaluation parameters
    parser.add_argument(
        '--eval-episodes', 
        type=int,
        default=20,
        help='Number of episodes for evaluation'
    )
    parser.add_argument(
        '--no-evaluation', 
        action='store_true',
        help='Skip post-training evaluation'
    )
    
    # Output options
    parser.add_argument(
        '--no-plots', 
        action='store_true',
        help='Skip plot generation'
    )
    parser.add_argument(
        '--no-reports', 
        action='store_true',
        help='Skip report generation'
    )
    
    # Debug options
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Print configuration and exit without training'
    )
    
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    # Environment required for single agent and compare modes
    if (args.agent or args.compare) and not args.env:
        raise ValueError("--env is required when using --agent or --compare")
    
    # Validate agent name
    if args.agent:
        available_agents = list_available_agents()
        if args.agent not in available_agents:
            raise ValueError(f"Unknown agent '{args.agent}'. Available: {available_agents}")


def list_available_options() -> None:
    """List available agents and environments."""
    print("🤖 Available Agents:")
    agents = list_available_agents()
    for agent in sorted(set(agents)):  # Remove duplicates from aliases
        if agent in ['standard_ppo', 'lstm', 'ld']:  # Show only main names
            print(f"  - {agent}")
    
    print("\n🏋️ Available Environments:")
    try:
        # Get environments from factory (no YAML needed)
        factory = EnvironmentFactory()
        environments = factory.get_available_environments()
        for env in sorted(environments):
            print(f"  - {env}")
    except Exception as e:
        print(f"  Error loading environments: {e}")


def main() -> int:
    """Main entry point."""
    try:
        # Parse arguments
        parser = setup_argument_parser()
        args = parser.parse_args()
        
        # Configure debug logging
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")
        
        # Handle list mode
        if args.list:
            list_available_options()
            return 0
        
        # Validate arguments
        validate_arguments(args)
        
        # Create output directory
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = create_session_directory()
        
        logger.info(f"📁 Output directory: {output_dir}")
        
        # Initialize runner (no config needed anymore)
        runner = SanityCheckRunner(base_dir=output_dir)
        
        # Show configuration in dry-run mode
        if args.dry_run:
            print(f"\n🔍 Dry Run Configuration:")
            print(f"Output: {output_dir}")
            print(f"Seed: {args.seed if hasattr(args, 'seed') else 'default'}")
            if args.agent:
                print(f"Mode: Single agent ({args.agent}) on {args.env}")
            elif args.compare:
                print(f"Mode: Compare all agents on {args.env}")
            elif args.comprehensive:
                print(f"Mode: Comprehensive benchmark")
            return 0
        
        # Execute based on mode
        if args.agent:
            logger.info(f"🚀 Training single agent: {args.agent} on {args.env}")
            single_result = runner.run_single(
                env_name=args.env,
                agent_name=args.agent,
                seed=args.seed
            )
            
            # Convert single result to comparison format for plots/reports
            results = {
                args.env: {
                    'individual_results': {
                        args.agent: single_result
                    },
                    'comparison_summary': {
                        'best_agent': args.agent,
                        'best_reward': single_result.get('mean_reward', 0),
                        'worst_agent': args.agent,  # Same as best for single agent
                        'worst_reward': single_result.get('mean_reward', 0),  # Same as best for single agent
                        'performance_gap': 0.0,  # No gap for single agent
                        'agent_ranking': [(args.agent, single_result.get('mean_reward', 0))],
                        'mean_rewards': {args.agent: single_result.get('mean_reward', 0)}
                    },
                    'env_name': args.env,
                    'agent_names': [args.agent],
                    'n_episodes': single_result.get('n_episodes', 20),
                    'tensorboard_data': runner.tensorboard_data
                }
            }
            
        elif args.compare:
            logger.info(f"🏆 Comparing all agents on: {args.env}")
            results = runner.run_comparison(
                env_name=args.env,
                seed=args.seed
            )
            
        elif args.comprehensive:
            logger.info(f"🎯 Running comprehensive benchmark")
            results = runner.run_comprehensive(
                seed=args.seed
            )
        
        # Generate outputs
        if not args.no_plots:
            logger.info("📊 Generating plots...")
            runner.generate_plots(results)
        
        if not args.no_reports:
            logger.info("📝 Generating reports...")
            runner.generate_reports(results)
        
        logger.info("✅ Benchmark completed successfully!")
        logger.info(f"📁 Results saved to: {output_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("⚠️ Interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        if args.debug if 'args' in locals() else False:
            logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())
