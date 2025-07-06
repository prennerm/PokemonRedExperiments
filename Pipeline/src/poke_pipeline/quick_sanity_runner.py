#!/usr/bin/env python3
"""
Quick runner for different levels of sanity checks
"""

import sys
import argparse
from pathlib import Path

def run_basic_check():
    """Run the simple 1x4 grid check"""
    print(" Running BASIC sanity check (1x4 Grid)...")
    from poke_pipeline.simple_check import run_sanity_check
    run_sanity_check()

def run_intermediate_check():
    """Run intermediate checks with 2-3 environments"""
    print(" Running INTERMEDIATE sanity check...")
    from advanced_sanity_check import AdvancedSanityCheck
    
    checker = AdvancedSanityCheck()
    # Test only easier environments
    environments = ["FrozenLake-v1", "CartPole-v1"]
    results = checker.run_comprehensive_check(environments)
    
    passed = sum(1 for r in results.values() if r['passed'])
    total = len(results)
    print(f" Intermediate check: {passed}/{total} passed")

def run_full_check():
    """Run complete advanced check with all environments"""
    print(" Running FULL advanced sanity check...")
    from advanced_sanity_check import main
    main()

def main():
    parser = argparse.ArgumentParser(description="λ-Discrepancy PPO Sanity Check Runner")
    parser.add_argument('level', choices=['basic', 'intermediate', 'full'], 
                       help='Level of sanity check to run')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.level == 'basic':
        run_basic_check()
    elif args.level == 'intermediate':
        run_intermediate_check()
    elif args.level == 'full':
        run_full_check()

if __name__ == "__main__":
    main()