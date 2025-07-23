#!/usr/bin/env python3
"""
Debug script to check all TensorBoard directories.
"""

from pathlib import Path
import sys

# Add the src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Path to the most recent results
results_dir = Path("results")
latest_result = max(results_dir.glob("*"), key=lambda x: x.name)
tensorboard_dir = latest_result / "tensorboard"

print(f"Checking TensorBoard logs in: {tensorboard_dir}")
print(f"Directory exists: {tensorboard_dir.exists()}")

if tensorboard_dir.exists():
    # List all subdirectories
    subdirs = [d for d in tensorboard_dir.iterdir() if d.is_dir()]
    print(f"\nFound {len(subdirs)} TensorBoard subdirectories:")
    for subdir in subdirs:
        print(f"  - {subdir.name}")
        # Check if it has event files
        event_files = list(subdir.glob("events.out.tfevents.*"))
        print(f"    Event files: {len(event_files)}")
        if event_files:
            print(f"    Example: {event_files[0].name}")
else:
    print("TensorBoard directory does not exist!")
