#!/usr/bin/env python3
"""
test_callback.py
Simple test to verify that the StatsCallback doesn't cause training to stop prematurely.
"""
import numpy as np
from pathlib import Path
import tempfile
from src.poke_pipeline.callbacks import StatsCallback


def test_callback_robustness():
    """Test that the callback doesn't crash on various edge cases."""
    with tempfile.TemporaryDirectory() as temp_dir:
        callback = StatsCallback(save_freq=5, save_path=temp_dir, verbose=1)
        
        # Simulate the callback being attached to a model
        class MockModel:
            def __init__(self):
                self.num_timesteps = 1000
        
        callback.model = MockModel()
        callback.num_timesteps = 1000
        
        # Test 1: Test _on_step returns True
        result = callback._on_step()
        assert result is True, "Callback should always return True from _on_step"
        
        # Test 2: Test _restructure_stats with normal data
        normal_stats = {
            "step": 42,
            "x": 10,
            "y": 20,
            "map": 1,
            "reward_total": 5.0,
            "hp": 100,
            "levels": [15, 12, 8],
            "badge": 2
        }
        
        restructured = callback._restructure_stats(normal_stats)
        assert restructured["step"] == 42
        assert restructured["position"]["x"] == 10
        assert restructured["rewards"]["total"] == 5.0
        
        # Test 3: Test _restructure_stats with missing data
        minimal_stats = {"step": 1}
        restructured_minimal = callback._restructure_stats(minimal_stats)
        assert restructured_minimal["step"] == 1
        assert "total_steps" in restructured_minimal
        
        # Test 4: Test _restructure_stats with corrupted data
        corrupted_stats = {"step": "invalid_step_value"}
        restructured_corrupted = callback._restructure_stats(corrupted_stats)
        # Should not crash and return some fallback structure
        assert "step" in restructured_corrupted
        
        print("✅ All StatsCallback tests passed!")
        

if __name__ == "__main__":
    test_callback_robustness()
