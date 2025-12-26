"""
Tests for refractory period enforcement and post-processing.
"""

import pytest
import numpy as np
import torch

from lstm_classifier.utils import RefractoryPeriodEnforcer, non_maximum_suppression


class TestRefractoryPeriodEnforcer:
    """Tests for RefractoryPeriodEnforcer."""
    
    def test_single_event_no_violation(self):
        """Test single event with no refractory violation."""
        enforcer = RefractoryPeriodEnforcer(
            num_events=16,
            refractory_period_samples=4000,
        )
        
        detected = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        timings = np.array([100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        
        filtered_events, filtered_timings = enforcer.enforce(detected, timings)
        
        # Should remain unchanged
        np.testing.assert_array_equal(filtered_events, detected)
        np.testing.assert_array_equal(filtered_timings, timings)
    
    def test_refractory_violation(self):
        """Test that refractory violations are detected and handled."""
        enforcer = RefractoryPeriodEnforcer(
            num_events=16,
            refractory_period_samples=4000,
        )
        
        # First detection
        detected1 = np.zeros(16)
        detected1[0] = 1
        timings1 = np.zeros(16)
        timings1[0] = 100
        
        enforcer.enforce(detected1, timings1, global_offset=0)
        
        # Second detection within refractory period (should be rejected)
        detected2 = np.zeros(16)
        detected2[0] = 1
        timings2 = np.zeros(16)
        timings2[0] = 100  # Only 100 samples later (< 4000)
        
        filtered_events, filtered_timings = enforcer.enforce(
            detected2, timings2, global_offset=100
        )
        
        # Event 0 should be rejected
        assert filtered_events[0] == 0
    
    def test_reset(self):
        """Test resetting enforcer state."""
        enforcer = RefractoryPeriodEnforcer(num_events=16)
        
        detected = np.zeros(16)
        detected[0] = 1
        timings = np.zeros(16)
        timings[0] = 100
        
        enforcer.enforce(detected, timings)
        assert len(enforcer.last_event_times) > 0
        
        enforcer.reset()
        assert len(enforcer.last_event_times) == 0
    
    def test_batch_enforcement(self):
        """Test batch enforcement."""
        enforcer = RefractoryPeriodEnforcer(num_events=16)
        
        batch_size = 4
        detected = np.random.randint(0, 2, (batch_size, 16)).astype(float)
        timings = np.random.randint(0, 1000, (batch_size, 16)).astype(float)
        
        filtered_events, filtered_timings = enforcer.enforce_batch(
            detected, timings
        )
        
        assert filtered_events.shape == (batch_size, 16)
        assert filtered_timings.shape == (batch_size, 16)


class TestNonMaximumSuppression:
    """Tests for NMS."""
    
    def test_nms_basic(self):
        """Test basic NMS functionality."""
        # Create simple probability distribution
        event_probs = np.zeros((100, 16))
        event_probs[50, 0] = 0.9  # Strong peak at t=50 for event 0
        event_probs[51, 0] = 0.7  # Weaker nearby peak
        
        event_timings = np.tile(np.arange(100)[:, np.newaxis], (1, 16))
        
        detected, timings, probs = non_maximum_suppression(
            event_probs,
            event_timings,
            threshold=0.5,
            nms_threshold=50,
        )
        
        # Should detect event 0
        assert detected[0] == 1
        # Should pick the stronger peak
        assert timings[0] == 50
        assert probs[0] == 0.9


if __name__ == "__main__":
    pytest.main([__file__])
