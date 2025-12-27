"""
Tests for data handling utilities.
"""

import torch
import pytest
import numpy as np

from lstm_classifier.data import (
    EventDataset,
    UnsupervisedDataset,
    ContrastiveDataset,
    SyntheticEventDataset,
)
from lstm_classifier.models import TemporalAugmentation


class TestEventDataset:
    """Tests for EventDataset."""
    
    def test_creation(self):
        """Test dataset creation."""
        data = np.random.randn(10, 100, 6).astype(np.float32)
        labels = np.random.randint(0, 2, (10, 16)).astype(np.float32)
        timings = np.random.randint(0, 100, (10, 16)).astype(np.float32)
        
        dataset = EventDataset(data, labels, timings)
        
        assert len(dataset) == 10
        
        item = dataset[0]
        assert "input" in item
        assert "event_labels" in item
        assert "event_timings" in item
    
    def test_without_labels(self):
        """Test dataset without labels."""
        data = np.random.randn(10, 100, 6).astype(np.float32)
        dataset = EventDataset(data)
        
        item = dataset[0]
        assert "input" in item
        assert "event_labels" not in item


class TestSyntheticDataset:
    """Tests for SyntheticEventDataset."""
    
    def test_generation(self):
        """Test synthetic data generation."""
        dataset = SyntheticEventDataset(
            num_samples=10,
            seq_len=200,
            input_size=6,
            num_events=16,
        )
        
        assert len(dataset) == 10
        
        item = dataset[0]
        assert item["input"].shape == (200, 6)
        assert item["event_labels"].shape == (16,)
        assert item["event_timings"].shape == (16,)
    
    def test_refractory_period(self):
        """Test that synthetic data respects some constraints."""
        dataset = SyntheticEventDataset(
            num_samples=5,
            seq_len=8000,  # 4 seconds
            num_events=16,
        )
        
        # Just verify it runs without errors
        item = dataset[0]
        assert item["input"].shape[0] == 8000


class TestUnsupervisedDataset:
    """Tests for UnsupervisedDataset."""
    
    def test_creation(self):
        """Test unsupervised dataset."""
        data = np.random.randn(10, 100, 6).astype(np.float32)
        dataset = UnsupervisedDataset(data)
        
        assert len(dataset) == 10
        
        item = dataset[0]
        assert item.shape == (100, 6)


class TestContrastiveDataset:
    """Tests for ContrastiveDataset."""
    
    def test_creation(self):
        """Test contrastive dataset."""
        data = np.random.randn(10, 100, 6).astype(np.float32)
        dataset = ContrastiveDataset(
            data,
            augmentation=TemporalAugmentation.random_augment,
        )
        
        assert len(dataset) == 10
        
        view1, view2 = dataset[0]
        assert view1.shape == (100, 6)
        assert view2.shape == (100, 6)
        
        # Views should be different (augmented)
        assert not torch.allclose(view1, view2)


class TestTemporalAugmentation:
    """Tests for temporal augmentations."""
    
    def test_add_noise(self):
        """Test noise augmentation."""
        x = torch.randn(4, 100, 6)
        x_aug = TemporalAugmentation.add_noise(x, noise_std=0.01)
        
        assert x_aug.shape == x.shape
        assert not torch.allclose(x, x_aug)
    
    def test_scale(self):
        """Test scaling augmentation."""
        x = torch.randn(4, 100, 6)
        x_aug = TemporalAugmentation.scale(x, scale_range=(0.9, 1.1))
        
        assert x_aug.shape == x.shape
    
    def test_temporal_crop(self):
        """Test temporal crop."""
        x = torch.randn(4, 100, 6)
        x_aug = TemporalAugmentation.temporal_crop(x, crop_ratio=0.9)
        
        assert x_aug.shape == x.shape
    
    def test_temporal_shift(self):
        """Test temporal shift."""
        x = torch.randn(4, 100, 6)
        x_aug = TemporalAugmentation.temporal_shift(x, max_shift=10)
        
        assert x_aug.shape == x.shape
    
    def test_random_augment(self):
        """Test random augmentation."""
        x = torch.randn(4, 100, 6)
        x_aug = TemporalAugmentation.random_augment(x)
        
        assert x_aug.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__])
