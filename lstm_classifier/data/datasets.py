"""
Dataset classes for supervised and unsupervised training.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Tuple, Dict, List


class EventDataset(Dataset):
    """
    Dataset for supervised event classification and timing prediction.
    
    Args:
        data: Input sequences of shape (num_samples, seq_len, input_size)
        event_labels: Binary event labels of shape (num_samples, num_events)
        event_timings: Event timing in samples of shape (num_samples, num_events)
        sample_rate: Sampling rate in Hz (default: 2000)
        transform: Optional transform to apply to data
    """
    
    def __init__(
        self,
        data: np.ndarray,
        event_labels: Optional[np.ndarray] = None,
        event_timings: Optional[np.ndarray] = None,
        sample_rate: int = 2000,
        transform=None,
    ):
        self.data = torch.FloatTensor(data)
        self.event_labels = (
            torch.FloatTensor(event_labels) if event_labels is not None else None
        )
        self.event_timings = (
            torch.FloatTensor(event_timings) if event_timings is not None else None
        )
        self.sample_rate = sample_rate
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = self.data[idx]
        
        if self.transform is not None:
            x = self.transform(x)
        
        item = {"input": x}
        
        if self.event_labels is not None:
            item["event_labels"] = self.event_labels[idx]
        
        if self.event_timings is not None:
            item["event_timings"] = self.event_timings[idx]
        
        return item


class UnsupervisedDataset(Dataset):
    """
    Dataset for unsupervised pretraining.
    
    Args:
        data: Input sequences of shape (num_samples, seq_len, input_size)
        sample_rate: Sampling rate in Hz (default: 2000)
        transform: Optional transform to apply to data
    """
    
    def __init__(
        self,
        data: np.ndarray,
        sample_rate: int = 2000,
        transform=None,
    ):
        self.data = torch.FloatTensor(data)
        self.sample_rate = sample_rate
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        x = self.data[idx]
        
        if self.transform is not None:
            x = self.transform(x)
        
        return x


class ContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning.
    
    Creates two augmented views of each sample for contrastive learning.
    
    Args:
        data: Input sequences of shape (num_samples, seq_len, input_size)
        augmentation: Augmentation function that takes a tensor and returns augmented tensor
        sample_rate: Sampling rate in Hz (default: 2000)
    """
    
    def __init__(
        self,
        data: np.ndarray,
        augmentation,
        sample_rate: int = 2000,
    ):
        self.data = torch.FloatTensor(data)
        self.augmentation = augmentation
        self.sample_rate = sample_rate
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx]
        
        # Create two augmented views
        x1 = self.augmentation(x.unsqueeze(0)).squeeze(0)
        x2 = self.augmentation(x.unsqueeze(0)).squeeze(0)
        
        return x1, x2


class SyntheticEventDataset(EventDataset):
    """
    Synthetic dataset generator for testing and development.
    
    Generates synthetic event data with known patterns for testing the model
    before real data is available.
    
    Args:
        num_samples: Number of samples to generate
        seq_len: Length of each sequence
        input_size: Number of input features (default: 6)
        num_events: Number of event types (default: 16)
        sample_rate: Sampling rate in Hz (default: 2000)
        event_probability: Probability of each event occurring (default: 0.3)
        noise_std: Standard deviation of Gaussian noise (default: 0.1)
    """
    
    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        input_size: int = 6,
        num_events: int = 16,
        sample_rate: int = 2000,
        event_probability: float = 0.3,
        noise_std: float = 0.1,
    ):
        # Generate synthetic data
        data, event_labels, event_timings = self._generate_synthetic_data(
            num_samples, seq_len, input_size, num_events, 
            event_probability, noise_std, sample_rate
        )
        
        super().__init__(data, event_labels, event_timings, sample_rate)
    
    @staticmethod
    def _generate_synthetic_data(
        num_samples: int,
        seq_len: int,
        input_size: int,
        num_events: int,
        event_probability: float,
        noise_std: float,
        sample_rate: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic event data."""
        np.random.seed(42)  # For reproducibility
        
        data = []
        event_labels = []
        event_timings = []
        
        refractory_period = 4000  # 2 seconds at 2000 Hz
        
        for _ in range(num_samples):
            # Start with baseline signal
            sample = np.random.randn(seq_len, input_size) * noise_std
            
            # Event labels for this sample
            sample_events = np.random.rand(num_events) < event_probability
            sample_timings = np.zeros(num_events)
            
            # Add events
            for event_idx in range(num_events):
                if sample_events[event_idx]:
                    # Random timing for this event
                    # Ensure we have enough space in the sequence
                    margin = min(100, seq_len // 4)
                    if seq_len > 2 * margin:
                        event_time = np.random.randint(margin, seq_len - margin)
                    else:
                        event_time = seq_len // 2
                    sample_timings[event_idx] = event_time
                    
                    # Add event signature to data
                    # Each event has a unique pattern
                    event_duration = 50  # Duration of event signature
                    start = max(0, event_time - event_duration // 2)
                    end = min(seq_len, event_time + event_duration // 2)
                    
                    # Unique signature for each event type
                    signature = np.sin(
                        np.linspace(0, 2 * np.pi * (event_idx + 1), end - start)
                    )
                    
                    # Apply signature to specific channels
                    channels = [event_idx % input_size, (event_idx + 1) % input_size]
                    for ch in channels:
                        sample[start:end, ch] += signature[:end - start] * 0.5
            
            data.append(sample)
            event_labels.append(sample_events.astype(np.float32))
            event_timings.append(sample_timings.astype(np.float32))
        
        return (
            np.array(data, dtype=np.float32),
            np.array(event_labels, dtype=np.float32),
            np.array(event_timings, dtype=np.float32),
        )


def collate_fn_supervised(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for supervised training with variable length sequences.
    
    Args:
        batch: List of dictionaries from EventDataset
        
    Returns:
        Dictionary with batched tensors and sequence lengths
    """
    inputs = torch.stack([item["input"] for item in batch])
    
    result = {
        "input": inputs,
        "lengths": torch.LongTensor([len(item["input"]) for item in batch]),
    }
    
    if "event_labels" in batch[0]:
        result["event_labels"] = torch.stack([item["event_labels"] for item in batch])
    
    if "event_timings" in batch[0]:
        result["event_timings"] = torch.stack([item["event_timings"] for item in batch])
    
    return result


def collate_fn_contrastive(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for contrastive learning.
    
    Args:
        batch: List of (view1, view2) tuples
        
    Returns:
        Tuple of (batch_view1, batch_view2)
    """
    view1 = torch.stack([item[0] for item in batch])
    view2 = torch.stack([item[1] for item in batch])
    
    return view1, view2
