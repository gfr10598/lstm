"""
Utility functions for post-processing predictions and enforcing constraints.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional


class RefractoryPeriodEnforcer:
    """
    Enforces refractory period constraint on event predictions.
    
    This class maintains state across batches to ensure that events of the
    same type don't occur within the refractory period.
    
    Args:
        num_events: Number of event types
        refractory_period_samples: Refractory period in samples (default: 4000 for 2s at 2000Hz)
        sample_rate: Sampling rate in Hz (default: 2000)
    """
    
    def __init__(
        self,
        num_events: int = 16,
        refractory_period_samples: int = 4000,
        sample_rate: int = 2000,
    ):
        self.num_events = num_events
        self.refractory_period_samples = refractory_period_samples
        self.sample_rate = sample_rate
        
        # Track last occurrence time for each event type
        self.last_event_times = {}  # event_idx -> timestamp in samples
        
    def reset(self):
        """Reset the state (last event times)."""
        self.last_event_times = {}
    
    def enforce(
        self,
        detected_events: np.ndarray,
        event_timings: np.ndarray,
        event_probs: Optional[np.ndarray] = None,
        global_offset: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enforce refractory period on detected events.
        
        Args:
            detected_events: Binary array of detected events (num_events,)
            event_timings: Event timings in samples (num_events,)
            event_probs: Optional event probabilities for prioritization (num_events,)
            global_offset: Global time offset in samples for tracking across batches
            
        Returns:
            Tuple of (filtered_events, filtered_timings)
        """
        filtered_events = detected_events.copy()
        filtered_timings = event_timings.copy()
        
        for event_idx in range(self.num_events):
            if detected_events[event_idx]:
                current_time = global_offset + event_timings[event_idx]
                
                # Check if this event violates refractory period
                if event_idx in self.last_event_times:
                    time_since_last = current_time - self.last_event_times[event_idx]
                    
                    if time_since_last < self.refractory_period_samples:
                        # Violation: choose which one to keep based on probability
                        if event_probs is not None:
                            # Keep the one with higher probability
                            # For simplicity, we reject the current one
                            filtered_events[event_idx] = 0
                            filtered_timings[event_idx] = 0
                        else:
                            # Default: reject the current one
                            filtered_events[event_idx] = 0
                            filtered_timings[event_idx] = 0
                    else:
                        # No violation: update last occurrence time
                        self.last_event_times[event_idx] = current_time
                else:
                    # First occurrence of this event
                    self.last_event_times[event_idx] = current_time
        
        return filtered_events, filtered_timings
    
    def enforce_batch(
        self,
        detected_events: np.ndarray,
        event_timings: np.ndarray,
        event_probs: Optional[np.ndarray] = None,
        sequence_lengths: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enforce refractory period on a batch of predictions.
        
        Args:
            detected_events: Binary array of shape (batch_size, num_events)
            event_timings: Event timings in samples of shape (batch_size, num_events)
            event_probs: Optional probabilities of shape (batch_size, num_events)
            sequence_lengths: Optional sequence lengths for each sample in batch
            
        Returns:
            Tuple of (filtered_events, filtered_timings)
        """
        batch_size = detected_events.shape[0]
        filtered_events = np.zeros_like(detected_events)
        filtered_timings = np.zeros_like(event_timings)
        
        global_offset = 0
        for i in range(batch_size):
            probs = event_probs[i] if event_probs is not None else None
            
            filtered_events[i], filtered_timings[i] = self.enforce(
                detected_events[i],
                event_timings[i],
                probs,
                global_offset,
            )
            
            # Update global offset
            if sequence_lengths is not None:
                global_offset += sequence_lengths[i]
            else:
                # Assume timings are relative to sequence start
                # Update based on max timing in current sequence
                if detected_events[i].any():
                    max_timing = event_timings[i][detected_events[i] > 0].max()
                    global_offset += max_timing
        
        return filtered_events, filtered_timings


def non_maximum_suppression(
    event_probs: np.ndarray,
    event_timings: np.ndarray,
    threshold: float = 0.5,
    nms_threshold: float = 50,  # samples
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply non-maximum suppression to filter overlapping event predictions.
    
    Args:
        event_probs: Event probabilities of shape (seq_len, num_events)
        event_timings: Event timings of shape (seq_len, num_events)
        threshold: Probability threshold for detection
        nms_threshold: Temporal threshold in samples for NMS
        
    Returns:
        Tuple of (detected_events, event_timings, event_probs)
    """
    num_events = event_probs.shape[1]
    detected_events = np.zeros(num_events)
    final_timings = np.zeros(num_events)
    final_probs = np.zeros(num_events)
    
    for event_idx in range(num_events):
        # Get detections for this event
        event_det = event_probs[:, event_idx]
        
        # Find peaks above threshold
        peaks = []
        for t in range(len(event_det)):
            if event_det[t] > threshold:
                # Check if local maximum
                is_max = True
                for dt in range(-2, 3):
                    if 0 <= t + dt < len(event_det) and dt != 0:
                        if event_det[t] < event_det[t + dt]:
                            is_max = False
                            break
                if is_max:
                    peaks.append((t, event_det[t]))
        
        if peaks:
            # Sort by probability (descending)
            peaks.sort(key=lambda x: x[1], reverse=True)
            
            # Keep the strongest peak
            best_peak = peaks[0]
            detected_events[event_idx] = 1
            final_timings[event_idx] = best_peak[0]
            final_probs[event_idx] = best_peak[1]
    
    return detected_events, final_timings, final_probs


def smooth_predictions(
    predictions: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """
    Apply temporal smoothing to predictions.
    
    Args:
        predictions: Predictions of shape (seq_len, num_events)
        window_size: Size of smoothing window
        
    Returns:
        Smoothed predictions
    """
    from scipy.ndimage import uniform_filter1d
    
    return uniform_filter1d(predictions, size=window_size, axis=0)


def detect_events_from_sequence(
    model,
    sequence: np.ndarray,
    threshold: float = 0.5,
    apply_refractory: bool = True,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """
    Detect events from a single sequence.
    
    Args:
        model: Trained LSTMEventClassifier
        sequence: Input sequence of shape (seq_len, input_size)
        threshold: Probability threshold for detection
        apply_refractory: Whether to enforce refractory period
        device: Device to run inference on
        
    Returns:
        Dictionary with detection results
    """
    model.eval()
    model.to(device)
    
    # Convert to tensor and add batch dimension
    x = torch.FloatTensor(sequence).unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        predictions = model.predict_events(x, threshold, apply_refractory)
    
    # Convert to numpy
    return {
        "detected_events": predictions["detected_events"].cpu().numpy()[0],
        "event_timings": predictions["event_timings"].cpu().numpy()[0],
        "event_probs": predictions["event_probs"].cpu().numpy()[0],
    }


def batch_inference(
    model,
    sequences: List[np.ndarray],
    batch_size: int = 32,
    threshold: float = 0.5,
    apply_refractory: bool = True,
    device: str = "cpu",
) -> List[Dict[str, np.ndarray]]:
    """
    Perform batch inference on multiple sequences.
    
    Args:
        model: Trained LSTMEventClassifier
        sequences: List of input sequences, each of shape (seq_len, input_size)
        batch_size: Batch size for inference
        threshold: Probability threshold for detection
        apply_refractory: Whether to enforce refractory period
        device: Device to run inference on
        
    Returns:
        List of detection results for each sequence
    """
    model.eval()
    model.to(device)
    
    results = []
    
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i + batch_size]
        
        # Pad sequences to same length
        max_len = max(seq.shape[0] for seq in batch_sequences)
        padded_batch = []
        
        for seq in batch_sequences:
            if seq.shape[0] < max_len:
                padding = np.zeros((max_len - seq.shape[0], seq.shape[1]))
                padded_seq = np.vstack([seq, padding])
            else:
                padded_seq = seq
            padded_batch.append(padded_seq)
        
        # Convert to tensor
        x = torch.FloatTensor(np.stack(padded_batch)).to(device)
        
        # Get predictions
        with torch.no_grad():
            predictions = model.predict_events(x, threshold, apply_refractory)
        
        # Convert to list of dicts
        for j in range(len(batch_sequences)):
            results.append({
                "detected_events": predictions["detected_events"][j].cpu().numpy(),
                "event_timings": predictions["event_timings"][j].cpu().numpy(),
                "event_probs": predictions["event_probs"][j].cpu().numpy(),
            })
    
    return results
