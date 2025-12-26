"""
Example script for inference on new data.

This script demonstrates how to use a trained model for event detection.
"""

import torch
import numpy as np

from lstm_classifier.models import LSTMEventClassifier
from lstm_classifier.utils import detect_events_from_sequence, RefractoryPeriodEnforcer


def main():
    # Configuration
    config = {
        "input_size": 6,
        "hidden_size": 128,
        "num_layers": 2,
        "num_events": 16,
        "dropout": 0.3,
        "bidirectional": True,
        "checkpoint_path": "./checkpoints/supervised/best_model.pth",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    
    print("Loading trained model...")
    # Create model
    model = LSTMEventClassifier(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_events=config["num_events"],
        dropout=config["dropout"],
        bidirectional=config["bidirectional"],
    )
    
    # Load checkpoint
    checkpoint = torch.load(config["checkpoint_path"], map_location=config["device"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config["device"])
    model.eval()
    
    print("Model loaded successfully!")
    
    # Generate example input sequence (in practice, this would be real sensor data)
    print("\nGenerating example input sequence...")
    seq_len = 2000  # 1 second at 2000 Hz
    input_sequence = np.random.randn(seq_len, config["input_size"]).astype(np.float32)
    
    # Detect events
    print("Detecting events...")
    results = detect_events_from_sequence(
        model=model,
        sequence=input_sequence,
        threshold=0.5,
        apply_refractory=True,
        device=config["device"],
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("Event Detection Results")
    print("=" * 60)
    
    detected_indices = np.where(results["detected_events"] > 0)[0]
    
    if len(detected_indices) > 0:
        print(f"\nDetected {len(detected_indices)} events:\n")
        for event_idx in detected_indices:
            timing_ms = results["event_timings"][event_idx] * 0.5  # Convert samples to ms
            probability = results["event_probs"][event_idx]
            print(f"  Event {event_idx:2d}: Time = {timing_ms:6.1f} ms, Probability = {probability:.3f}")
    else:
        print("\nNo events detected.")
    
    print("\n" + "=" * 60)
    
    # Demonstrate refractory period enforcement across multiple sequences
    print("\n\nDemonstrating refractory period enforcement...")
    
    enforcer = RefractoryPeriodEnforcer(
        num_events=config["num_events"],
        refractory_period_samples=4000,  # 2 seconds at 2000 Hz
    )
    
    # Simulate processing multiple sequences
    num_sequences = 3
    for seq_idx in range(num_sequences):
        # Generate sequence
        sequence = np.random.randn(seq_len, config["input_size"]).astype(np.float32)
        
        # Detect events
        results = detect_events_from_sequence(
            model=model,
            sequence=sequence,
            threshold=0.5,
            apply_refractory=False,  # We'll apply it manually
            device=config["device"],
        )
        
        # Apply refractory period enforcement
        filtered_events, filtered_timings = enforcer.enforce(
            detected_events=results["detected_events"],
            event_timings=results["event_timings"],
            event_probs=results["event_probs"],
            global_offset=seq_idx * seq_len,
        )
        
        detected_indices = np.where(filtered_events > 0)[0]
        print(f"\nSequence {seq_idx + 1}: Detected {len(detected_indices)} events (after refractory filtering)")
        for event_idx in detected_indices:
            timing_ms = filtered_timings[event_idx] * 0.5
            print(f"  Event {event_idx:2d}: Time = {timing_ms:6.1f} ms")


if __name__ == "__main__":
    main()
