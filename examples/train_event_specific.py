"""
Example training script for event-specific classifier.

Demonstrates usage of EventSpecificClassifier with synthetic data
that has event-specific patterns and feature delays.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

from lstm_classifier.models.event_specific import EventSpecificClassifier
from lstm_classifier.training.losses import EventSpecificLoss
from lstm_classifier.utils.visualization import visualize_learned_templates


def generate_event_specific_data(
    num_samples: int = 1000,
    seq_len: int = 200,
    num_features: int = 6,
    num_events: int = 16,
    noise_level: float = 0.1,
):
    """
    Generate synthetic data with event-specific patterns.
    
    Each event type has:
    - Distinct feature shapes
    - Feature-specific onset delays (0-5 samples)
    - Exponential decay after onset
    
    Args:
        num_samples: Number of sequences to generate
        seq_len: Length of each sequence
        num_features: Number of features
        num_events: Number of event types
        noise_level: Noise standard deviation
        
    Returns:
        Tuple of (data, event_labels, event_timings)
    """
    data = np.zeros((num_samples, seq_len, num_features), dtype=np.float32)
    event_labels = np.zeros((num_samples, num_events), dtype=np.float32)
    event_timings = np.zeros((num_samples, num_events), dtype=np.float32)
    
    # Create base event shapes (different per event type)
    event_shapes = []
    for event_idx in range(num_events):
        # Random shape parameters for each event
        shape = np.zeros((num_features, 80))  # 40ms duration
        for feature_idx in range(num_features):
            # Different patterns per feature per event
            amplitude = np.random.uniform(0.5, 2.0)
            frequency = np.random.uniform(0.5, 3.0)
            phase = np.random.uniform(0, 2 * np.pi)
            
            t = np.arange(80)
            # Sine wave with exponential decay
            shape[feature_idx] = amplitude * np.sin(frequency * t / 10 + phase) * np.exp(-t / 20)
        
        event_shapes.append(shape)
    
    # Generate sequences
    for sample_idx in range(num_samples):
        # Add background noise
        data[sample_idx] = np.random.randn(seq_len, num_features) * noise_level
        
        # Randomly select 1-3 events to occur
        num_events_in_sample = np.random.randint(1, 4)
        selected_events = np.random.choice(num_events, size=num_events_in_sample, replace=False)
        
        for event_idx in selected_events:
            # Random timing (with some margin for event duration)
            timing = np.random.randint(20, seq_len - 100)
            
            # Record labels
            event_labels[sample_idx, event_idx] = 1.0
            event_timings[sample_idx, event_idx] = float(timing)
            
            # Get event shape
            shape = event_shapes[event_idx]
            
            # Apply feature-specific delays (0-5 samples)
            feature_delays = np.random.randint(0, 6, size=num_features)
            
            # Embed event in sequence
            for feature_idx in range(num_features):
                delay = feature_delays[feature_idx]
                start = timing + delay
                end = start + 80
                
                if end <= seq_len:
                    data[sample_idx, start:end, feature_idx] += shape[feature_idx]
    
    return data, event_labels, event_timings


def train_event_specific_model():
    """Main training function."""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate data
    print("Generating synthetic data...")
    train_data, train_labels, train_timings = generate_event_specific_data(
        num_samples=800, seq_len=200
    )
    val_data, val_labels, val_timings = generate_event_specific_data(
        num_samples=200, seq_len=200
    )
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.from_numpy(train_data),
        torch.from_numpy(train_labels),
        torch.from_numpy(train_timings),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_data),
        torch.from_numpy(val_labels),
        torch.from_numpy(val_timings),
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    print("Creating model...")
    model = EventSpecificClassifier(
        input_size=6,
        hidden_size=128,
        num_layers=2,
        num_events=16,
        dropout=0.3,
        template_length=80,
        bidirectional=True,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = EventSpecificLoss(
        event_weight=1.0,
        timing_weight=1.0,
        template_weight=0.5,
        refractory_weight=0.1,
    )
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Create output directory
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    num_epochs = 100
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_event_loss = 0.0
        train_timing_loss = 0.0
        train_template_loss = 0.0
        
        for batch_idx, (x, labels, timings) in enumerate(train_loader):
            x = x.to(device)
            labels = labels.to(device)
            timings = timings.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            event_logits, timing_logits, template_scores = model(x)
            
            # Compute loss
            loss_dict = criterion(
                event_logits, timing_logits, template_scores,
                labels, timings
            )
            
            loss = loss_dict['total']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_event_loss += loss_dict['event'].item()
            train_timing_loss += loss_dict['timing'].item()
            train_template_loss += loss_dict['template'].item()
        
        train_loss /= len(train_loader)
        train_event_loss /= len(train_loader)
        train_timing_loss /= len(train_loader)
        train_template_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_event_loss = 0.0
        val_timing_loss = 0.0
        val_template_loss = 0.0
        
        with torch.no_grad():
            for x, labels, timings in val_loader:
                x = x.to(device)
                labels = labels.to(device)
                timings = timings.to(device)
                
                event_logits, timing_logits, template_scores = model(x)
                
                loss_dict = criterion(
                    event_logits, timing_logits, template_scores,
                    labels, timings
                )
                
                val_loss += loss_dict['total'].item()
                val_event_loss += loss_dict['event'].item()
                val_timing_loss += loss_dict['timing'].item()
                val_template_loss += loss_dict['template'].item()
        
        val_loss /= len(val_loader)
        val_event_loss /= len(val_loader)
        val_timing_loss /= len(val_loader)
        val_template_loss /= len(val_loader)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train - Total: {train_loss:.4f}, Event: {train_event_loss:.4f}, "
              f"Timing: {train_timing_loss:.4f}, Template: {train_template_loss:.4f}")
        print(f"  Val   - Total: {val_loss:.4f}, Event: {val_event_loss:.4f}, "
              f"Timing: {val_timing_loss:.4f}, Template: {val_template_loss:.4f}")
        
        # Learning rate scheduler
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'checkpoints/best_event_specific.pth')
            print(f"  Saved best model (val_loss: {val_loss:.4f})")
        
        # Visualize templates every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            save_path = f'visualizations/templates_epoch_{epoch+1}.png'
            visualize_learned_templates(model, save_path)
            model.train()
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Final visualization
    model.eval()
    checkpoint = torch.load('checkpoints/best_event_specific.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    visualize_learned_templates(model, 'visualizations/templates_final.png')


if __name__ == '__main__':
    train_event_specific_model()
