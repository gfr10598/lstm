"""
Visualization utilities for event-specific models.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

# Import constant from model module
from lstm_classifier.models.event_specific import MAX_ONSET_DELAY_SAMPLES


def visualize_learned_templates(model, save_path: str = 'templates.png'):
    """
    Visualize learned event templates with feature-specific delays.
    
    Args:
        model: EventSpecificClassifier instance (must have template_bank attribute)
        save_path: Path to save the figure
    """
    # Extract templates and delays from model
    if not hasattr(model, 'template_bank'):
        raise ValueError("Model must have a template_bank attribute")
    
    template_bank = model.template_bank
    templates = template_bank.templates.detach().cpu()  # (num_events, num_features, template_length)
    delays = template_bank.onset_delays.detach().cpu()  # (num_events, num_features)
    
    num_events = templates.shape[0]
    num_features = templates.shape[1]
    template_length = templates.shape[2]
    
    # Create 4x4 subplot grid
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    # Plot each event
    for event_idx in range(min(num_events, 16)):
        ax = axes[event_idx]
        
        # Apply delays to templates for visualization
        delayed_templates = []
        for feature_idx in range(num_features):
            template = templates[event_idx, feature_idx].numpy()
            delay = delays[event_idx, feature_idx].item()
            
            # Clamp delay using constant from model
            delay_clamped = np.clip(delay, -MAX_ONSET_DELAY_SAMPLES, MAX_ONSET_DELAY_SAMPLES)
            delay_int = int(np.round(delay_clamped))
            
            # Apply delay using roll
            delayed = np.roll(template, shift=delay_int)
            
            # Zero out wrapped-around values
            if delay_int > 0:
                delayed[:delay_int] = 0
            elif delay_int < 0:
                delayed[delay_int:] = 0
            
            delayed_templates.append(delayed)
        
        # Plot all features
        time_axis = np.arange(template_length)
        for feature_idx, delayed_template in enumerate(delayed_templates):
            ax.plot(time_axis, delayed_template, label=f'F{feature_idx}', alpha=0.7)
        
        ax.set_title(f'Event {event_idx}', fontsize=10)
        ax.set_xlabel('Samples', fontsize=8)
        ax.set_ylabel('Amplitude', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Only add legend to first subplot
        if event_idx == 0:
            ax.legend(fontsize=6, loc='upper right')
        
        ax.tick_params(labelsize=7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Template visualization saved to {save_path}")
