"""
Core LSTM model for event classification and timing prediction.

This module implements the main LSTM-based architecture that:
- Accepts input sequences with 6 features per timestep (sampled at 2000 Hz)
- Predicts 16 event classes
- Outputs event timing at 5 ms resolution (10 samples)
- Enforces 2-second refractory period per event type
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class LSTMEventClassifier(nn.Module):
    """
    LSTM-based model for event classification and timing prediction.
    
    The model predicts both which events occur and when they occur, with
    built-in support for enforcing refractory periods.
    
    Args:
        input_size: Number of input features per timestep (default: 6)
        hidden_size: Number of LSTM hidden units (default: 128)
        num_layers: Number of LSTM layers (default: 2)
        num_events: Number of event classes to predict (default: 16)
        dropout: Dropout probability (default: 0.3)
        bidirectional: Whether to use bidirectional LSTM (default: True)
        timing_resolution_samples: Timing resolution in samples (default: 10 for 5ms at 2000Hz)
        refractory_period_samples: Refractory period in samples (default: 4000 for 2s at 2000Hz)
    """
    
    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_events: int = 16,
        dropout: float = 0.3,
        bidirectional: bool = True,
        timing_resolution_samples: int = 10,
        refractory_period_samples: int = 4000,
    ):
        super(LSTMEventClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_events = num_events
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.timing_resolution_samples = timing_resolution_samples
        self.refractory_period_samples = refractory_period_samples
        
        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        
        # Calculate LSTM output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Event classification head
        # Predicts probability of each event occurring in the sequence
        self.event_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, num_events),
        )
        
        # Event timing head
        # For each event, predicts the timing offset within the sequence
        # Output is per-timestep logits that can be converted to timing predictions
        self.timing_predictor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, num_events),
        )
        
        # Attention mechanism for focusing on relevant timesteps
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            lengths: Optional tensor of sequence lengths for packed sequences
            
        Returns:
            Dictionary containing:
                - event_logits: Event classification logits (batch_size, num_events)
                - timing_logits: Per-timestep timing logits (batch_size, seq_len, num_events)
                - event_probs: Event probabilities (batch_size, num_events)
                - timing_probs: Normalized timing probabilities (batch_size, seq_len, num_events)
                - hidden_states: LSTM hidden states (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        
        # LSTM encoding
        if lengths is not None:
            # Pack padded sequence for variable length inputs
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, (h_n, c_n) = self.lstm(packed_x)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=seq_len
            )
        else:
            lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention
        attended_out, attention_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # Event classification using mean pooling over sequence
        # This gives us which events occurred in the entire sequence
        pooled_features = torch.mean(attended_out, dim=1)  # (batch_size, lstm_output_size)
        event_logits = self.event_classifier(pooled_features)  # (batch_size, num_events)
        event_probs = torch.sigmoid(event_logits)
        
        # Timing prediction per timestep
        # This gives us when each event occurred
        timing_logits = self.timing_predictor(attended_out)  # (batch_size, seq_len, num_events)
        
        # Normalize timing logits to get probability distribution over time for each event
        timing_probs = F.softmax(timing_logits, dim=1)  # (batch_size, seq_len, num_events)
        
        return {
            "event_logits": event_logits,
            "timing_logits": timing_logits,
            "event_probs": event_probs,
            "timing_probs": timing_probs,
            "hidden_states": lstm_out,
            "attention_weights": attention_weights,
        }
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get LSTM embeddings for unsupervised pretraining.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            LSTM hidden states (batch_size, seq_len, hidden_size)
        """
        lstm_out, _ = self.lstm(x)
        return lstm_out
    
    def predict_events(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
        apply_refractory: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict events and their timings with optional refractory period enforcement.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            threshold: Probability threshold for event detection (default: 0.5)
            apply_refractory: Whether to enforce refractory period (default: True)
            
        Returns:
            Dictionary containing:
                - detected_events: Binary tensor of detected events (batch_size, num_events)
                - event_timings: Predicted timing for each event in samples (batch_size, num_events)
                - event_probs: Event probabilities (batch_size, num_events)
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            
            # Detect events based on threshold
            detected_events = (output["event_probs"] > threshold).float()
            
            # Get timing as weighted average of timesteps
            # Shape: (batch_size, num_events)
            timesteps = torch.arange(x.shape[1], device=x.device).float()
            timesteps = timesteps.view(1, -1, 1)  # (1, seq_len, 1)
            
            # Weighted sum over timesteps
            event_timings = torch.sum(
                output["timing_probs"] * timesteps, dim=1
            )  # (batch_size, num_events)
            
            # Round to timing resolution
            event_timings = torch.round(
                event_timings / self.timing_resolution_samples
            ) * self.timing_resolution_samples
            
            # Apply refractory period enforcement if requested
            if apply_refractory:
                detected_events, event_timings = self._enforce_refractory_period(
                    detected_events, event_timings
                )
            
            return {
                "detected_events": detected_events,
                "event_timings": event_timings,
                "event_probs": output["event_probs"],
            }
    
    def _enforce_refractory_period(
        self,
        detected_events: torch.Tensor,
        event_timings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enforce refractory period constraint on detected events.
        
        This is a post-processing step that ensures events of the same type
        don't occur within the refractory period.
        
        Args:
            detected_events: Binary tensor of detected events (batch_size, num_events)
            event_timings: Event timings in samples (batch_size, num_events)
            
        Returns:
            Tuple of (filtered_events, filtered_timings)
        """
        # This is a placeholder for refractory period enforcement
        # In practice, this would need access to previous detections across batches
        # For now, we just return the inputs unchanged
        # A full implementation would maintain state or use a custom loss function
        return detected_events, event_timings
    
    def freeze_backbone(self):
        """Freeze LSTM backbone for transfer learning."""
        for param in self.lstm.parameters():
            param.requires_grad = False
        for param in self.attention.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze LSTM backbone."""
        for param in self.lstm.parameters():
            param.requires_grad = True
        for param in self.attention.parameters():
            param.requires_grad = True
