"""
Event-specific neural architecture components.

This module implements event-specific components that incorporate domain knowledge
about events having distinct feature shapes and cross-feature onset skews.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


# Constants
MAX_ONSET_DELAY_SAMPLES = 10  # Maximum onset delay in samples (±5ms @ 2000Hz)


class EventTemplateBank(nn.Module):
    """
    Learnable template bank for event-specific pattern matching.
    
    Each event type has:
    - A learnable template capturing the event's feature patterns
    - Feature-specific onset delays representing cross-feature skews
    
    Note: The onset delays use discrete operations (torch.roll) for applying
    temporal shifts, which means gradients do not flow through the delay
    parameters. This is a design trade-off for computational efficiency.
    For applications requiring differentiable delays, consider using
    spatial transformer networks or sinc interpolation.
    
    Args:
        num_events: Number of event types (default: 16)
        num_features: Number of input features (default: 6)
        template_length: Length of templates in samples (default: 80 for ~40ms @ 2000Hz)
    """
    
    def __init__(
        self,
        num_events: int = 16,
        num_features: int = 6,
        template_length: int = 80,
    ):
        super(EventTemplateBank, self).__init__()
        
        self.num_events = num_events
        self.num_features = num_features
        self.template_length = template_length
        
        # Learnable templates: (num_events, num_features, template_length)
        self.templates = nn.Parameter(
            torch.randn(num_events, num_features, template_length) * 0.1
        )
        
        # Feature-specific onset delays: (num_events, num_features)
        # Initialize to zeros
        self.onset_delays = nn.Parameter(
            torch.zeros(num_events, num_features)
        )
        
    def apply_delays(
        self,
        templates: torch.Tensor,
        delays: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply feature-specific delays to templates.
        
        Note: This uses discrete operations (torch.roll) which are not differentiable.
        Gradients will not flow through the delay parameters.
        
        Args:
            templates: Templates of shape (num_events, num_features, template_length)
            delays: Delays of shape (num_events, num_features)
            
        Returns:
            Delayed templates of shape (num_events, num_features, template_length)
        """
        # Clamp delays to ±MAX_ONSET_DELAY_SAMPLES
        clamped_delays = torch.clamp(delays, -MAX_ONSET_DELAY_SAMPLES, MAX_ONSET_DELAY_SAMPLES)
        
        batch_size = templates.shape[0]
        delayed_templates = []
        
        for event_idx in range(batch_size):
            event_templates = []
            for feature_idx in range(self.num_features):
                template = templates[event_idx, feature_idx]
                delay = clamped_delays[event_idx, feature_idx]
                
                # Apply delay using roll
                # Positive delay shifts right (later onset)
                # Negative delay shifts left (earlier onset)
                delay_int = int(delay.round().item())
                delayed = torch.roll(template, shifts=delay_int, dims=0)
                
                # Zero out wrapped-around values
                if delay_int > 0:
                    delayed[:delay_int] = 0
                elif delay_int < 0:
                    delayed[delay_int:] = 0
                    
                event_templates.append(delayed)
            
            delayed_templates.append(torch.stack(event_templates))
        
        return torch.stack(delayed_templates)
    
    def match_score(
        self,
        x: torch.Tensor,
        event_idx: int,
    ) -> torch.Tensor:
        """
        Compute template match scores for a specific event.
        
        Args:
            x: Input sequence of shape (batch, seq_len, num_features)
            event_idx: Index of event to match
            
        Returns:
            Match scores of shape (batch, seq_len)
        """
        batch_size, seq_len, num_features = x.shape
        
        # Get template and delays for this event
        template = self.templates[event_idx:event_idx+1]  # (1, num_features, template_length)
        delay = self.onset_delays[event_idx:event_idx+1]  # (1, num_features)
        
        # Apply delays
        delayed_template = self.apply_delays(template, delay)  # (1, num_features, template_length)
        delayed_template = delayed_template.squeeze(0)  # (num_features, template_length)
        
        # Prepare input for conv1d: (batch, num_features, seq_len)
        x_transposed = x.transpose(1, 2)
        
        # For each feature, convolve with template and sum
        match_scores = []
        for feature_idx in range(num_features):
            feature_signal = x_transposed[:, feature_idx:feature_idx+1, :]  # (batch, 1, seq_len)
            feature_template = delayed_template[feature_idx:feature_idx+1, :]  # (1, template_length)
            feature_template = feature_template.unsqueeze(0)  # (1, 1, template_length)
            
            # Cross-correlation using conv1d
            # Flip template for correlation (conv is correlation with flipped kernel)
            feature_template_flipped = torch.flip(feature_template, dims=[2])
            
            # Convolve - use 'same' padding to maintain sequence length
            padding = (self.template_length - 1) // 2
            score = F.conv1d(
                feature_signal,
                feature_template_flipped,
                padding=padding
            )  # (batch, 1, seq_len_out)
            
            # Trim to exact seq_len if needed
            if score.shape[2] > seq_len:
                score = score[:, :, :seq_len]
            elif score.shape[2] < seq_len:
                # Pad if needed
                padding_needed = seq_len - score.shape[2]
                score = F.pad(score, (0, padding_needed))
            
            match_scores.append(score.squeeze(1))  # (batch, seq_len)
        
        # Sum scores across features
        total_score = torch.stack(match_scores, dim=0).sum(dim=0)  # (batch, seq_len)
        
        # Normalize by template length and number of features
        total_score = total_score / (self.template_length * num_features)
        
        return total_score
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute template match scores for all events.
        
        Args:
            x: Input sequence of shape (batch, seq_len, num_features)
            
        Returns:
            Match scores of shape (batch, seq_len, num_events)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute match scores for each event
        all_scores = []
        for event_idx in range(self.num_events):
            scores = self.match_score(x, event_idx)  # (batch, seq_len)
            all_scores.append(scores)
        
        # Stack to get (batch, seq_len, num_events)
        match_scores = torch.stack(all_scores, dim=2)
        
        return match_scores


class EventSpecificCNN(nn.Module):
    """
    Event-specific convolutional pathways.
    
    Each event type has its own convolutional stack to learn
    event-specific temporal patterns at multiple scales.
    
    Args:
        num_events: Number of event types (default: 16)
        input_channels: Number of input features (default: 6)
        output_channels: Number of output channels per conv layer (default: 32)
        use_shared: Whether to include shared pathway (default: True)
    """
    
    def __init__(
        self,
        num_events: int = 16,
        input_channels: int = 6,
        output_channels: int = 32,
        use_shared: bool = True,
    ):
        super(EventSpecificCNN, self).__init__()
        
        self.num_events = num_events
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.use_shared = use_shared
        
        # Per-event convolutional pathways
        self.event_convs = nn.ModuleList()
        for _ in range(num_events):
            # 3 conv layers with different kernel sizes
            conv_stack = nn.Sequential(
                # Layer 1: kernel_size=11 (5ms) for onset detection
                nn.Conv1d(input_channels, output_channels, kernel_size=11, padding=5),
                nn.BatchNorm1d(output_channels),
                nn.ReLU(),
                
                # Layer 2: kernel_size=21 (10ms) for shape detection
                nn.Conv1d(output_channels, output_channels, kernel_size=21, padding=10),
                nn.BatchNorm1d(output_channels),
                nn.ReLU(),
                
                # Layer 3: kernel_size=41 (20ms) for full event pattern
                nn.Conv1d(output_channels, output_channels, kernel_size=41, padding=20),
                nn.BatchNorm1d(output_channels),
                nn.ReLU(),
            )
            self.event_convs.append(conv_stack)
        
        # Optional shared pathway
        if use_shared:
            self.shared_conv = nn.Sequential(
                nn.Conv1d(input_channels, output_channels, kernel_size=11, padding=5),
                nn.BatchNorm1d(output_channels),
                nn.ReLU(),
            )
        else:
            self.shared_conv = None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through event-specific and shared pathways.
        
        Args:
            x: Input of shape (batch, seq_len, input_channels)
            
        Returns:
            Tuple of:
                - event_features: (batch, seq_len, num_events, output_channels)
                - shared_features: (batch, seq_len, output_channels) or None
        """
        batch_size, seq_len, _ = x.shape
        
        # Transpose for conv1d: (batch, input_channels, seq_len)
        x_transposed = x.transpose(1, 2)
        
        # Process through each event-specific pathway
        event_features = []
        for event_idx in range(self.num_events):
            features = self.event_convs[event_idx](x_transposed)  # (batch, output_channels, seq_len)
            features = features.transpose(1, 2)  # (batch, seq_len, output_channels)
            event_features.append(features)
        
        # Stack event features: (batch, seq_len, num_events, output_channels)
        event_features = torch.stack(event_features, dim=2)
        
        # Process through shared pathway if it exists
        shared_features = None
        if self.shared_conv is not None:
            shared_features = self.shared_conv(x_transposed)  # (batch, output_channels, seq_len)
            shared_features = shared_features.transpose(1, 2)  # (batch, seq_len, output_channels)
        
        return event_features, shared_features


class EventSpecificTimingHeads(nn.Module):
    """
    Per-event timing prediction heads with temporal attention.
    
    Each event type has its own timing predictor and attention mechanism
    to focus on relevant temporal patterns.
    
    Args:
        num_events: Number of event types (default: 16)
        hidden_size: Size of LSTM hidden states (default: 128)
        dropout: Dropout probability (default: 0.2)
    """
    
    def __init__(
        self,
        num_events: int = 16,
        hidden_size: int = 128,
        dropout: float = 0.2,
    ):
        super(EventSpecificTimingHeads, self).__init__()
        
        self.num_events = num_events
        self.hidden_size = hidden_size
        
        # Per-event timing predictors
        self.timing_heads = nn.ModuleList()
        for _ in range(num_events):
            head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1),
            )
            self.timing_heads.append(head)
        
        # Per-event temporal attention
        self.event_attentions = nn.ModuleList()
        for _ in range(num_events):
            attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )
            self.event_attentions.append(attention)
    
    def forward(
        self,
        lstm_features: torch.Tensor,
        event_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict timing for each event.
        
        Args:
            lstm_features: LSTM features of shape (batch, seq_len, hidden_size)
            event_mask: Optional mask of shape (batch, num_events) for present events
            
        Returns:
            Timing logits of shape (batch, seq_len, num_events)
        """
        batch_size, seq_len, _ = lstm_features.shape
        
        # Process each event
        timing_logits_list = []
        for event_idx in range(self.num_events):
            # Apply attention for this event
            attended_features, _ = self.event_attentions[event_idx](
                lstm_features, lstm_features, lstm_features
            )  # (batch, seq_len, hidden_size)
            
            # Predict timing
            logits = self.timing_heads[event_idx](attended_features)  # (batch, seq_len, 1)
            logits = logits.squeeze(-1)  # (batch, seq_len)
            
            timing_logits_list.append(logits)
        
        # Stack: (batch, seq_len, num_events)
        timing_logits = torch.stack(timing_logits_list, dim=2)
        
        # Apply event mask if provided
        if event_mask is not None:
            # Expand mask: (batch, 1, num_events)
            mask_expanded = event_mask.unsqueeze(1)
            # Set logits to very negative for absent events
            timing_logits = timing_logits * mask_expanded + (1 - mask_expanded) * (-1e9)
        
        return timing_logits


class EventSpecificClassifier(nn.Module):
    """
    Main event-specific classifier model.
    
    This model combines:
    1. Template matching via EventTemplateBank
    2. Event-specific CNN features
    3. Bidirectional LSTM processing
    4. Event classification and timing prediction
    
    Args:
        input_size: Number of input features (default: 6)
        hidden_size: LSTM hidden size (default: 128)
        num_layers: Number of LSTM layers (default: 2)
        num_events: Number of event types (default: 16)
        dropout: Dropout probability (default: 0.3)
        template_length: Length of event templates in samples (default: 80)
        bidirectional: Use bidirectional LSTM (default: True)
    """
    
    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_events: int = 16,
        dropout: float = 0.3,
        template_length: int = 80,
        bidirectional: bool = True,
    ):
        super(EventSpecificClassifier, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_events = num_events
        self.dropout = dropout
        self.template_length = template_length
        self.bidirectional = bidirectional
        
        # Event template bank
        self.template_bank = EventTemplateBank(
            num_events=num_events,
            num_features=input_size,
            template_length=template_length,
        )
        
        # Event-specific CNN
        self.event_cnn = EventSpecificCNN(
            num_events=num_events,
            input_channels=input_size,
            output_channels=32,
            use_shared=True,
        )
        
        # Calculate input size for LSTM
        # Template scores (num_events) + shared CNN features (32) + original input (6)
        lstm_input_size = num_events + 32 + input_size
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        
        # Calculate LSTM output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Event classification head
        self.event_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, num_events),
        )
        
        # Event-specific timing heads
        self.timing_heads = EventSpecificTimingHeads(
            num_events=num_events,
            hidden_size=lstm_output_size,
            dropout=dropout,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input of shape (batch, seq_len, input_size)
            lengths: Optional sequence lengths for packing
            
        Returns:
            Tuple of:
                - event_logits: (batch, num_events)
                - timing_logits: (batch, seq_len, num_events)
                - template_scores: (batch, seq_len, num_events)
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. Template matching
        template_scores = self.template_bank(x)  # (batch, seq_len, num_events)
        
        # 2. Event-specific CNN
        event_features, shared_features = self.event_cnn(x)
        # event_features: (batch, seq_len, num_events, 32)
        # shared_features: (batch, seq_len, 32)
        
        # 3. Combine features for LSTM
        # We'll use template scores and shared features, ignoring per-event CNN for now
        # to keep dimensionality manageable
        combined_features = torch.cat([
            x,                    # (batch, seq_len, 6)
            template_scores,      # (batch, seq_len, num_events)
            shared_features,      # (batch, seq_len, 32)
        ], dim=2)  # (batch, seq_len, 6 + num_events + 32)
        
        # 4. LSTM encoding
        if lengths is not None:
            packed_x = nn.utils.rnn.pack_padded_sequence(
                combined_features, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed_x)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=seq_len
            )
        else:
            lstm_out, _ = self.lstm(combined_features)
        
        # 5. Event classification (mean pooling)
        pooled_features = torch.mean(lstm_out, dim=1)  # (batch, lstm_output_size)
        event_logits = self.event_classifier(pooled_features)  # (batch, num_events)
        
        # 6. Event-specific timing prediction
        timing_logits = self.timing_heads(lstm_out)  # (batch, seq_len, num_events)
        
        return event_logits, timing_logits, template_scores
