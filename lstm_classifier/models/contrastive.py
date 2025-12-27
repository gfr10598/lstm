"""
Contrastive LSTM for self-supervised pretraining.

This module implements a contrastive learning approach for pretraining
the LSTM backbone using temporal augmentations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class ContrastiveLSTM(nn.Module):
    """
    LSTM with contrastive learning for self-supervised pretraining.
    
    Uses temporal augmentations to create positive pairs and learns
    representations that are invariant to these augmentations.
    
    Args:
        input_size: Number of input features per timestep (default: 6)
        hidden_size: Number of LSTM hidden units (default: 128)
        num_layers: Number of LSTM layers (default: 2)
        projection_size: Size of projection head output (default: 64)
        dropout: Dropout probability (default: 0.3)
        bidirectional: Whether to use bidirectional LSTM (default: True)
        temperature: Temperature parameter for contrastive loss (default: 0.07)
    """
    
    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 128,
        num_layers: int = 2,
        projection_size: int = 64,
        dropout: float = 0.3,
        bidirectional: bool = True,
        temperature: float = 0.07,
    ):
        super(ContrastiveLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.projection_size = projection_size
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.temperature = temperature
        
        # LSTM backbone
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        
        encoder_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_output_size, encoder_output_size),
            nn.ReLU(),
            nn.Linear(encoder_output_size, projection_size),
        )
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequence.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Encoded representation (batch_size, hidden_size)
        """
        # Encode
        encoder_out, (h_n, c_n) = self.encoder(x)
        
        # Use mean pooling over sequence
        pooled = torch.mean(encoder_out, dim=1)
        
        return pooled
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the contrastive model.
        
        Args:
            x1: First view of input (batch_size, seq_len, input_size)
            x2: Second view of input (batch_size, seq_len, input_size), optional
            
        Returns:
            Dictionary containing:
                - z1: Projection of first view (batch_size, projection_size)
                - z2: Projection of second view if provided (batch_size, projection_size)
                - h1: Encoding of first view (batch_size, hidden_size)
                - h2: Encoding of second view if provided (batch_size, hidden_size)
        """
        # Encode first view
        h1 = self.encode(x1)
        z1 = self.projection_head(h1)
        z1 = F.normalize(z1, dim=1)
        
        result = {
            "z1": z1,
            "h1": h1,
        }
        
        # Encode second view if provided
        if x2 is not None:
            h2 = self.encode(x2)
            z2 = self.projection_head(h2)
            z2 = F.normalize(z2, dim=1)
            result["z2"] = z2
            result["h2"] = h2
        
        return result
    
    def contrastive_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute NT-Xent (normalized temperature-scaled cross entropy) loss.
        
        Args:
            z1: Projections of first view (batch_size, projection_size)
            z2: Projections of second view (batch_size, projection_size)
            
        Returns:
            Contrastive loss scalar
        """
        batch_size = z1.shape[0]
        
        # Concatenate z1 and z2
        z = torch.cat([z1, z2], dim=0)  # (2 * batch_size, projection_size)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t())  # (2 * batch_size, 2 * batch_size)
        
        # Mask to remove self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
        
        # Scale by temperature
        sim_matrix = sim_matrix / self.temperature
        
        # Create labels: each sample i in first half is positive pair with i+batch_size
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(batch_size, device=z.device),
        ])
        
        # Compute cross entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
    
    def get_encoder_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get encoder weights for transfer to main model.
        
        Returns:
            Dictionary of encoder state dict
        """
        return self.encoder.state_dict()
    
    def transfer_encoder_to_classifier(self, classifier_model):
        """
        Transfer learned encoder weights to the main classifier model.
        
        Args:
            classifier_model: LSTMEventClassifier instance to transfer weights to
        """
        # Transfer LSTM weights
        encoder_state = self.encoder.state_dict()
        classifier_state = classifier_model.lstm.state_dict()
        
        # Map encoder weights to classifier
        for key in encoder_state:
            if key in classifier_state:
                # Check if shapes match
                if encoder_state[key].shape == classifier_state[key].shape:
                    classifier_state[key] = encoder_state[key]
        
        classifier_model.lstm.load_state_dict(classifier_state, strict=False)
        
        return classifier_model


class TemporalAugmentation:
    """
    Temporal augmentations for creating positive pairs in contrastive learning.
    """
    
    @staticmethod
    def add_noise(x: torch.Tensor, noise_std: float = 0.01) -> torch.Tensor:
        """Add Gaussian noise to the input."""
        noise = torch.randn_like(x) * noise_std
        return x + noise
    
    @staticmethod
    def scale(x: torch.Tensor, scale_range: Tuple[float, float] = (0.8, 1.2)) -> torch.Tensor:
        """Scale the input by a random factor."""
        scale_factor = torch.FloatTensor(1).uniform_(*scale_range).to(x.device)
        return x * scale_factor
    
    @staticmethod
    def temporal_crop(x: torch.Tensor, crop_ratio: float = 0.9) -> torch.Tensor:
        """Randomly crop the temporal dimension."""
        seq_len = x.shape[1]
        crop_len = int(seq_len * crop_ratio)
        start_idx = torch.randint(0, seq_len - crop_len + 1, (1,)).item()
        
        # Crop and pad back to original length
        cropped = x[:, start_idx:start_idx + crop_len, :]
        pad_len = seq_len - crop_len
        if pad_len > 0:
            padding = torch.zeros(x.shape[0], pad_len, x.shape[2], device=x.device)
            cropped = torch.cat([cropped, padding], dim=1)
        
        return cropped
    
    @staticmethod
    def temporal_shift(x: torch.Tensor, max_shift: int = 10) -> torch.Tensor:
        """Shift the sequence in time."""
        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        
        if shift == 0:
            return x
        elif shift > 0:
            # Shift right
            padding = torch.zeros(x.shape[0], shift, x.shape[2], device=x.device)
            return torch.cat([padding, x[:, :-shift, :]], dim=1)
        else:
            # Shift left
            padding = torch.zeros(x.shape[0], -shift, x.shape[2], device=x.device)
            return torch.cat([x[:, -shift:, :], padding], dim=1)
    
    @staticmethod
    def random_augment(x: torch.Tensor) -> torch.Tensor:
        """Apply random combination of augmentations."""
        aug_funcs = [
            lambda x: TemporalAugmentation.add_noise(x, 0.01),
            lambda x: TemporalAugmentation.scale(x, (0.9, 1.1)),
            lambda x: TemporalAugmentation.temporal_crop(x, 0.95),
            lambda x: TemporalAugmentation.temporal_shift(x, 5),
        ]
        
        # Randomly select and apply 1-2 augmentations
        num_augs = torch.randint(1, 3, (1,)).item()
        selected_augs = torch.randperm(len(aug_funcs))[:num_augs]
        
        x_aug = x.clone()
        for idx in selected_augs:
            x_aug = aug_funcs[idx](x_aug)
        
        return x_aug
