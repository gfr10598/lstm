"""
LSTM Autoencoder for unsupervised pretraining.

This module implements an autoencoder architecture that can be used to
pretrain the LSTM backbone before fine-tuning on labeled data.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class LSTMAutoencoder(nn.Module):
    """
    LSTM-based autoencoder for unsupervised pretraining.
    
    The autoencoder learns to reconstruct input sequences, which helps
    the LSTM learn useful temporal representations that can be transferred
    to the supervised task.
    
    Args:
        input_size: Number of input features per timestep (default: 6)
        hidden_size: Number of LSTM hidden units (default: 128)
        num_layers: Number of LSTM layers (default: 2)
        latent_size: Size of the latent representation (default: 64)
        dropout: Dropout probability (default: 0.3)
        bidirectional: Whether to use bidirectional LSTM in encoder (default: True)
    """
    
    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 128,
        num_layers: int = 2,
        latent_size: int = 64,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super(LSTMAutoencoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.latent_size = latent_size
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        
        encoder_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Latent space projection
        self.encoder_fc = nn.Linear(encoder_output_size, latent_size)
        
        # Decoder projection
        self.decoder_fc = nn.Linear(latent_size, hidden_size)
        
        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        
        # Output projection
        self.output_layer = nn.Linear(hidden_size, input_size)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequence to latent representation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Latent representation (batch_size, latent_size)
        """
        # Encode
        encoder_out, (h_n, c_n) = self.encoder(x)
        
        # Use mean pooling over sequence
        pooled = torch.mean(encoder_out, dim=1)
        
        # Project to latent space
        latent = self.encoder_fc(pooled)
        
        return latent
    
    def decode(self, latent: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Decode latent representation to output sequence.
        
        Args:
            latent: Latent tensor of shape (batch_size, latent_size)
            seq_len: Length of output sequence
            
        Returns:
            Reconstructed sequence (batch_size, seq_len, input_size)
        """
        batch_size = latent.shape[0]
        
        # Project latent to decoder input size
        decoder_input = self.decoder_fc(latent)  # (batch_size, hidden_size)
        
        # Repeat for each timestep
        decoder_input = decoder_input.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Decode
        decoder_out, _ = self.decoder(decoder_input)
        
        # Project to output space
        reconstruction = self.output_layer(decoder_out)
        
        return reconstruction
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Dictionary containing:
                - reconstruction: Reconstructed input (batch_size, seq_len, input_size)
                - latent: Latent representation (batch_size, latent_size)
        """
        seq_len = x.shape[1]
        
        # Encode
        latent = self.encode(x)
        
        # Decode
        reconstruction = self.decode(latent, seq_len)
        
        return {
            "reconstruction": reconstruction,
            "latent": latent,
        }
    
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
        # Handle bidirectional encoder to bidirectional/unidirectional classifier
        for key in encoder_state:
            if key in classifier_state:
                # Check if shapes match
                if encoder_state[key].shape == classifier_state[key].shape:
                    classifier_state[key] = encoder_state[key]
        
        classifier_model.lstm.load_state_dict(classifier_state, strict=False)
        
        return classifier_model


class VariationalLSTMAutoencoder(LSTMAutoencoder):
    """
    Variational LSTM Autoencoder for unsupervised pretraining.
    
    Extends the basic autoencoder with a variational bottleneck to learn
    a more robust latent representation.
    
    Args:
        Same as LSTMAutoencoder
    """
    
    def __init__(self, *args, **kwargs):
        super(VariationalLSTMAutoencoder, self).__init__(*args, **kwargs)
        
        encoder_output_size = (
            self.hidden_size * 2 if self.bidirectional else self.hidden_size
        )
        
        # Replace encoder_fc with mu and logvar projections
        self.mu_fc = nn.Linear(encoder_output_size, self.latent_size)
        self.logvar_fc = nn.Linear(encoder_output_size, self.latent_size)
        
        # Remove the original encoder_fc
        del self.encoder_fc
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        
        Args:
            mu: Mean of latent distribution (batch_size, latent_size)
            logvar: Log variance of latent distribution (batch_size, latent_size)
            
        Returns:
            Sampled latent vector (batch_size, latent_size)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x: torch.Tensor) -> tuple:
        """
        Encode input sequence to latent distribution parameters.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Tuple of (mu, logvar, z) where z is the sampled latent vector
        """
        # Encode
        encoder_out, (h_n, c_n) = self.encoder(x)
        
        # Use mean pooling over sequence
        pooled = torch.mean(encoder_out, dim=1)
        
        # Get distribution parameters
        mu = self.mu_fc(pooled)
        logvar = self.logvar_fc(pooled)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, logvar)
        
        return mu, logvar, z
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the variational autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Dictionary containing:
                - reconstruction: Reconstructed input (batch_size, seq_len, input_size)
                - mu: Mean of latent distribution (batch_size, latent_size)
                - logvar: Log variance of latent distribution (batch_size, latent_size)
                - latent: Sampled latent vector (batch_size, latent_size)
        """
        seq_len = x.shape[1]
        
        # Encode
        mu, logvar, latent = self.encode(x)
        
        # Decode
        reconstruction = self.decode(latent, seq_len)
        
        return {
            "reconstruction": reconstruction,
            "mu": mu,
            "logvar": logvar,
            "latent": latent,
        }
