"""
Custom loss functions for event classification and timing prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class EventTimingLoss(nn.Module):
    """
    Combined loss for event classification and timing prediction.
    
    This loss combines:
    1. Binary cross-entropy for event detection
    2. Timing prediction loss (only for detected events)
    3. Optional refractory period constraint
    
    Args:
        event_weight: Weight for event classification loss (default: 1.0)
        timing_weight: Weight for timing prediction loss (default: 1.0)
        refractory_weight: Weight for refractory period constraint (default: 0.1)
        refractory_period_samples: Refractory period in samples (default: 4000)
        timing_resolution_samples: Timing resolution in samples (default: 10)
    """
    
    def __init__(
        self,
        event_weight: float = 1.0,
        timing_weight: float = 1.0,
        refractory_weight: float = 0.1,
        refractory_period_samples: int = 4000,
        timing_resolution_samples: int = 10,
    ):
        super(EventTimingLoss, self).__init__()
        
        self.event_weight = event_weight
        self.timing_weight = timing_weight
        self.refractory_weight = refractory_weight
        self.refractory_period_samples = refractory_period_samples
        self.timing_resolution_samples = timing_resolution_samples
        
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        event_logits: torch.Tensor,
        timing_logits: torch.Tensor,
        event_labels: torch.Tensor,
        event_timings: torch.Tensor,
        apply_refractory: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            event_logits: Event classification logits (batch_size, num_events)
            timing_logits: Timing prediction logits (batch_size, seq_len, num_events)
            event_labels: Ground truth event labels (batch_size, num_events)
            event_timings: Ground truth event timings in samples (batch_size, num_events)
            apply_refractory: Whether to apply refractory period constraint
            
        Returns:
            Dictionary containing total loss and individual loss components
        """
        # Event classification loss
        event_loss = self.bce_loss(event_logits, event_labels)
        
        # Timing prediction loss (only for events that actually occurred)
        timing_loss = self._compute_timing_loss(
            timing_logits, event_timings, event_labels
        )
        
        # Total loss
        total_loss = (
            self.event_weight * event_loss +
            self.timing_weight * timing_loss
        )
        
        # Optional refractory period constraint
        refractory_loss = torch.tensor(0.0, device=event_logits.device)
        if apply_refractory and self.refractory_weight > 0:
            refractory_loss = self._compute_refractory_loss(
                timing_logits, event_labels, event_timings
            )
            total_loss = total_loss + self.refractory_weight * refractory_loss
        
        return {
            "total_loss": total_loss,
            "event_loss": event_loss,
            "timing_loss": timing_loss,
            "refractory_loss": refractory_loss,
        }
    
    def _compute_timing_loss(
        self,
        timing_logits: torch.Tensor,
        event_timings: torch.Tensor,
        event_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute timing prediction loss using cross-entropy.
        
        The timing logits are treated as a probability distribution over timesteps,
        and we compute the cross-entropy with a target distribution centered at
        the ground truth timing.
        """
        batch_size, seq_len, num_events = timing_logits.shape
        
        # Create target distribution for timing
        # We create a Gaussian-like target centered at the true timing
        timesteps = torch.arange(seq_len, device=timing_logits.device).float()
        timesteps = timesteps.view(1, -1, 1)  # (1, seq_len, 1)
        
        # Expand event_timings to match shape
        event_timings_expanded = event_timings.unsqueeze(1)  # (batch_size, 1, num_events)
        
        # Create Gaussian target with width = timing_resolution_samples
        sigma = self.timing_resolution_samples / 2
        target_dist = torch.exp(
            -((timesteps - event_timings_expanded) ** 2) / (2 * sigma ** 2)
        )
        
        # Normalize to sum to 1
        target_dist = target_dist / (target_dist.sum(dim=1, keepdim=True) + 1e-8)
        
        # Apply mask for events that didn't occur
        event_mask = event_labels.unsqueeze(1)  # (batch_size, 1, num_events)
        
        # Compute KL divergence for events that occurred
        timing_probs = F.softmax(timing_logits, dim=1)
        kl_div = F.kl_div(
            torch.log(timing_probs + 1e-8),
            target_dist,
            reduction='none',
        )
        
        # Average over sequence and events, weighted by event mask
        kl_div_masked = kl_div * event_mask
        timing_loss = kl_div_masked.sum() / (event_mask.sum() + 1e-8)
        
        return timing_loss
    
    def _compute_refractory_loss(
        self,
        timing_logits: torch.Tensor,
        event_labels: torch.Tensor,
        event_timings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss to discourage predictions within refractory period.
        
        This is a soft constraint that penalizes having high probability mass
        within the refractory period of the predicted event timing.
        """
        batch_size, seq_len, num_events = timing_logits.shape
        
        # Get timing probabilities
        timing_probs = F.softmax(timing_logits, dim=1)
        
        # For each event, create a mask for the refractory period
        timesteps = torch.arange(seq_len, device=timing_logits.device).float()
        timesteps = timesteps.view(1, -1, 1)
        
        event_timings_expanded = event_timings.unsqueeze(1)
        
        # Create refractory mask: 1 where we should penalize, 0 otherwise
        refractory_mask = torch.abs(timesteps - event_timings_expanded) < self.refractory_period_samples
        refractory_mask = refractory_mask.float()
        
        # Zero out the exact event timing location (allow the true event)
        exact_timing_mask = torch.abs(timesteps - event_timings_expanded) < self.timing_resolution_samples
        refractory_mask = refractory_mask * (1 - exact_timing_mask.float())
        
        # Apply event mask (only for events that occurred)
        event_mask = event_labels.unsqueeze(1)
        refractory_mask = refractory_mask * event_mask
        
        # Penalize probability mass in refractory period
        refractory_loss = (timing_probs * refractory_mask).sum() / (event_mask.sum() + 1e-8)
        
        return refractory_loss


class AutoencoderLoss(nn.Module):
    """
    Loss for LSTM autoencoder.
    
    Args:
        reconstruction_weight: Weight for reconstruction loss (default: 1.0)
    """
    
    def __init__(self, reconstruction_weight: float = 1.0):
        super(AutoencoderLoss, self).__init__()
        self.reconstruction_weight = reconstruction_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reconstruction loss.
        
        Args:
            reconstruction: Reconstructed sequence (batch_size, seq_len, input_size)
            target: Original sequence (batch_size, seq_len, input_size)
            
        Returns:
            Dictionary containing loss
        """
        recon_loss = self.mse_loss(reconstruction, target)
        total_loss = self.reconstruction_weight * recon_loss
        
        return {
            "total_loss": total_loss,
            "reconstruction_loss": recon_loss,
        }


class VariationalAutoencoderLoss(nn.Module):
    """
    Loss for variational LSTM autoencoder.
    
    Args:
        reconstruction_weight: Weight for reconstruction loss (default: 1.0)
        kl_weight: Weight for KL divergence loss (default: 0.1)
    """
    
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        kl_weight: float = 0.1,
    ):
        super(VariationalAutoencoderLoss, self).__init__()
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss (reconstruction + KL divergence).
        
        Args:
            reconstruction: Reconstructed sequence (batch_size, seq_len, input_size)
            target: Original sequence (batch_size, seq_len, input_size)
            mu: Mean of latent distribution (batch_size, latent_size)
            logvar: Log variance of latent distribution (batch_size, latent_size)
            
        Returns:
            Dictionary containing total loss and components
        """
        # Reconstruction loss
        recon_loss = self.mse_loss(reconstruction, target)
        
        # KL divergence loss
        # KL(N(mu, sigma^2) || N(0, 1))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / target.shape[0]  # Normalize by batch size
        
        # Total loss
        total_loss = (
            self.reconstruction_weight * recon_loss +
            self.kl_weight * kl_loss
        )
        
        return {
            "total_loss": total_loss,
            "reconstruction_loss": recon_loss,
            "kl_loss": kl_loss,
        }
