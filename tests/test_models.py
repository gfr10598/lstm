"""
Tests for LSTM Event Classifier models.
"""

import torch
import pytest
import numpy as np

from lstm_classifier.models import (
    LSTMEventClassifier,
    LSTMAutoencoder,
    VariationalLSTMAutoencoder,
    ContrastiveLSTM,
)


class TestLSTMEventClassifier:
    """Tests for LSTMEventClassifier."""
    
    def test_forward_pass(self):
        """Test basic forward pass."""
        model = LSTMEventClassifier(
            input_size=6,
            hidden_size=64,
            num_layers=2,
            num_events=16,
        )
        
        batch_size = 4
        seq_len = 100
        x = torch.randn(batch_size, seq_len, 6)
        
        output = model(x)
        
        assert "event_logits" in output
        assert "timing_logits" in output
        assert "event_probs" in output
        assert "timing_probs" in output
        
        assert output["event_logits"].shape == (batch_size, 16)
        assert output["timing_logits"].shape == (batch_size, seq_len, 16)
        assert output["event_probs"].shape == (batch_size, 16)
        assert output["timing_probs"].shape == (batch_size, seq_len, 16)
    
    def test_predict_events(self):
        """Test event prediction."""
        model = LSTMEventClassifier(
            input_size=6,
            hidden_size=64,
            num_layers=2,
            num_events=16,
        )
        
        batch_size = 4
        seq_len = 100
        x = torch.randn(batch_size, seq_len, 6)
        
        predictions = model.predict_events(x, threshold=0.5)
        
        assert "detected_events" in predictions
        assert "event_timings" in predictions
        assert "event_probs" in predictions
        
        assert predictions["detected_events"].shape == (batch_size, 16)
        assert predictions["event_timings"].shape == (batch_size, 16)
        assert predictions["event_probs"].shape == (batch_size, 16)
    
    def test_freeze_unfreeze(self):
        """Test freezing and unfreezing backbone."""
        model = LSTMEventClassifier(input_size=6, hidden_size=64)
        
        # Initially all parameters should be trainable
        for param in model.lstm.parameters():
            assert param.requires_grad
        
        # Freeze
        model.freeze_backbone()
        for param in model.lstm.parameters():
            assert not param.requires_grad
        
        # Unfreeze
        model.unfreeze_backbone()
        for param in model.lstm.parameters():
            assert param.requires_grad


class TestLSTMAutoencoder:
    """Tests for LSTM Autoencoder."""
    
    def test_forward_pass(self):
        """Test autoencoder forward pass."""
        model = LSTMAutoencoder(
            input_size=6,
            hidden_size=64,
            num_layers=2,
            latent_size=32,
        )
        
        batch_size = 4
        seq_len = 100
        x = torch.randn(batch_size, seq_len, 6)
        
        output = model(x)
        
        assert "reconstruction" in output
        assert "latent" in output
        
        assert output["reconstruction"].shape == (batch_size, seq_len, 6)
        assert output["latent"].shape == (batch_size, 32)
    
    def test_encode_decode(self):
        """Test encode and decode separately."""
        model = LSTMAutoencoder(
            input_size=6,
            hidden_size=64,
            latent_size=32,
        )
        
        x = torch.randn(4, 100, 6)
        
        latent = model.encode(x)
        assert latent.shape == (4, 32)
        
        reconstruction = model.decode(latent, seq_len=100)
        assert reconstruction.shape == (4, 100, 6)
    
    def test_transfer_to_classifier(self):
        """Test transferring weights to classifier."""
        autoencoder = LSTMAutoencoder(
            input_size=6,
            hidden_size=64,
            num_layers=2,
            bidirectional=True,
        )
        
        classifier = LSTMEventClassifier(
            input_size=6,
            hidden_size=64,
            num_layers=2,
            bidirectional=True,
        )
        
        # Transfer should not raise error
        autoencoder.transfer_encoder_to_classifier(classifier)


class TestVariationalAutoencoder:
    """Tests for Variational LSTM Autoencoder."""
    
    def test_forward_pass(self):
        """Test VAE forward pass."""
        model = VariationalLSTMAutoencoder(
            input_size=6,
            hidden_size=64,
            latent_size=32,
        )
        
        x = torch.randn(4, 100, 6)
        output = model(x)
        
        assert "reconstruction" in output
        assert "mu" in output
        assert "logvar" in output
        assert "latent" in output
        
        assert output["reconstruction"].shape == (4, 100, 6)
        assert output["mu"].shape == (4, 32)
        assert output["logvar"].shape == (4, 32)
        assert output["latent"].shape == (4, 32)
    
    def test_reparameterization(self):
        """Test reparameterization trick."""
        model = VariationalLSTMAutoencoder(input_size=6, latent_size=32)
        
        mu = torch.randn(4, 32)
        logvar = torch.randn(4, 32)
        
        z = model.reparameterize(mu, logvar)
        assert z.shape == (4, 32)


class TestContrastiveLSTM:
    """Tests for Contrastive LSTM."""
    
    def test_forward_single_view(self):
        """Test forward with single view."""
        model = ContrastiveLSTM(
            input_size=6,
            hidden_size=64,
            projection_size=32,
        )
        
        x = torch.randn(4, 100, 6)
        output = model(x)
        
        assert "z1" in output
        assert "h1" in output
        assert output["z1"].shape == (4, 32)
    
    def test_forward_two_views(self):
        """Test forward with two views."""
        model = ContrastiveLSTM(
            input_size=6,
            hidden_size=64,
            projection_size=32,
        )
        
        x1 = torch.randn(4, 100, 6)
        x2 = torch.randn(4, 100, 6)
        
        output = model(x1, x2)
        
        assert "z1" in output
        assert "z2" in output
        assert output["z1"].shape == (4, 32)
        assert output["z2"].shape == (4, 32)
    
    def test_contrastive_loss(self):
        """Test contrastive loss computation."""
        model = ContrastiveLSTM(
            input_size=6,
            hidden_size=64,
            projection_size=32,
        )
        
        x1 = torch.randn(4, 100, 6)
        x2 = torch.randn(4, 100, 6)
        
        output = model(x1, x2)
        loss = model.contrastive_loss(output["z1"], output["z2"])
        
        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0


if __name__ == "__main__":
    pytest.main([__file__])
