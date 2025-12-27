"""
LSTM Event Classifier Package

A PyTorch-based LSTM model for event classification and timing prediction
with support for unsupervised pretraining and supervised learning.
"""

__version__ = "0.1.0"

from .models.lstm_model import LSTMEventClassifier
from .models.autoencoder import LSTMAutoencoder
from .models.contrastive import ContrastiveLSTM

__all__ = [
    "LSTMEventClassifier",
    "LSTMAutoencoder",
    "ContrastiveLSTM",
]
