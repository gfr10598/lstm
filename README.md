# LSTM Event Classifier for Timing Prediction

A PyTorch-based LSTM model for event classification and timing prediction with support for unsupervised/semi-supervised pretraining and supervised fine-tuning.

## Overview

This project implements a comprehensive LSTM-based deep learning system designed to:
- **Identify 16 discrete events** from 6-dimensional sensor data
- **Predict event timing** at 5 millisecond resolution (10 samples at 2000 Hz)
- **Enforce 2-second refractory period** per event type (no event recurrence within 4000 samples)
- **Support unsupervised pretraining** when labeled data is unavailable
- **Enable easy transition** to supervised learning when labels become available

## Features

### Core Capabilities
- ✅ **Multi-event classification**: Simultaneous detection of up to 16 event types
- ✅ **High-precision timing**: 5ms resolution timing prediction (10 samples @ 2000 Hz)
- ✅ **Refractory period enforcement**: Built-in constraint to prevent event recurrence within 2 seconds
- ✅ **Bidirectional LSTM**: Captures both past and future context
- ✅ **Attention mechanism**: Focuses on relevant temporal regions
- ✅ **Modular architecture**: Easy to customize and extend

### Unsupervised Pretraining
- ✅ **LSTM Autoencoder**: Reconstruction-based unsupervised learning
- ✅ **Variational Autoencoder**: Robust latent representation learning
- ✅ **Contrastive Learning**: Self-supervised learning with temporal augmentations
- ✅ **Transfer Learning**: Easy weight transfer from pretrained to supervised models

### Training & Inference
- ✅ **Custom loss functions**: Combined event classification, timing prediction, and refractory constraints
- ✅ **Multiple training modes**: Supervised, unsupervised, and contrastive learning
- ✅ **TensorBoard integration**: Real-time training monitoring
- ✅ **Checkpoint management**: Save and resume training
- ✅ **Batch inference**: Efficient processing of multiple sequences

## Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- NumPy
- SciPy
- scikit-learn

### Setup

```bash
# Clone the repository
git clone https://github.com/gfr10598/lstm.git
cd lstm

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### 1. Supervised Training (when labels are available)

```python
from lstm_classifier.models import LSTMEventClassifier
from lstm_classifier.data import EventDataset, collate_fn_supervised
from lstm_classifier.training import EventTimingLoss, SupervisedTrainer
from torch.utils.data import DataLoader
import torch

# Create model
model = LSTMEventClassifier(
    input_size=6,
    hidden_size=128,
    num_layers=2,
    num_events=16,
    dropout=0.3,
    bidirectional=True,
)

# Prepare your data
# data: numpy array of shape (num_samples, seq_len, 6)
# event_labels: numpy array of shape (num_samples, 16) - binary labels
# event_timings: numpy array of shape (num_samples, 16) - timing in samples

dataset = EventDataset(data, event_labels, event_timings)
train_loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn_supervised)

# Setup training
loss_fn = EventTimingLoss(
    event_weight=1.0,
    timing_weight=1.0,
    refractory_weight=0.1,
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

trainer = SupervisedTrainer(model, loss_fn, optimizer)
trainer.fit(train_loader, num_epochs=100)
```

### 2. Unsupervised Pretraining (when labels are NOT available)

```python
from lstm_classifier.models import LSTMAutoencoder
from lstm_classifier.data import UnsupervisedDataset
from lstm_classifier.training import AutoencoderLoss, UnsupervisedTrainer

# Create autoencoder
autoencoder = LSTMAutoencoder(
    input_size=6,
    hidden_size=128,
    num_layers=2,
    latent_size=64,
)

# Prepare unlabeled data
dataset = UnsupervisedDataset(data)  # Only needs input data
train_loader = DataLoader(dataset, batch_size=32)

# Train autoencoder
loss_fn = AutoencoderLoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)

trainer = UnsupervisedTrainer(autoencoder, loss_fn, optimizer)
trainer.fit(train_loader, num_epochs=50)

# Transfer learned weights to classifier
classifier = LSTMEventClassifier(input_size=6, hidden_size=128, num_layers=2)
autoencoder.transfer_encoder_to_classifier(classifier)

# Now fine-tune classifier on labeled data when available
```

### 3. Contrastive Learning Pretraining

```python
from lstm_classifier.models import ContrastiveLSTM, TemporalAugmentation
from lstm_classifier.data import ContrastiveDataset, collate_fn_contrastive
from lstm_classifier.training import ContrastiveTrainer

# Create contrastive model
model = ContrastiveLSTM(
    input_size=6,
    hidden_size=128,
    projection_size=64,
)

# Create dataset with augmentations
dataset = ContrastiveDataset(
    data,
    augmentation=TemporalAugmentation.random_augment,
)
train_loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn_contrastive)

# Train with contrastive learning
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
trainer = ContrastiveTrainer(model, optimizer)
trainer.fit(train_loader, num_epochs=50)

# Transfer to classifier
classifier = LSTMEventClassifier(input_size=6, hidden_size=128)
model.transfer_encoder_to_classifier(classifier)
```

### 4. Inference

```python
from lstm_classifier.utils import detect_events_from_sequence
import numpy as np

# Load trained model
model = LSTMEventClassifier(...)
checkpoint = torch.load("checkpoints/best_model.pth")
model.load_state_dict(checkpoint["model_state_dict"])

# Detect events from new sequence
sequence = np.random.randn(2000, 6)  # 1 second @ 2000 Hz
results = detect_events_from_sequence(
    model=model,
    sequence=sequence,
    threshold=0.5,
    apply_refractory=True,
)

print("Detected events:", results["detected_events"])
print("Event timings (samples):", results["event_timings"])
print("Event probabilities:", results["event_probs"])
```

## Architecture

### Model Components

1. **LSTM Backbone**
   - Bidirectional LSTM for temporal feature extraction
   - Multi-layer architecture with dropout regularization
   - Processes 6-dimensional input at 2000 Hz

2. **Event Classification Head**
   - Multi-label binary classification for 16 events
   - Sigmoid activation for independent event probabilities
   - Mean-pooled temporal features

3. **Timing Prediction Head**
   - Per-timestep logits for each event type
   - Softmax normalization over time dimension
   - 5ms resolution (10 samples @ 2000 Hz)

4. **Attention Mechanism**
   - Multi-head self-attention for temporal modeling
   - Helps focus on relevant time periods
   - Improves timing precision

### Refractory Period Enforcement

The model enforces a 2-second (4000 samples) refractory period per event type through:

1. **Loss-based constraint**: Soft penalty in training loss
2. **Post-processing filter**: Hard constraint during inference
3. **Stateful tracking**: Maintains event history across batches

```python
from lstm_classifier.utils import RefractoryPeriodEnforcer

enforcer = RefractoryPeriodEnforcer(
    num_events=16,
    refractory_period_samples=4000,
)

# Process sequences with refractory enforcement
for sequence in sequences:
    results = detect_events_from_sequence(model, sequence)
    filtered_events, filtered_timings = enforcer.enforce(
        results["detected_events"],
        results["event_timings"],
        global_offset=current_time,
    )
```

## Examples

The `examples/` directory contains complete training and inference scripts:

- `train_supervised.py` - Supervised training with synthetic data
- `train_unsupervised.py` - Autoencoder-based pretraining
- `train_contrastive.py` - Contrastive learning pretraining
- `inference.py` - Event detection on new data

Run examples:
```bash
python examples/train_supervised.py
python examples/train_unsupervised.py
python examples/train_contrastive.py
python examples/inference.py
```

## Testing

Run tests with pytest:

```bash
# Install test dependencies
pip install pytest

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest -v tests/
```

## Model Configuration

### Key Hyperparameters

```python
LSTMEventClassifier(
    input_size=6,                    # Number of input features
    hidden_size=128,                 # LSTM hidden units
    num_layers=2,                    # Number of LSTM layers
    num_events=16,                   # Number of event classes
    dropout=0.3,                     # Dropout probability
    bidirectional=True,              # Bidirectional LSTM
    timing_resolution_samples=10,    # 5ms @ 2000Hz
    refractory_period_samples=4000,  # 2s @ 2000Hz
)
```

### Training Configuration

```python
EventTimingLoss(
    event_weight=1.0,        # Weight for event classification
    timing_weight=1.0,       # Weight for timing prediction
    refractory_weight=0.1,   # Weight for refractory constraint
)
```

## Data Format

### Input Data
- **Shape**: `(num_samples, seq_len, 6)`
- **Type**: `float32`
- **Range**: Normalized sensor readings
- **Sampling rate**: 2000 Hz (0.5 ms intervals)

### Event Labels (Supervised)
- **Shape**: `(num_samples, 16)`
- **Type**: `float32` (0 or 1)
- **Meaning**: Binary indicator for each event type

### Event Timings (Supervised)
- **Shape**: `(num_samples, 16)`
- **Type**: `float32`
- **Range**: 0 to `seq_len-1` (sample indices)
- **Resolution**: Rounded to multiples of 10 (5ms)

## Design Decisions & Trade-offs

### 1. Unsupervised vs Supervised
- **Challenge**: Event timing is inherently a supervised task
- **Solution**: Pretrain feature extractor (LSTM) unsupervised, then fine-tune on labels
- **Benefit**: Learn temporal patterns before labels are available

### 2. Timing Representation
- **Approach**: Softmax over timesteps per event
- **Benefit**: Differentiable, handles uncertainty
- **Alternative**: Regression (less flexible for multi-modal distributions)

### 3. Refractory Period
- **Soft constraint**: Loss penalty during training
- **Hard constraint**: Post-processing filter during inference
- **Benefit**: Both encourage compliance without being overly restrictive

### 4. Multi-task Learning
- **Combined loss**: Event detection + timing prediction
- **Benefit**: Shared representations improve both tasks
- **Trade-off**: Requires loss balancing (event_weight, timing_weight)

## Future Enhancements

Potential improvements when real data becomes available:

1. **Data-driven augmentation**: Learn augmentations from real sensor characteristics
2. **Event-specific models**: Separate timing predictors per event type
3. **Uncertainty quantification**: Bayesian approaches for confidence estimates
4. **Real-time inference**: Optimize for streaming data processing
5. **Multi-modal fusion**: Incorporate additional sensor modalities

## Contributing

Contributions are welcome! Areas of interest:
- Additional pretraining strategies
- Improved refractory period enforcement
- Real-time inference optimizations
- Visualization tools for predictions

## License

MIT License - see LICENSE file for details

## Citation

If you use this code in your research, please cite:

```bibtex
@software{lstm_event_classifier,
  title = {LSTM Event Classifier for Timing Prediction},
  author = {gfr10598},
  year = {2024},
  url = {https://github.com/gfr10598/lstm}
}
```

## Contact

For questions or issues, please open a GitHub issue.
