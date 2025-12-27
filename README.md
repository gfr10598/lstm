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

## Event-Specific Architecture

### Overview

The event-specific architecture incorporates domain knowledge about events having distinct feature shapes and cross-feature onset skews. This specialized approach improves event detection and timing prediction by learning event-type-specific patterns.

### Key Components

#### 1. EventTemplateBank

Learns one template per event type with feature-specific onset delays:

```python
from lstm_classifier.models import EventTemplateBank

template_bank = EventTemplateBank(
    num_events=16,
    num_features=6,
    template_length=80,  # 40ms @ 2000Hz
)

# Compute template match scores
template_scores = template_bank(x)  # (batch, seq_len, num_events)
```

Features:
- **Learnable templates**: Shape `(num_events, num_features, template_length)` captures full event duration (~40ms)
- **Feature-specific onset delays**: Shape `(num_events, num_features)` models cross-feature skews (±5ms)
- **Template matching**: Cross-correlation using `F.conv1d` for efficient matching

#### 2. EventSpecificCNN

Per-event convolutional pathways for learning event-specific temporal patterns:

```python
from lstm_classifier.models import EventSpecificCNN

event_cnn = EventSpecificCNN(
    num_events=16,
    input_channels=6,
    output_channels=32,
    use_shared=True,
)

# Extract event-specific features
event_features, shared_features = event_cnn(x)
# event_features: (batch, seq_len, num_events, 32)
# shared_features: (batch, seq_len, 32)
```

Each event has 3 conv layers at different temporal scales:
- **Layer 1**: kernel=11 (5ms) for onset detection
- **Layer 2**: kernel=21 (10ms) for shape detection  
- **Layer 3**: kernel=41 (20ms) for full event pattern

#### 3. EventSpecificTimingHeads

Per-event timing predictors with temporal attention:

```python
from lstm_classifier.models import EventSpecificTimingHeads

timing_heads = EventSpecificTimingHeads(
    num_events=16,
    hidden_size=128,
    dropout=0.2,
)

# Predict timing for each event
timing_logits = timing_heads(lstm_features, event_mask)
```

Features:
- **Per-event MLPs**: 2-layer networks for timing prediction
- **Temporal attention**: 4-head attention per event type
- **Event masking**: Focus on present events only

#### 4. EventSpecificClassifier (Integrated Model)

Complete end-to-end model combining all components:

```python
from lstm_classifier.models import EventSpecificClassifier
from lstm_classifier.training import EventSpecificLoss
from lstm_classifier.utils import visualize_learned_templates

# Create model
model = EventSpecificClassifier(
    input_size=6,
    hidden_size=128,
    num_layers=2,
    num_events=16,
    dropout=0.3,
    template_length=80,
    bidirectional=True,
)

# Forward pass
event_logits, timing_logits, template_scores = model(x, lengths)

# Training with event-specific loss
criterion = EventSpecificLoss(
    event_weight=1.0,
    timing_weight=1.0,
    template_weight=0.5,
    refractory_weight=0.1,
)

loss_dict = criterion(
    event_logits, timing_logits, template_scores,
    event_labels, event_timings
)

# Visualize learned templates
visualize_learned_templates(model, 'templates.png')
```

**Architecture Flow**:
1. Template matching → match scores for each event
2. Event-specific CNN → event-specific and shared features
3. Concatenate features (input + templates + CNN)
4. Bidirectional LSTM → temporal encoding
5. Event classification head → which events are present
6. Event-specific timing heads → when each event occurs

### Loss Function

The `EventSpecificLoss` combines multiple objectives:

```python
from lstm_classifier.training import EventSpecificLoss

loss_fn = EventSpecificLoss(
    event_weight=1.0,      # Event classification (BCE)
    timing_weight=1.0,     # Timing prediction (NLL)
    template_weight=0.5,   # Template alignment (MSE with Gaussian targets)
    refractory_weight=0.1, # Refractory period constraint
)
```

**Loss Components**:
1. **Event classification**: Binary cross-entropy for event presence
2. **Timing prediction**: Negative log-likelihood at true timing (masked by event presence)
3. **Template alignment**: MSE between template scores and Gaussian targets (σ=10 samples) centered at true timing
4. **Refractory period**: Optional constraint (can reuse from base model)

### Template Visualization

Visualize learned event templates with feature-specific delays:

```python
from lstm_classifier.utils import visualize_learned_templates

# After training
visualize_learned_templates(model, save_path='templates.png')
```

Creates a 4×4 grid showing:
- All 16 event types
- 6 features per event (F0-F5)
- Feature-specific delays applied
- Temporal patterns learned

### Training Example

Complete training script available in `examples/train_event_specific.py`:

```bash
python examples/train_event_specific.py
```

Key features:
- Generates synthetic data with event-specific patterns
- Trains EventSpecificClassifier for 100 epochs
- Saves template visualizations every 10 epochs
- Checkpoints best validation model

### Use Cases

The event-specific architecture is ideal when:
- Different event types have distinct temporal signatures
- Features show onset skews (delayed activation across channels)
- Events have characteristic decay patterns
- You need interpretable learned patterns (via template visualization)

### Performance Characteristics

**Advantages**:
- Better timing precision for events with clear templates
- Learns interpretable event-specific patterns
- Handles feature-specific onset delays naturally
- Improved robustness to event-specific noise

**Trade-offs**:
- Larger model (16× more parameters in event-specific pathways)
- Requires more data to train per-event components
- Template matching adds computational cost

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
