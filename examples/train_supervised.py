"""
Example script for supervised training of event classifier.

This script demonstrates how to train the LSTMEventClassifier on labeled data.
"""

import torch
from torch.utils.data import DataLoader, random_split

from lstm_classifier.models import LSTMEventClassifier
from lstm_classifier.data import SyntheticEventDataset, collate_fn_supervised
from lstm_classifier.training import EventTimingLoss, SupervisedTrainer


def main():
    # Configuration
    config = {
        "input_size": 6,
        "hidden_size": 128,
        "num_layers": 2,
        "num_events": 16,
        "dropout": 0.3,
        "bidirectional": True,
        "seq_len": 2000,  # 1 second at 2000 Hz
        "num_samples": 1000,
        "batch_size": 32,
        "num_epochs": 100,
        "learning_rate": 1e-3,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    
    print("Creating synthetic dataset...")
    # Create synthetic dataset for demonstration
    dataset = SyntheticEventDataset(
        num_samples=config["num_samples"],
        seq_len=config["seq_len"],
        input_size=config["input_size"],
        num_events=config["num_events"],
    )
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn_supervised,
        num_workers=4,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn_supervised,
        num_workers=4,
    )
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Create model
    print("Initializing model...")
    model = LSTMEventClassifier(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_events=config["num_events"],
        dropout=config["dropout"],
        bidirectional=config["bidirectional"],
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function
    loss_fn = EventTimingLoss(
        event_weight=1.0,
        timing_weight=1.0,
        refractory_weight=0.1,
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
    )
    
    # Create trainer
    trainer = SupervisedTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=config["device"],
        log_dir="./runs/supervised",
    )
    
    # Train
    print("Starting training...")
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config["num_epochs"],
        save_dir="./checkpoints/supervised",
        save_freq=10,
    )
    
    print("Training completed!")


if __name__ == "__main__":
    main()
