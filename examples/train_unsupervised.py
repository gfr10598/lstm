"""
Example script for unsupervised pretraining with autoencoder.

This script demonstrates how to pretrain the LSTM backbone using an autoencoder
before fine-tuning on labeled data.
"""

import torch
from torch.utils.data import DataLoader

from lstm_classifier.models import LSTMAutoencoder, LSTMEventClassifier
from lstm_classifier.data import SyntheticEventDataset, UnsupervisedDataset
from lstm_classifier.training import AutoencoderLoss, UnsupervisedTrainer


def main():
    # Configuration
    config = {
        "input_size": 6,
        "hidden_size": 128,
        "num_layers": 2,
        "latent_size": 64,
        "dropout": 0.3,
        "bidirectional": True,
        "seq_len": 2000,
        "num_samples": 1000,
        "batch_size": 32,
        "num_epochs": 50,
        "learning_rate": 1e-3,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    
    print("Creating dataset for unsupervised pretraining...")
    # Create dataset (using synthetic data, but labels are ignored)
    full_dataset = SyntheticEventDataset(
        num_samples=config["num_samples"],
        seq_len=config["seq_len"],
        input_size=config["input_size"],
    )
    
    # Convert to unsupervised dataset (only uses input data)
    import numpy as np
    data = np.array([full_dataset[i]["input"].numpy() for i in range(len(full_dataset))])
    dataset = UnsupervisedDataset(data)
    
    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
    )
    
    print(f"Training samples: {len(dataset)}")
    
    # Create autoencoder model
    print("Initializing autoencoder...")
    autoencoder = LSTMAutoencoder(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        latent_size=config["latent_size"],
        dropout=config["dropout"],
        bidirectional=config["bidirectional"],
    )
    
    print(f"Model parameters: {sum(p.numel() for p in autoencoder.parameters()):,}")
    
    # Create loss function
    loss_fn = AutoencoderLoss()
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        autoencoder.parameters(),
        lr=config["learning_rate"],
    )
    
    # Create trainer
    trainer = UnsupervisedTrainer(
        model=autoencoder,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=config["device"],
        log_dir="./runs/unsupervised_autoencoder",
    )
    
    # Train
    print("Starting unsupervised pretraining...")
    trainer.fit(
        train_loader=train_loader,
        num_epochs=config["num_epochs"],
        save_dir="./checkpoints/unsupervised_autoencoder",
        save_freq=10,
    )
    
    print("Pretraining completed!")
    
    # Transfer learned weights to classifier
    print("\nTransferring encoder weights to classifier...")
    classifier = LSTMEventClassifier(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_events=16,
        dropout=config["dropout"],
        bidirectional=config["bidirectional"],
    )
    
    autoencoder.transfer_encoder_to_classifier(classifier)
    
    # Save pretrained classifier
    torch.save(
        classifier.state_dict(),
        "./checkpoints/pretrained_classifier.pth"
    )
    
    print("Pretrained classifier saved to ./checkpoints/pretrained_classifier.pth")
    print("You can now fine-tune this model on labeled data using train_supervised.py")


if __name__ == "__main__":
    main()
