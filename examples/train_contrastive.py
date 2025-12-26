"""
Example script for contrastive learning pretraining.

This script demonstrates how to pretrain the LSTM backbone using contrastive learning
with temporal augmentations.
"""

import torch
from torch.utils.data import DataLoader

from lstm_classifier.models import ContrastiveLSTM, TemporalAugmentation, LSTMEventClassifier
from lstm_classifier.data import SyntheticEventDataset, ContrastiveDataset
from lstm_classifier.training import ContrastiveTrainer


def main():
    # Configuration
    config = {
        "input_size": 6,
        "hidden_size": 128,
        "num_layers": 2,
        "projection_size": 64,
        "dropout": 0.3,
        "bidirectional": True,
        "temperature": 0.07,
        "seq_len": 2000,
        "num_samples": 1000,
        "batch_size": 32,
        "num_epochs": 50,
        "learning_rate": 1e-3,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    
    print("Creating dataset for contrastive learning...")
    # Create dataset
    full_dataset = SyntheticEventDataset(
        num_samples=config["num_samples"],
        seq_len=config["seq_len"],
        input_size=config["input_size"],
    )
    
    # Convert to contrastive dataset
    import numpy as np
    data = np.array([full_dataset[i]["input"].numpy() for i in range(len(full_dataset))])
    dataset = ContrastiveDataset(
        data,
        augmentation=TemporalAugmentation.random_augment,
    )
    
    # Create data loader
    from lstm_classifier.data import collate_fn_contrastive
    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn_contrastive,
        num_workers=4,
    )
    
    print(f"Training samples: {len(dataset)}")
    
    # Create contrastive model
    print("Initializing contrastive model...")
    model = ContrastiveLSTM(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        projection_size=config["projection_size"],
        dropout=config["dropout"],
        bidirectional=config["bidirectional"],
        temperature=config["temperature"],
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
    )
    
    # Create trainer
    trainer = ContrastiveTrainer(
        model=model,
        optimizer=optimizer,
        device=config["device"],
        log_dir="./runs/contrastive",
    )
    
    # Train
    print("Starting contrastive learning...")
    trainer.fit(
        train_loader=train_loader,
        num_epochs=config["num_epochs"],
        save_dir="./checkpoints/contrastive",
        save_freq=10,
    )
    
    print("Contrastive pretraining completed!")
    
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
    
    model.transfer_encoder_to_classifier(classifier)
    
    # Save pretrained classifier
    torch.save(
        classifier.state_dict(),
        "./checkpoints/contrastive_pretrained_classifier.pth"
    )
    
    print("Pretrained classifier saved to ./checkpoints/contrastive_pretrained_classifier.pth")
    print("You can now fine-tune this model on labeled data")


if __name__ == "__main__":
    main()
