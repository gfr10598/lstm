"""
Training loops for supervised and unsupervised learning.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from typing import Dict, Optional, Callable


class SupervisedTrainer:
    """
    Trainer for supervised event classification and timing prediction.
    
    Args:
        model: LSTMEventClassifier instance
        loss_fn: Loss function
        optimizer: PyTorch optimizer
        device: Device to train on (default: "cuda" if available)
        log_dir: Directory for tensorboard logs (default: "./runs")
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: Optional[str] = None,
        log_dir: str = "./runs",
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            Dictionary of average losses
        """
        self.model.train()
        epoch_losses = {
            "total_loss": 0.0,
            "event_loss": 0.0,
            "timing_loss": 0.0,
            "refractory_loss": 0.0,
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            inputs = batch["input"].to(self.device)
            event_labels = batch["event_labels"].to(self.device)
            event_timings = batch["event_timings"].to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Compute loss
            losses = self.loss_fn(
                event_logits=outputs["event_logits"],
                timing_logits=outputs["timing_logits"],
                event_labels=event_labels,
                event_timings=event_timings,
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            losses["total_loss"].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            
            # Log to tensorboard
            if batch_idx % 10 == 0:
                for key, value in losses.items():
                    self.writer.add_scalar(
                        f"train/{key}", value.item(), self.global_step
                    )
            
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                "loss": losses["total_loss"].item(),
                "event": losses["event_loss"].item(),
                "timing": losses["timing_loss"].item(),
            })
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
        
        return epoch_losses
    
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            epoch: Current epoch number
            
        Returns:
            Dictionary of average validation losses
        """
        self.model.eval()
        val_losses = {
            "total_loss": 0.0,
            "event_loss": 0.0,
            "timing_loss": 0.0,
            "refractory_loss": 0.0,
        }
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move data to device
                inputs = batch["input"].to(self.device)
                event_labels = batch["event_labels"].to(self.device)
                event_timings = batch["event_timings"].to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute loss
                losses = self.loss_fn(
                    event_logits=outputs["event_logits"],
                    timing_logits=outputs["timing_logits"],
                    event_labels=event_labels,
                    event_timings=event_timings,
                )
                
                # Update metrics
                for key in val_losses:
                    if key in losses:
                        val_losses[key] += losses[key].item()
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= len(val_loader)
        
        # Log to tensorboard
        for key, value in val_losses.items():
            self.writer.add_scalar(f"val/{key}", value, epoch)
        
        return val_losses
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        save_dir: str = "./checkpoints",
        save_freq: int = 10,
    ):
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            save_freq: Save checkpoint every N epochs
        """
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float("inf")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_losses = self.train_epoch(train_loader, epoch)
            
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_losses['total_loss']:.4f}")
            
            # Validate
            if val_loader is not None:
                val_losses = self.validate(val_loader, epoch)
                print(f"Val Loss: {val_losses['total_loss']:.4f}")
                
                # Save best model
                if val_losses["total_loss"] < best_val_loss:
                    best_val_loss = val_losses["total_loss"]
                    self.save_checkpoint(
                        os.path.join(save_dir, "best_model.pth"),
                        epoch,
                        val_losses["total_loss"],
                    )
            
            # Save periodic checkpoint
            if epoch % save_freq == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth"),
                    epoch,
                )
        
        self.writer.close()
    
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        val_loss: Optional[float] = None,
    ):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }
        if val_loss is not None:
            checkpoint["val_loss"] = val_loss
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        print(f"Loaded checkpoint from {path}")
        return checkpoint


class UnsupervisedTrainer:
    """
    Trainer for unsupervised pretraining with autoencoders.
    
    Args:
        model: Autoencoder model
        loss_fn: Loss function
        optimizer: PyTorch optimizer
        device: Device to train on
        log_dir: Directory for tensorboard logs
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: Optional[str] = None,
        log_dir: str = "./runs",
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {"total_loss": 0.0, "reconstruction_loss": 0.0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, inputs in enumerate(pbar):
            # Move data to device
            if isinstance(inputs, dict):
                inputs = inputs["input"]
            inputs = inputs.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Compute loss
            losses = self.loss_fn(
                reconstruction=outputs["reconstruction"],
                target=inputs,
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update metrics
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            
            # Log to tensorboard
            if batch_idx % 10 == 0:
                for key, value in losses.items():
                    self.writer.add_scalar(
                        f"train/{key}", value.item(), self.global_step
                    )
            
            self.global_step += 1
            pbar.set_postfix({"loss": losses["total_loss"].item()})
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
        
        return epoch_losses
    
    def fit(
        self,
        train_loader: DataLoader,
        num_epochs: int = 100,
        save_dir: str = "./checkpoints",
        save_freq: int = 10,
    ):
        """Train the model for multiple epochs."""
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(1, num_epochs + 1):
            train_losses = self.train_epoch(train_loader, epoch)
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_losses['total_loss']:.4f}")
            
            if epoch % save_freq == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth"),
                    epoch,
                )
        
        self.writer.close()
    
    def save_checkpoint(self, path: str, epoch: int):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")


class ContrastiveTrainer:
    """
    Trainer for contrastive learning.
    
    Args:
        model: Contrastive model
        optimizer: PyTorch optimizer
        device: Device to train on
        log_dir: Directory for tensorboard logs
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: Optional[str] = None,
        log_dir: str = "./runs",
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (view1, view2) in enumerate(pbar):
            # Move data to device
            view1 = view1.to(self.device)
            view2 = view2.to(self.device)
            
            # Forward pass
            outputs = self.model(view1, view2)
            
            # Compute contrastive loss
            loss = self.model.contrastive_loss(outputs["z1"], outputs["z2"])
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Log to tensorboard
            if batch_idx % 10 == 0:
                self.writer.add_scalar("train/contrastive_loss", loss.item(), self.global_step)
            
            self.global_step += 1
            pbar.set_postfix({"loss": loss.item()})
        
        # Average loss
        epoch_loss /= len(train_loader)
        
        return {"contrastive_loss": epoch_loss}
    
    def fit(
        self,
        train_loader: DataLoader,
        num_epochs: int = 100,
        save_dir: str = "./checkpoints",
        save_freq: int = 10,
    ):
        """Train the model for multiple epochs."""
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(1, num_epochs + 1):
            train_losses = self.train_epoch(train_loader, epoch)
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_losses['contrastive_loss']:.4f}")
            
            if epoch % save_freq == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth"),
                    epoch,
                )
        
        self.writer.close()
    
    def save_checkpoint(self, path: str, epoch: int):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
