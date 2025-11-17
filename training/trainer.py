"""
Training utilities for the Audio Language Model (ALM)
Handles model training, validation, and checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
from tqdm import tqdm
from typing import Dict, Any, Tuple
import wandb  # For experiment tracking (optional)


class ALMTrainer:
    """
    Trainer class for the Audio Language Model
    Handles training loop, validation, and checkpointing
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any], device: torch.device):
        """
        Initialize the trainer
        
        Args:
            model: ALM model to train
            config: Configuration dictionary
            device: Device to train on (CPU/GPU)
        """
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=0.01
        )
        
        # Initialize learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['num_epochs'],
            eta_min=1e-6
        )
        
        # Loss functions for different tasks
        self.speech_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.event_loss_fn = nn.CrossEntropyLoss()
        self.speaker_loss_fn = nn.CrossEntropyLoss()
        self.paralinguistic_loss_fn = nn.CrossEntropyLoss()
        
        # Training statistics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Initialize wandb for experiment tracking (optional)
        self.use_wandb = False
        
    def compute_loss(self, model_outputs: Dict[str, torch.Tensor], 
                     targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss for all tasks
        
        Args:
            model_outputs: Dictionary of model outputs
            targets: Dictionary of target tensors
            
        Returns:
            Tuple of (total_loss, individual_losses)
        """
        losses = {}
        
        # Speech recognition loss
        if model_outputs['speech_logits'] is not None and targets.get('speech_target') is not None:
            speech_loss = self.speech_loss_fn(
                model_outputs['speech_logits'].view(-1, model_outputs['speech_logits'].size(-1)),
                targets['speech_target'].view(-1)
            )
            losses['speech_loss'] = speech_loss
        
        # Audio event detection loss
        if model_outputs['audio_event_logits'] is not None and targets.get('audio_event_target') is not None:
            event_loss = self.event_loss_fn(
                model_outputs['audio_event_logits'],
                targets['audio_event_target'].squeeze()
            )
            losses['event_loss'] = event_loss
        
        # Speaker diarization loss
        if model_outputs['speaker_logits'] is not None and targets.get('speaker_target') is not None:
            speaker_loss = self.speaker_loss_fn(
                model_outputs['speaker_logits'].view(-1, model_outputs['speaker_logits'].size(-1)),
                targets['speaker_target'].view(-1)
            )
            losses['speaker_loss'] = speaker_loss
        
        # Paralinguistic analysis loss
        if model_outputs['paralinguistic_logits'] is not None and targets.get('paralinguistic_target') is not None:
            paralinguistic_loss = self.paralinguistic_loss_fn(
                model_outputs['paralinguistic_logits'],
                targets['paralinguistic_target'].squeeze()
            )
            losses['paralinguistic_loss'] = paralinguistic_loss
        
        # Total loss (weighted combination)
        weights = {
            'speech_loss': 1.0,
            'event_loss': 0.5,
            'speaker_loss': 0.5,
            'paralinguistic_loss': 0.3
        }
        
        total_loss = sum(weights[key] * loss for key, loss in losses.items() if key in weights)
        weighted_losses = {key: weights.get(key, 1.0) * loss for key, loss in losses.items()}
        
        return total_loss, weighted_losses
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train for one epoch
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(
                audio=batch['audio'],
                mel_spectrogram=batch['mel_spectrogram'],
                language_ids=None  # In a real implementation, you would pass language IDs
            )
            
            # Prepare targets
            targets = {
                'speech_target': batch['speech_target'],
                'audio_event_target': batch['audio_event_target'],
                'speaker_target': batch['speaker_target'],
                'paralinguistic_target': batch['paralinguistic_target']
            }
            
            # Compute loss
            loss, individual_losses = self.compute_loss(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['training']['gradient_clip']
            )
            
            # Update weights
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'speech': f"{individual_losses.get('speech_loss', 0):.4f}",
                'event': f"{individual_losses.get('event_loss', 0):.4f}"
            })
            
            # Log to wandb if enabled
            if self.use_wandb:
                wandb.log({
                    'train_batch_loss': loss.item(),
                    **{f'train_{k}': v.item() for k, v in individual_losses.items()}
                })
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, dataloader: DataLoader) -> float:
        """
        Validate the model
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    audio=batch['audio'],
                    mel_spectrogram=batch['mel_spectrogram'],
                    language_ids=None
                )
                
                # Prepare targets
                targets = {
                    'speech_target': batch['speech_target'],
                    'audio_event_target': batch['audio_event_target'],
                    'speaker_target': batch['speaker_target'],
                    'paralinguistic_target': batch['paralinguistic_target']
                }
                
                # Compute loss
                loss, _ = self.compute_loss(outputs, targets)
                
                # Update statistics
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int = None) -> None:
        """
        Main training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train (defaults to config value)
        """
        if num_epochs is None:
            num_epochs = self.config['training']['num_epochs']
        
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train for one epoch
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Print epoch statistics
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save checkpoint if validation loss improved
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(f"best_model_epoch_{epoch+1}.pt")
                print(f"New best model saved with validation loss: {val_loss:.4f}")
            
            # Save checkpoint periodically
            if (epoch + 1) % self.config['training']['save_checkpoint_every'] == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
            
            # Log to wandb if enabled
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'lr': self.optimizer.param_groups[0]['lr']
                })
    
    def save_checkpoint(self, filename: str) -> None:
        """
        Save model checkpoint
        
        Args:
            filename: Name of the checkpoint file
        """
        checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        print(f"Checkpoint loaded from {checkpoint_path}")


# Example usage
if __name__ == '__main__':
    # This would typically be run from a training script
    pass