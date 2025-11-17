"""
Main training script for the Audio Language Model (ALM)
"""

import torch
import yaml
import argparse
import os
from torch.utils.data import DataLoader
from data.datasets import get_data_loader
from models.alm_model import ALMModel
from training.trainer import ALMTrainer


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main():
    """
    Main training function
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Audio Language Model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = get_data_loader(
        data_dir=args.data_dir,
        config=config,
        split='train',
        shuffle=True
    )
    
    val_loader = get_data_loader(
        data_dir=args.data_dir,
        config=config,
        split='val',
        shuffle=False
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Initialize model
    print("Initializing model...")
    model = ALMModel(config)
    
    # Initialize trainer
    trainer = ALMTrainer(model, config, device)
    
    # Load checkpoint if specified
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    
    # Start training
    print("Starting training...")
    trainer.train(train_loader, val_loader)
    
    # Save final model
    trainer.save_checkpoint('final_model.pt')
    print("Training completed!")


if __name__ == '__main__':
    main()