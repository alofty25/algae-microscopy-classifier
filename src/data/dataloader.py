"""
Data loader utilities for Algae Microscopy Detection
"""
import torch
from torch.utils.data import DataLoader
import yaml
from pathlib import Path
from typing import Tuple, Optional

from .algae_dataset import AlgaeDataset, collate_fn


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(
    config_path: str = "configs/config.yaml",
    batch_size: Optional[int] = None,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        config_path: Path to configuration file
        batch_size: Batch size (if None, uses config value)
        num_workers: Number of worker processes for data loading
    
    Returns:
        train_loader, val_loader, test_loader
    """
    config = load_config(config_path)
    
    # Get paths from config
    data_config = config['data']
    train_img_dir = data_config['train_images']
    train_lbl_dir = data_config['train_labels']
    val_img_dir = data_config['val_images']
    val_lbl_dir = data_config['val_labels']
    test_img_dir = data_config['test_images']
    test_lbl_dir = data_config['test_labels']
    
    # Get training config
    training_config = config['training']
    if batch_size is None:
        batch_size = training_config['batch_size']
    
    img_size = data_config['image_size']
    
    # Get augmentation config
    aug_config = config.get('augmentation', {}).get('train', {})
    
    # Create datasets
    train_dataset = AlgaeDataset(
        image_dir=train_img_dir,
        label_dir=train_lbl_dir,
        img_size=img_size,
        augment=True,
        augmentation_config=aug_config
    )
    
    val_dataset = AlgaeDataset(
        image_dir=val_img_dir,
        label_dir=val_lbl_dir,
        img_size=img_size,
        augment=False
    )
    
    test_dataset = AlgaeDataset(
        image_dir=test_img_dir,
        label_dir=test_lbl_dir,
        img_size=img_size,
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"\nDataLoader Summary:")
    print(f"  Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} images, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} images, {len(test_loader)} batches")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}x{img_size}")
    
    return train_loader, val_loader, test_loader


def get_class_names(config_path: str = "configs/config.yaml") -> list:
    """Get class names from config"""
    config = load_config(config_path)
    return config['data']['classes']


def get_num_classes(config_path: str = "configs/config.yaml") -> int:
    """Get number of classes from config"""
    return len(get_class_names(config_path))
