"""
Custom Dataset class for Algae Microscopy Detection (YOLO format)
"""
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class AlgaeDataset(Dataset):
    """
    PyTorch Dataset for algae microscopy images with YOLO format annotations.
    
    YOLO format: class_id x_center y_center width height (all normalized 0-1)
    """
    
    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        img_size: int = 640,
        augment: bool = False,
        augmentation_config: Optional[Dict] = None
    ):
        """
        Args:
            image_dir: Directory containing images
            label_dir: Directory containing YOLO format labels (.txt files)
            img_size: Target image size for resizing
            augment: Whether to apply augmentations
            augmentation_config: Dict with augmentation parameters
        """
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.augment = augment
        
        # Get all image paths
        self.image_paths = sorted(list(self.image_dir.glob("*.jpg")))
        
        # Verify corresponding labels exist
        self.valid_samples = []
        for img_path in self.image_paths:
            label_path = self.label_dir / (img_path.stem + ".txt")
            if label_path.exists():
                self.valid_samples.append((img_path, label_path))
        
        print(f"Found {len(self.valid_samples)} valid image-label pairs")
        
        # Setup augmentations
        if augment and augmentation_config:
            self.transform = self._get_augmentations(augmentation_config)
        else:
            self.transform = self._get_base_transform()
    
    def __len__(self) -> int:
        return len(self.valid_samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Returns:
            image: Tensor of shape (3, H, W)
            targets: Tensor of shape (N, 6) where N is number of objects
                     Format: [batch_idx, class_id, x_center, y_center, width, height]
            metadata: Dict with image path and original size
        """
        img_path, label_path = self.valid_samples[idx]
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        # Load labels
        boxes, class_ids = self._parse_yolo_label(label_path)
        
        # Apply transformations
        if len(boxes) > 0:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                class_labels=class_ids
            )
            image = transformed['image']
            boxes = np.array(transformed['bboxes']) if len(transformed['bboxes']) > 0 else np.zeros((0, 4))
            class_ids = np.array(transformed['class_labels']) if len(transformed['class_labels']) > 0 else np.zeros((0,))
        else:
            # No objects in image
            transformed = self.transform(image=image)
            image = transformed['image']
            boxes = np.zeros((0, 4))
            class_ids = np.zeros((0,))
        
        # Convert to YOLO format targets
        targets = self._create_targets(boxes, class_ids, idx)
        
        metadata = {
            'path': str(img_path),
            'orig_size': (orig_h, orig_w),
            'img_size': self.img_size
        }
        
        return image, targets, metadata
    
    def _parse_yolo_label(self, label_path: Path) -> Tuple[List, List]:
        """
        Parse YOLO format label file.
        
        Returns:
            boxes: List of [x_center, y_center, width, height] (normalized)
            class_ids: List of class IDs
        """
        boxes = []
        class_ids = []
        
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                boxes.append([x_center, y_center, width, height])
                class_ids.append(class_id)
        
        return boxes, class_ids
    
    def _create_targets(
        self, 
        boxes: np.ndarray, 
        class_ids: np.ndarray, 
        batch_idx: int
    ) -> torch.Tensor:
        """
        Create target tensor in format: [batch_idx, class_id, x, y, w, h]
        """
        if len(boxes) == 0:
            return torch.zeros((0, 6))
        
        targets = np.zeros((len(boxes), 6))
        targets[:, 0] = batch_idx  # Batch index
        targets[:, 1] = class_ids  # Class IDs
        targets[:, 2:] = boxes     # Bounding boxes
        
        return torch.from_numpy(targets).float()
    
    def _get_base_transform(self):
        """Base transformation without augmentation"""
        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
    
    def _get_augmentations(self, config: Dict):
        """
        Create augmentation pipeline from config.
        
        Args:
            config: Dict with keys like 'horizontal_flip', 'rotation', etc.
        """
        transforms = [
            A.Resize(self.img_size, self.img_size),
        ]
        
        # Add augmentations based on config
        if config.get('horizontal_flip', 0) > 0:
            transforms.append(A.HorizontalFlip(p=config['horizontal_flip']))
        
        if config.get('vertical_flip', 0) > 0:
            transforms.append(A.VerticalFlip(p=config['vertical_flip']))
        
        if config.get('rotation', 0) > 0:
            # Rotation in 90° increments for microscopy
            transforms.append(A.RandomRotate90(p=0.5))
        
        if config.get('brightness', 0) > 0:
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=config['brightness'],
                contrast_limit=config.get('contrast', 0.2),
                p=0.5
            ))
        
        if config.get('gaussian_noise', 0) > 0:
            transforms.append(A.GaussNoise(
                var_limit=(0, config['gaussian_noise'] * 255),
                p=0.3
            ))
        
        # Always normalize and convert to tensor
        transforms.extend([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.3  # Remove boxes with <30% visibility after augmentation
            )
        )


def collate_fn(batch):
    """
    Custom collate function for DataLoader to handle variable number of objects.
    
    Args:
        batch: List of (image, targets, metadata) tuples
    
    Returns:
        images: Tensor of shape (B, 3, H, W)
        targets: Tensor of shape (total_objects, 6)
        metadata: List of metadata dicts
    """
    images, targets, metadata = zip(*batch)
    
    # Stack images
    images = torch.stack(images, 0)
    
    # Concatenate targets and update batch indices
    for i, target in enumerate(targets):
        target[:, 0] = i  # Update batch index
    targets = torch.cat(targets, 0)
    
    return images, targets, list(metadata)
