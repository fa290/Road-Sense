"""
PyTorch Dataset Classes with On-the-Fly Augmentation

This module provides PyTorch Dataset classes for loading KITTI data
with on-the-fly augmentation support for efficient training.
"""

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Callable, List

from .kitti_utils import load_kitti_image, load_kitti_labels, CLASS_ID_TO_NAME
from .augmentations import (
    get_training_augmentation_with_bbox,
    get_validation_augmentation_with_bbox,
    get_inference_augmentation
)


class KITTIDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        label_dir: str,
        transform: Optional[Callable] = None,
        mode: str = 'train',
        image_size: Optional[Tuple[int, int]] = None,
        augmentation_preset: str = 'medium',
        return_image_path: bool = False
    ):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.mode = mode
        self.image_size = image_size
        self.return_image_path = return_image_path
        
        # Get list of image files
        self.image_files = sorted(
            [f for f in self.img_dir.iterdir()
             if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
        )
        
        # Set up transforms
        if transform is not None:
            self.transform = transform
        else:
            if mode == 'train':
                from .augmentations import get_custom_augmentation
                self.transform = get_custom_augmentation(
                    preset=augmentation_preset,
                    image_size=image_size,
                    with_bbox=True
                )
            elif mode == 'val':
                self.transform = get_validation_augmentation_with_bbox(
                    image_size=image_size
                )
            else:  # test/inference
                self.transform = get_inference_augmentation(
                    image_size=image_size
                )
        
        print(f"Initialized KITTI Dataset:")
        print(f"  Mode: {mode}")
        print(f"  Images: {len(self.image_files)}")
        print(f"  Image size: {image_size if image_size else 'Original'}")
        print(f"  Augmentation: {'On' if mode == 'train' else 'Off'}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_files[idx]
        label_path = self.label_dir / f"{img_path.stem}.txt"
        
        # Load image
        image = load_kitti_image(str(img_path))
        img_height, img_width = image.shape[:2]
        
        # Load labels
        bboxes, class_labels, class_names = load_kitti_labels(
            str(label_path), img_width, img_height
        )
        
        # Handle case with no labels (for inference)
        if not bboxes:
            bboxes = []
            class_labels = []
        
        # Apply augmentation
        if self.mode in ['train', 'val'] and bboxes:
            augmented = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            image = augmented['image']
            bboxes = augmented['bboxes']
            class_labels = augmented['class_labels']
        elif self.mode == 'test' and self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # Prepare output
        sample = {
            'image': image,
            'bboxes': np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 4), dtype=np.float32),
            'labels': np.array(class_labels, dtype=np.int64) if class_labels else np.zeros((0,), dtype=np.int64)
        }
        
        if self.return_image_path:
            sample['image_path'] = str(img_path)
        
        return sample
    
    def get_class_distribution(self) -> dict:
        class_counts = {name: 0 for name in CLASS_ID_TO_NAME.values()}
        
        for img_file in self.image_files:
            label_path = self.label_dir / f"{img_file.stem}.txt"
            
            # Quick load to get dimensions (use first image as reference)
            if not hasattr(self, '_img_width'):
                img = cv2.imread(str(img_file))
                self._img_height, self._img_width = img.shape[:2]
            
            _, _, class_names = load_kitti_labels(
                str(label_path), self._img_width, self._img_height
            )
            
            for class_name in class_names:
                if class_name in class_counts:
                    class_counts[class_name] += 1
        
        return class_counts


class KITTIDatasetTorch(KITTIDataset):
    
    def __init__(
        self,
        img_dir: str,
        label_dir: str,
        transform: Optional[Callable] = None,
        mode: str = 'train',
        image_size: Optional[Tuple[int, int]] = None,
        augmentation_preset: str = 'medium',
        normalize: bool = True,
        return_image_path: bool = False
    ):
        super().__init__(
            img_dir=img_dir,
            label_dir=label_dir,
            transform=transform,
            mode=mode,
            image_size=image_size,
            augmentation_preset=augmentation_preset,
            return_image_path=return_image_path
        )
        self.normalize = normalize
    
    def __getitem__(self, idx: int) -> dict:
        sample = super().__getitem__(idx)
        
        # Convert image to tensor (H, W, C) -> (C, H, W)
        image = sample['image']
        if self.normalize:
            image = image.astype(np.float32) / 255.0
        
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        
        sample['image'] = image_tensor
        sample['bboxes'] = torch.from_numpy(sample['bboxes'])
        sample['labels'] = torch.from_numpy(sample['labels'])
        
        return sample


def collate_fn(batch: List[dict]) -> dict:

    images = torch.stack([item['image'] for item in batch])
    
    # Keep bboxes and labels as lists since they have variable lengths
    bboxes = [item['bboxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    batched = {
        'images': images,
        'bboxes': bboxes,
        'labels': labels
    }
    
    # Include image paths if present
    if 'image_path' in batch[0]:
        batched['image_paths'] = [item['image_path'] for item in batch]
    
    return batched


def create_data_loaders(
    train_img_dir: str,
    train_label_dir: str,
    val_img_dir: Optional[str] = None,
    val_label_dir: Optional[str] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: Optional[Tuple[int, int]] = None,
    augmentation_preset: str = 'medium'
) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
    # Create training dataset
    train_dataset = KITTIDatasetTorch(
        img_dir=train_img_dir,
        label_dir=train_label_dir,
        mode='train',
        image_size=image_size,
        augmentation_preset=augmentation_preset
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create validation dataset if specified
    val_loader = None
    if val_img_dir and val_label_dir:
        val_dataset = KITTIDatasetTorch(
            img_dir=val_img_dir,
            label_dir=val_label_dir,
            mode='val',
            image_size=image_size
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    return train_loader, val_loader

