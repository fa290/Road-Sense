"""
KITTI Dataset Utilities for Road-Sense Project

This module provides utilities for loading, processing, and visualizing
KITTI dataset images and labels.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Tuple, List, Optional


# KITTI class mapping
KITTI_CLASSES = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
    'Pedestrian': 3,
    'Person_sitting': 4,
    'Cyclist': 5,
    'Tram': 6,
    'Misc': 7,
    'DontCare': 8
}

CLASS_ID_TO_NAME = {v: k for k, v in KITTI_CLASSES.items() if k != 'DontCare'}

# Colors for visualization (BGR format for OpenCV)
CLASS_COLORS = {
    'Car': (0, 255, 0),           # Green
    'Van': (0, 255, 255),          # Yellow
    'Truck': (255, 165, 0),        # Orange
    'Pedestrian': (255, 0, 0),     # Blue
    'Person_sitting': (255, 0, 255), # Magenta
    'Cyclist': (0, 165, 255),      # Orange-Red
    'Tram': (128, 0, 128),         # Purple
    'Misc': (128, 128, 128),       # Gray
}


def load_kitti_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_kitti_labels(
    label_path: str,
    img_width: int,
    img_height: int,
    skip_dontcare: bool = True
) -> Tuple[List[List[float]], List[int], List[str]]:
    """
    Load KITTI format labels and convert to YOLO format (normalized).
    Returns:
        Tuple containing:
            - List of bboxes in YOLO format [[x_center, y_center, width, height], ...]
            - List of class IDs
            - List of class names
    """
    bboxes = []
    class_labels = []
    class_names = []
    
    if not Path(label_path).exists():
        return bboxes, class_labels, class_names
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:  # KITTI format has 15 fields
                continue
            
            class_name = parts[0]
            
            # Skip DontCare objects if requested
            if skip_dontcare and class_name == 'DontCare':
                continue
            
            # Get bbox coordinates (pixels)
            try:
                left, top, right, bottom = map(float, parts[4:8])
            except ValueError:
                continue
            
            # Convert to YOLO format (normalized)
            x_center = ((left + right) / 2) / img_width
            y_center = ((top + bottom) / 2) / img_height
            width = (right - left) / img_width
            height = (bottom - top) / img_height
            
            # Ensure values are within [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            # Skip invalid boxes
            if width <= 0 or height <= 0:
                continue
            
            bboxes.append([x_center, y_center, width, height])
            class_labels.append(KITTI_CLASSES.get(class_name, 7))  # Default to 'Misc'
            class_names.append(class_name)
    
    return bboxes, class_labels, class_names


def save_yolo_labels(
    label_path: str,
    bboxes: List[List[float]],
    class_labels: List[int]
) -> None:
    with open(label_path, 'w') as f:
        for cls, box in zip(class_labels, bboxes):
            # Format to 6 decimal places for precision
            box_str = " ".join([f"{val:.6f}" for val in box])
            f.write(f"{int(cls)} {box_str}\n")


def load_yolo_labels(label_path: str) -> Tuple[List[List[float]], List[int]]:
    bboxes = []
    class_labels = []
    
    if not Path(label_path).exists():
        return bboxes, class_labels
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            try:
                class_id = int(parts[0])
                bbox = [float(x) for x in parts[1:5]]
                
                class_labels.append(class_id)
                bboxes.append(bbox)
            except ValueError:
                continue
    
    return bboxes, class_labels


def yolo_to_pixel(
    bboxes: List[List[float]],
    img_width: int,
    img_height: int
) -> List[List[int]]:
    pixel_bboxes = []
    
    for bbox in bboxes:
        x_center, y_center, width, height = bbox
        
        x_center_px = x_center * img_width
        y_center_px = y_center * img_height
        width_px = width * img_width
        height_px = height * img_height
        
        x_min = int(x_center_px - width_px / 2)
        y_min = int(y_center_px - height_px / 2)
        x_max = int(x_center_px + width_px / 2)
        y_max = int(y_center_px + height_px / 2)
        
        pixel_bboxes.append([x_min, y_min, x_max, y_max])
    
    return pixel_bboxes


def visualize_bboxes(
    image: np.ndarray,
    bboxes: List[List[float]],
    class_names: List[str],
    title: str = "Image with Bounding Boxes",
    show: bool = True,
    figsize: Tuple[int, int] = (12, 8)
) -> np.ndarray:
    
    img_height, img_width = image.shape[:2]
    image_copy = image.copy()
    
    # Convert YOLO to pixel coordinates
    pixel_bboxes = yolo_to_pixel(bboxes, img_width, img_height)
    
    # Draw bounding boxes
    for bbox, class_name in zip(pixel_bboxes, class_names):
        x_min, y_min, x_max, y_max = bbox
        color = CLASS_COLORS.get(class_name, (255, 255, 255))
        
        # Draw rectangle
        cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Draw label background
        label = class_name
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            image_copy,
            (x_min, y_min - text_height - 4),
            (x_min + text_width, y_min),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            image_copy,
            label,
            (x_min, y_min - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
    
    if show:
        plt.figure(figsize=figsize)
        plt.imshow(image_copy)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return image_copy


def get_dataset_statistics(
    image_dir: str,
    label_dir: str,
    max_samples: Optional[int] = None
) -> dict:
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    
    image_files = sorted(list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg')))
    
    if max_samples:
        image_files = image_files[:max_samples]
    
    stats = {
        'total_images': len(image_files),
        'total_objects': 0,
        'class_counts': {name: 0 for name in KITTI_CLASSES if name != 'DontCare'},
        'images_with_no_labels': 0,
        'avg_objects_per_image': 0,
        'image_sizes': []
    }
    
    for img_path in image_files:
        # Load image to get size
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            stats['image_sizes'].append((w, h))
        
        # Load labels
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            stats['images_with_no_labels'] += 1
            continue
        
        img_width, img_height = w, h
        _, _, class_names = load_kitti_labels(str(label_path), img_width, img_height)
        
        if not class_names:
            stats['images_with_no_labels'] += 1
            continue
        
        stats['total_objects'] += len(class_names)
        for class_name in class_names:
            if class_name in stats['class_counts']:
                stats['class_counts'][class_name] += 1
    
    if stats['total_images'] > 0:
        stats['avg_objects_per_image'] = stats['total_objects'] / stats['total_images']
    
    return stats


def print_dataset_statistics(stats: dict) -> None:
    print("=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)
    print(f"Total images: {stats['total_images']}")
    print(f"Total objects: {stats['total_objects']}")
    print(f"Average objects per image: {stats['avg_objects_per_image']:.2f}")
    print(f"Images with no labels: {stats['images_with_no_labels']}")
    
    if stats['image_sizes']:
        widths, heights = zip(*stats['image_sizes'])
        print(f"\nImage dimensions:")
        print(f"  Width range: {min(widths)} - {max(widths)}")
        print(f"  Height range: {min(heights)} - {max(heights)}")
    
    print("\nClass distribution:")
    for class_name, count in sorted(stats['class_counts'].items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            percentage = (count / stats['total_objects']) * 100 if stats['total_objects'] > 0 else 0
            print(f"  {class_name:20s}: {count:5d} ({percentage:5.2f}%)")
    print("=" * 50)

