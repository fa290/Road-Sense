# Data Module - Road-Sense Project

This module provides a complete toolkit for working with the KITTI dataset, including data loading, augmentation, and batch processing.

## Module Structure

```
src/data/
├── __init__.py              # Module exports
├── augmentations.py         # Augmentation pipelines
├── kitti_utils.py          # KITTI data loading utilities
├── kitti_dataset.py        # PyTorch Dataset classes
├── augment_dataset.py      # Batch augmentation script
└── README.md               # This file
```

## Quick Start

### 1. On-the-Fly Augmentation (our plan for now)

```python
from src.data import create_data_loaders

# Create DataLoaders with on-the-fly augmentation
train_loader, val_loader = create_data_loaders(
    train_img_dir='data/raw/KITTI/training/image_2',
    train_label_dir='data/raw/KITTI/training/label_2',
    val_img_dir='data/raw/KITTI/validation/image_2',
    val_label_dir='data/raw/KITTI/validation/label_2',
    batch_size=8,
    image_size=(384, 1280),
    augmentation_preset='medium'
)

# Use in training loop
for batch in train_loader:
    images = batch['images']  # Shape: (B, 3, H, W)
    bboxes = batch['bboxes']  # List of tensors
    labels = batch['labels']  # List of tensors
    # ... training code ...
```

### 2. Batch Augmentation (Offline)

#### Using Python API:

```python
from src.data import augment_dataset

stats = augment_dataset(
    img_dir='data/raw/KITTI/training/image_2',
    label_dir='data/raw/KITTI/training/label_2',
    output_img_dir='data/augmented/images',
    output_label_dir='data/augmented/labels',
    num_images=100,
    augmentations_per_image=3,
    preset='medium'
)

print(f"Successfully augmented: {stats['successful']} images")
```

#### Using Command Line:

```bash
# From project root
cd src/data

# Augment dataset
python augment_dataset.py \
    --img-dir ../../data/raw/KITTI/training/image_2 \
    --label-dir ../../data/raw/KITTI/training/label_2 \
    --output-img-dir ../../data/augmented/images \
    --output-label-dir ../../data/augmented/labels \
    --num-images 100 \
    --augmentations-per-image 3 \
    --preset medium
```

### 3. Custom Augmentation Pipeline

```python
from src.data import get_custom_augmentation
from src.data import load_kitti_image, load_kitti_labels

# Get augmentation pipeline
transform = get_custom_augmentation(
    preset='heavy',          # 'light', 'medium', or 'heavy'
    image_size=(384, 1280),  # Target size or None
    with_bbox=True,          # Include bbox support
    min_visibility=0.3       # Min bbox visibility
)

# Load and augment
image = load_kitti_image('path/to/image.png')
bboxes, labels, names = load_kitti_labels('path/to/label.txt', image.shape[1], image.shape[0])

augmented = transform(image=image, bboxes=bboxes, class_labels=labels)
aug_image = augmented['image']
aug_bboxes = augmented['bboxes']
aug_labels = augmented['class_labels']
```

## Dataset Statistics

```python
from src.data import get_dataset_statistics, print_dataset_statistics

stats = get_dataset_statistics(
    image_dir='data/raw/KITTI/training/image_2',
    label_dir='data/raw/KITTI/training/label_2',
    max_samples=None  # Analyze all images
)

print_dataset_statistics(stats)
```
or 
```
python3 scripts/quick_stats.py
```

Output:
```
==================================================
DATASET STATISTICS
==================================================
Total images: 7481
Total objects: 40570
Average objects per image: 5.42
Images with no labels: 0

Image dimensions:
  Width range: 1224 - 1242
  Height range: 370 - 376

Class distribution:
  Car                 : 28742 (70.85%)
  Pedestrian          :  4487 (11.06%)
  Van                 :  2914 ( 7.18%)
  Cyclist             :  1627 ( 4.01%)
  Truck               :  1094 ( 2.70%)
  Misc                :   973 ( 2.40%)
  Tram                :   511 ( 1.26%)
  Person_sitting      :   222 ( 0.55%)
==================================================
```

### Augmentation Techniques Included:

- **Geometric**: Horizontal flip, shift, scale, rotate
- **Color**: Brightness, contrast, gamma, hue, saturation
- **Noise**: Motion blur, Gaussian noise, Gaussian blur
- **Weather**: Rain, fog, sun flare

##  Visualization

```python
from src.data import load_kitti_image, load_kitti_labels, visualize_bboxes

# Load data
image = load_kitti_image('path/to/image.png')
bboxes, labels, names = load_kitti_labels('path/to/label.txt', image.shape[1], image.shape[0])

# Visualize
visualize_bboxes(
    image=image,
    bboxes=bboxes,
    class_names=names,
    title="KITTI Image with Bounding Boxes"
)
```

## KITTI Data Format

### Input Format (KITTI):
```
# label.txt
Car 0.00 0 -1.58 587.01 173.33 614.12 200.12 1.65 1.67 3.64 -0.65 1.71 46.70 -1.59
Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01
```

Fields:
- Class name
- Truncated, Occluded, Alpha
- **2D bbox**: left, top, right, bottom (pixels)
- 3D dimensions and location
- Rotation

### Output Format (YOLO):
```
# label.txt
0 0.485000 0.415000 0.021000 0.069000
3 0.612500 0.600000 0.079000 0.439000
```

Format: `class_id x_center y_center width height` (all normalized 0-1)

## API Reference

### Augmentation Functions

| Function | Description |
|----------|-------------|
| `get_training_augmentation()` | Training augmentation without bboxes |
| `get_training_augmentation_with_bbox()` | Training augmentation with bbox support |
| `get_validation_augmentation()` | Validation (minimal augmentation) |
| `get_validation_augmentation_with_bbox()` | Validation with bbox support |
| `get_inference_augmentation()` | Inference preprocessing only |
| `get_custom_augmentation()` | Custom preset-based augmentation |

### KITTI Utilities

| Function | Description |
|----------|-------------|
| `load_kitti_image()` | Load image and convert to RGB |
| `load_kitti_labels()` | Load KITTI labels, convert to YOLO |
| `save_yolo_labels()` | Save labels in YOLO format |
| `load_yolo_labels()` | Load labels in YOLO format |
| `yolo_to_pixel()` | Convert YOLO to pixel coordinates |
| `visualize_bboxes()` | Visualize bboxes on image |
| `get_dataset_statistics()` | Calculate dataset statistics |

### Dataset Classes

| Class | Description |
|-------|-------------|
| `KITTIDataset` | Basic dataset with NumPy output |
| `KITTIDatasetTorch` | PyTorch dataset with tensor output |
| `collate_fn()` | Custom collate for variable-length bboxes |
| `create_data_loaders()` | Create train/val DataLoaders |


## Testing

Test individual modules:

```bash
# Test augmentations
python src/data/augmentations.py

# Test KITTI utilities
python src/data/kitti_utils.py

# Test dataset classes
python src/data/kitti_dataset.py
```

## Example Notebook

See `notebooks/data_augmentation.ipynb` for a complete interactive tutorial with:
- Visualization of augmentation effects
- Bounding box augmentation examples
- Batch processing demonstration
- Output verification

## Requirements

- Python >= 3.7
- OpenCV (cv2)
- NumPy
- Albumentations
- Matplotlib
- PyTorch (for dataset classes)
- tqdm (for progress bars)

Install dependencies:
```bash
pip install opencv-python numpy albumentations matplotlib torch tqdm
```
or 
```bash
pip install -r requirements.txt
```