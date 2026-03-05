"""
Batch Augmentation Script for KITTI Dataset

This script performs batch augmentation on KITTI dataset images with bounding boxes.
Can be used for on-the-fly augmentation.
"""

import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict

from .augmentations import get_training_augmentation_with_bbox, get_custom_augmentation
from .kitti_utils import load_kitti_image, load_kitti_labels, save_yolo_labels


def augment_dataset(
    img_dir: str,
    label_dir: str,
    output_img_dir: str,
    output_label_dir: str,
    num_images: Optional[int] = None,
    augmentations_per_image: int = 1,
    preset: str = 'medium',
    min_visibility: float = 0.3,
    image_size: Optional[tuple] = None
) -> Dict[str, int]:
    
    # Create output directories
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # Get list of images
    img_dir_path = Path(img_dir)
    all_images = sorted(
        [f for f in img_dir_path.iterdir() 
         if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    )
    
    # Limit number of images if specified
    if num_images:
        all_images = all_images[:num_images]
    
    # Get augmentation pipeline
    transform = get_custom_augmentation(
        preset=preset,
        image_size=image_size,
        with_bbox=True,
        min_visibility=min_visibility
    )
    
    # Statistics
    stats = {
        'total': len(all_images),
        'successful': 0,
        'skipped': 0,
        'failed': 0
    }
    
    print(f"Starting augmentation of {len(all_images)} images...")
    print(f"Augmentations per image: {augmentations_per_image}")
    print(f"Preset: {preset}")
    print(f"Output directories:")
    print(f"  Images: {output_img_dir}")
    print(f"  Labels: {output_label_dir}")
    print()
    
    # Process each image
    for img_path in tqdm(all_images, desc="Augmenting images"):
        base_name = img_path.stem
        label_path = Path(label_dir) / f"{base_name}.txt"
        
        try:
            # Load image
            image = load_kitti_image(str(img_path))
            img_height, img_width = image.shape[:2]
            
            # Load labels
            bboxes, class_labels, class_names = load_kitti_labels(
                str(label_path), img_width, img_height
            )
            
            # Skip if no valid labels
            if not bboxes:
                stats['skipped'] += 1
                continue
            
            # Generate augmented versions
            for aug_idx in range(augmentations_per_image):
                try:
                    # Apply augmentation
                    augmented = transform(
                        image=image,
                        bboxes=bboxes,
                        class_labels=class_labels
                    )
                    
                    aug_img = augmented['image']
                    aug_boxes = augmented['bboxes']
                    aug_labels = augmented['class_labels']
                    
                    # Skip if all boxes were removed
                    if not aug_boxes:
                        continue
                    
                    # Save augmented image
                    out_name = f"aug_{aug_idx}_{img_path.name}"
                    out_img_path = Path(output_img_dir) / out_name
                    cv2.imwrite(
                        str(out_img_path),
                        cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                    )
                    
                    # Save augmented labels
                    out_label_name = f"aug_{aug_idx}_{base_name}.txt"
                    out_label_path = Path(output_label_dir) / out_label_name
                    save_yolo_labels(str(out_label_path), aug_boxes, aug_labels)
                    
                except Exception as e:
                    print(f"\nWarning: Failed augmentation {aug_idx} for {img_path.name}: {e}")
                    continue
            
            stats['successful'] += 1
            
        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
            stats['failed'] += 1
    
    return stats


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Augment KITTI dataset images with bounding boxes'
    )
    
    parser.add_argument(
        '--img-dir',
        type=str,
        required=True,
        help='Input image directory'
    )
    parser.add_argument(
        '--label-dir',
        type=str,
        required=True,
        help='Input label directory'
    )
    parser.add_argument(
        '--output-img-dir',
        type=str,
        required=True,
        help='Output directory for augmented images'
    )
    parser.add_argument(
        '--output-label-dir',
        type=str,
        required=True,
        help='Output directory for augmented labels'
    )
    parser.add_argument(
        '--num-images',
        type=int,
        default=None,
        help='Number of images to process (default: all)'
    )
    parser.add_argument(
        '--augmentations-per-image',
        type=int,
        default=3,
        help='Number of augmented versions per image (default: 3)'
    )
    parser.add_argument(
        '--preset',
        type=str,
        choices=['light', 'medium', 'heavy'],
        default='medium',
        help='Augmentation intensity preset (default: medium)'
    )
    parser.add_argument(
        '--min-visibility',
        type=float,
        default=0.3,
        help='Minimum bbox visibility after augmentation (default: 0.3)'
    )
    parser.add_argument(
        '--image-size',
        type=str,
        default=None,
        help='Target image size as "height,width" (default: None, keeps original)'
    )
    
    args = parser.parse_args()
    
    # Parse image size
    image_size = None
    if args.image_size:
        try:
            h, w = map(int, args.image_size.split(','))
            image_size = (h, w)
        except ValueError:
            print(f"Error: Invalid image size format. Use 'height,width'")
            return
    
    # Run augmentation
    stats = augment_dataset(
        img_dir=args.img_dir,
        label_dir=args.label_dir,
        output_img_dir=args.output_img_dir,
        output_label_dir=args.output_label_dir,
        num_images=args.num_images,
        augmentations_per_image=args.augmentations_per_image,
        preset=args.preset,
        min_visibility=args.min_visibility,
        image_size=image_size
    )
    
    # Print summary
    print("\n" + "=" * 50)
    print("AUGMENTATION SUMMARY")
    print("=" * 50)
    print(f"Total images processed: {stats['total']}")
    print(f"Successfully augmented: {stats['successful']}")
    print(f"Skipped (no labels): {stats['skipped']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success rate: {(stats['successful']/stats['total']*100):.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
