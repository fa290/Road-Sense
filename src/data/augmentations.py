"""
Data Augmentation Pipelines for Road-Sense Project

This module provides augmentation pipelines using Albumentations library
for training and validation phases. Includes support for bounding boxes.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_training_augmentation(image_size=None):
    transforms = []
    
    # Resize if specified
    if image_size:
        transforms.append(A.Resize(height=image_size[0], width=image_size[1]))
    
    # Geometric transformations
    transforms.extend([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),
    ])
    
    # Color augmentations
    transforms.extend([
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=0.3
        ),
    ])
    
    # Weather and lighting effects
    transforms.extend([
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussNoise(var_limit=(10.0, 30.0), p=1.0),
            A.GaussianBlur(blur_limit=3, p=1.0),
        ], p=0.3),
        
        A.OneOf([
            A.RandomRain(
                slant_lower=-10,
                slant_upper=10,
                drop_length=20,
                drop_width=1,
                p=1.0
            ),
            A.RandomFog(
                fog_coef_lower=0.3,
                fog_coef_upper=0.5,
                alpha_coef=0.08,
                p=1.0
            ),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                angle_lower=0.5,
                src_radius=100,
                p=1.0
            ),
        ], p=0.2),
    ])
    
    return A.Compose(transforms)


def get_training_augmentation_with_bbox(image_size=None, min_visibility=0.3):
    transforms = []
    
    # Resize if specified
    if image_size:
        transforms.append(A.Resize(height=image_size[0], width=image_size[1]))
    
    # Geometric transformations (safe for bounding boxes)
    transforms.extend([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=5,  # Smaller rotation to preserve more boxes
            border_mode=cv2.BORDER_CONSTANT,
            p=0.4
        ),
    ])
    
    # Color augmentations (don't affect bounding boxes)
    transforms.extend([
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=0.3
        ),
    ])
    
    # Weather and lighting effects
    transforms.extend([
        A.OneOf([
            A.MotionBlur(blur_limit=5, p=1.0),
            A.GaussNoise(var_limit=(10.0, 30.0), p=1.0),
        ], p=0.3),
        
        A.RandomSunFlare(
            flare_roi=(0, 0, 1, 0.5),
            angle_lower=0.5,
            src_radius=100,
            p=0.2
        ),
    ])
    
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=min_visibility
        )
    )


def get_validation_augmentation(image_size=None):
    transforms = []
    
    # Only resize for validation, no other augmentations
    if image_size:
        transforms.append(A.Resize(height=image_size[0], width=image_size[1]))
    
    return A.Compose(transforms)


def get_validation_augmentation_with_bbox(image_size=None):
    transforms = []
    
    # Only resize for validation
    if image_size:
        transforms.append(A.Resize(height=image_size[0], width=image_size[1]))
    
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        )
    )


def get_inference_augmentation(image_size=None):
    transforms = []
    
    if image_size:
        transforms.append(A.Resize(height=image_size[0], width=image_size[1]))
    
    return A.Compose(transforms)


# Preset configurations
AUGMENTATION_PRESETS = {
    'light': {
        'horizontal_flip_p': 0.3,
        'shift_scale_rotate_p': 0.2,
        'brightness_contrast_p': 0.3,
        'weather_p': 0.1,
    },
    'medium': {
        'horizontal_flip_p': 0.5,
        'shift_scale_rotate_p': 0.4,
        'brightness_contrast_p': 0.5,
        'weather_p': 0.2,
    },
    'heavy': {
        'horizontal_flip_p': 0.7,
        'shift_scale_rotate_p': 0.6,
        'brightness_contrast_p': 0.7,
        'weather_p': 0.4,
    }
}


def get_custom_augmentation(preset='medium', image_size=None, with_bbox=False, min_visibility=0.3):
    if preset not in AUGMENTATION_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(AUGMENTATION_PRESETS.keys())}")
    
    params = AUGMENTATION_PRESETS[preset]
    transforms = []
    
    if image_size:
        transforms.append(A.Resize(height=image_size[0], width=image_size[1]))
    
    transforms.extend([
        A.HorizontalFlip(p=params['horizontal_flip_p']),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=10 if not with_bbox else 5,
            p=params['shift_scale_rotate_p']
        ),
        A.RandomBrightnessContrast(p=params['brightness_contrast_p']),
        A.OneOf([
            A.MotionBlur(p=1.0),
            A.GaussNoise(p=1.0),
        ], p=0.3),
        A.RandomSunFlare(p=params['weather_p']),
    ])
    
    if with_bbox:
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=min_visibility
            )
        )
    else:
        return A.Compose(transforms)
