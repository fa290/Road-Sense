# Dataset Exploration Report

## 1. Dataset Overview

For this project, we use the **KITTI Vision Benchmark Dataset**, which is a widely used dataset for autonomous driving research.

The dataset contains real-world driving scenes captured using sensors mounted on a vehicle. These include RGB images and annotations for object detection tasks such as cars, pedestrians, and cyclists.

Dataset Source: KITTI Vision Benchmark Suite

The dataset is used to train computer vision models that help autonomous vehicles understand their environment and detect objects on the road.

---

## 2. Dataset Statistics

Total Images: ~7481
Annotation Files: ~7481

Classes used in this project:

* Car
* Pedestrian
* Cyclist

Image Resolution (original):
Varies across images

Image Resolution (after preprocessing):
640 × 640

Dataset Split:

Train Set: 70%
Validation Set: 20%
Test Set: 10%

---

## 3. Dataset Exploration

During the exploration phase we inspected:

* Image distribution
* Annotation structure
* Object classes
* Bounding box coordinates

Annotations in the KITTI dataset follow this format:

Object Type | Truncation | Occlusion | Alpha | Bbox Coordinates | Dimensions | Location | Rotation

For the purposes of this project we only used:

Object class and bounding box coordinates.

---

## 4. Preprocessing Steps

The following preprocessing steps were applied to prepare the dataset for model training:

1. Image Resizing
   All images were resized to 640 × 640 pixels to match the model input size.

2. Annotation Conversion
   Original KITTI annotations were converted to **YOLO format**.

YOLO format structure:

class_id x_center y_center width height

All coordinates were normalized relative to image width and height.

3. Dataset Splitting
   The dataset was split into:

Train set (70%)
Validation set (20%)
Test set (10%)

4. Folder Organization
   Images and labels were organized into a structured dataset format compatible with YOLO training pipelines.

---

## 5. Data Augmentation

At this stage, basic preprocessing was applied. Advanced augmentation will be applied during training using the model training pipeline.

Possible augmentations include:

* Horizontal flipping
* Random scaling
* Brightness adjustment
* Rotation

These augmentations help improve model robustness.

---

## 6. Sample Dataset Structure

The processed dataset follows the following structure:

data/
├── processed_dataset
│   ├── images
│   │   ├── train
│   │   ├── val
│   │   └── test
│   │
│   └── labels
│       ├── train
│       ├── val
│       └── test

Each image has a corresponding label file with the same name.

Example:

000123.png
000123.txt

---

## 7. Reproducibility

The dataset preparation process is fully reproducible using the provided Python preprocessing script.

The script performs:

* Image resizing
* Annotation conversion
* Dataset splitting
* Folder structuring

Running the script again will reproduce the exact dataset preparation pipeline.
