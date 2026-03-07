# Road-Sense
Real-Time Object Detection for Autonomous Vehicles

# Road-Sense: Autonomous Driving Object Detection

Road-Sense is a computer vision project developed as part of the **Digital Egypt Initiative**. The goal is to build an object detection system for autonomous vehicles using the KITTI dataset.

The system detects important road objects such as cars, pedestrians, and cyclists.

---

# Dataset

This project uses the **KITTI Vision Benchmark Dataset**, which is widely used in autonomous driving research.

Dataset Source:
http://www.cvlibs.net/datasets/kitti/

The dataset contains real-world driving images captured from a moving vehicle with labeled objects.

---

# Classes Used

* Car
* Pedestrian
* Cyclist

---

# Folder Structure

The processed dataset follows this structure:

data/

processed_dataset/

images/
train/
val/
test/

labels/
train/
val/
test/

Each image has a corresponding label file with the same name.

Example:

000001.png
000001.txt

---

# Dataset Preparation

To prepare the dataset, run the preprocessing script:

python prepare_dataset.py

This script performs:

* Image resizing to 640x640
* Conversion of KITTI annotations to YOLO format
* Splitting dataset into Train / Validation / Test
* Organizing files into structured folders

---

# Setup Instructions

1. Clone the repository

git clone https://github.com/your-username/Road-Sense.git

2. Navigate to the project directory

cd Road-Sense

3. Install dependencies

pip install -r requirements.txt

4. Run dataset preparation

python prepare_dataset.py

---

# Project Goal

The main objective of this project is to build a reliable object detection system that can assist autonomous vehicles in understanding road environments.
