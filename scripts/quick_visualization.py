import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import load_kitti_image, load_kitti_labels, visualize_bboxes

# Load data
image = load_kitti_image('data/raw/KITTI/training/image_2/000000.png')
bboxes, labels, names = load_kitti_labels('data/raw/KITTI/training/label_2/000000.txt', image.shape[1], image.shape[0])

# Visualize
visualize_bboxes(
    image=image,
    bboxes=bboxes,
    class_names=names,
    title="KITTI Image with Bounding Boxes"
)