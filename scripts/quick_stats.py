import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import get_dataset_statistics, print_dataset_statistics

stats = get_dataset_statistics(
    image_dir='data/raw/KITTI/training/image_2',
    label_dir='data/raw/KITTI/training/label_2',
    max_samples=100  # Fast! Change to None for all images
)

print_dataset_statistics(stats)
