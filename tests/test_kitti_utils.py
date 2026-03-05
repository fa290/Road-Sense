"""
Unit tests for KITTI utilities.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.kitti_utils import (
    KITTI_CLASSES,
    CLASS_ID_TO_NAME,
    yolo_to_pixel,
    save_yolo_labels,
    load_yolo_labels
)


class TestKITTIClasses:
    def test_kitti_classes_exist(self):
        assert isinstance(KITTI_CLASSES, dict)
        assert len(KITTI_CLASSES) > 0
    
    def test_common_classes_present(self):
        required_classes = ['Car', 'Pedestrian', 'Cyclist', 'Van', 'Truck']
        for cls in required_classes:
            assert cls in KITTI_CLASSES
    
    def test_class_id_mapping(self):
        for class_name, class_id in KITTI_CLASSES.items():
            if class_name != 'DontCare':
                assert class_id in CLASS_ID_TO_NAME
                assert CLASS_ID_TO_NAME[class_id] == class_name


class TestYoloToPixel:
    
    def test_yolo_to_pixel_conversion(self):
        bboxes_yolo = [[0.5, 0.5, 0.4, 0.4]]  # Center bbox
        img_width, img_height = 1000, 800
        
        pixel_bboxes = yolo_to_pixel(bboxes_yolo, img_width, img_height)
        
        assert len(pixel_bboxes) == 1
        x_min, y_min, x_max, y_max = pixel_bboxes[0]
        
        # Check bbox is centered
        assert x_min < img_width / 2 < x_max
        assert y_min < img_height / 2 < y_max
    
    def test_yolo_to_pixel_multiple_boxes(self):
        bboxes_yolo = [
            [0.25, 0.25, 0.2, 0.2],
            [0.75, 0.75, 0.2, 0.2]
        ]
        img_width, img_height = 1000, 800
        
        pixel_bboxes = yolo_to_pixel(bboxes_yolo, img_width, img_height)
        
        assert len(pixel_bboxes) == 2
        for bbox in pixel_bboxes:
            assert len(bbox) == 4
            x_min, y_min, x_max, y_max = bbox
            assert 0 <= x_min < x_max <= img_width
            assert 0 <= y_min < y_max <= img_height
    
    def test_yolo_to_pixel_edge_cases(self):
        bboxes_yolo = [
            [0.05, 0.05, 0.1, 0.1],  # Top-left corner
            [0.95, 0.95, 0.1, 0.1]   # Bottom-right corner
        ]
        img_width, img_height = 1000, 800
        
        pixel_bboxes = yolo_to_pixel(bboxes_yolo, img_width, img_height)
        
        # Check all boxes are within image bounds
        for bbox in pixel_bboxes:
            x_min, y_min, x_max, y_max = bbox
            assert x_min >= 0 and x_max <= img_width
            assert y_min >= 0 and y_max <= img_height


class TestYoloLabelIO:
    
    def test_save_and_load_yolo_labels(self, tmp_path):
        # Create test data
        bboxes = [
            [0.5, 0.5, 0.2, 0.3],
            [0.3, 0.4, 0.15, 0.25]
        ]
        labels = [0, 1]
        
        # Save labels
        label_file = tmp_path / "test.txt"
        save_yolo_labels(str(label_file), bboxes, labels)
        
        # Load labels
        loaded_bboxes, loaded_labels = load_yolo_labels(str(label_file))
        
        # Verify
        assert len(loaded_bboxes) == len(bboxes)
        assert len(loaded_labels) == len(labels)
        
        for orig_bbox, loaded_bbox in zip(bboxes, loaded_bboxes):
            np.testing.assert_array_almost_equal(orig_bbox, loaded_bbox, decimal=6)
        
        assert loaded_labels == labels
    
    def test_load_nonexistent_file(self):
        bboxes, labels = load_yolo_labels("nonexistent_file.txt")
        
        assert len(bboxes) == 0
        assert len(labels) == 0
    
    def test_save_empty_labels(self, tmp_path):
        label_file = tmp_path / "empty.txt"
        save_yolo_labels(str(label_file), [], [])
        
        # File should exist but be empty
        assert label_file.exists()
        loaded_bboxes, loaded_labels = load_yolo_labels(str(label_file))
        assert len(loaded_bboxes) == 0
        assert len(loaded_labels) == 0
    
    def test_yolo_label_precision(self, tmp_path):
        bboxes = [[0.123456789, 0.987654321, 0.111111111, 0.222222222]]
        labels = [0]
        
        label_file = tmp_path / "precision.txt"
        save_yolo_labels(str(label_file), bboxes, labels)
        
        loaded_bboxes, loaded_labels = load_yolo_labels(str(label_file))
        
        # Should maintain 6 decimal places
        np.testing.assert_array_almost_equal(bboxes[0], loaded_bboxes[0], decimal=6)


class TestBoundingBoxValidation:
    
    def test_valid_yolo_bbox(self):
        valid_bboxes = [
            [0.5, 0.5, 0.2, 0.3],  # Centered
            [0.1, 0.1, 0.05, 0.05],  # Small, top-left
            [0.9, 0.9, 0.1, 0.1]  # Bottom-right
        ]
        
        for bbox in valid_bboxes:
            # All values should be between 0 and 1
            assert all(0 <= val <= 1 for val in bbox)
            # Width and height should be positive
            assert bbox[2] > 0 and bbox[3] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
