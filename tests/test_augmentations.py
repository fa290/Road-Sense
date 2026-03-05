"""
Unit tests for data augmentation pipelines.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.augmentations import (
    get_training_augmentation,
    get_training_augmentation_with_bbox,
    get_validation_augmentation,
    get_validation_augmentation_with_bbox,
    get_inference_augmentation,
    get_custom_augmentation,
    AUGMENTATION_PRESETS
)


@pytest.fixture
def dummy_image():
    return np.random.randint(0, 255, (375, 1242, 3), dtype=np.uint8)


@pytest.fixture
def dummy_bboxes():
    return [[0.5, 0.5, 0.2, 0.3], [0.3, 0.3, 0.1, 0.2]]


@pytest.fixture
def dummy_labels():
    return [0, 1]


class TestTrainingAugmentation:
    
    def test_training_augmentation_without_bbox(self, dummy_image):
        transform = get_training_augmentation()
        result = transform(image=dummy_image)
        
        assert 'image' in result
        assert result['image'].shape == dummy_image.shape
        assert result['image'].dtype == np.uint8
    
    def test_training_augmentation_with_resize(self, dummy_image):
        target_size = (384, 1280)
        transform = get_training_augmentation(image_size=target_size)
        result = transform(image=dummy_image)
        
        assert result['image'].shape[:2] == target_size
    
    def test_training_augmentation_with_bbox(self, dummy_image, dummy_bboxes, dummy_labels):
        transform = get_training_augmentation_with_bbox()
        result = transform(
            image=dummy_image,
            bboxes=dummy_bboxes,
            class_labels=dummy_labels
        )
        
        assert 'image' in result
        assert 'bboxes' in result
        assert 'class_labels' in result
        assert result['image'].shape == dummy_image.shape
        assert len(result['bboxes']) <= len(dummy_bboxes)  # Some may be filtered
        assert len(result['bboxes']) == len(result['class_labels'])
    
    def test_training_augmentation_bbox_format(self, dummy_image, dummy_bboxes, dummy_labels):
        transform = get_training_augmentation_with_bbox()
        result = transform(
            image=dummy_image,
            bboxes=dummy_bboxes,
            class_labels=dummy_labels
        )
        
        for bbox in result['bboxes']:
            assert len(bbox) == 4
            # All values should be between 0 and 1 (YOLO format)
            assert all(0 <= val <= 1 for val in bbox)
    
    def test_min_visibility_filter(self, dummy_image, dummy_bboxes, dummy_labels):
        transform = get_training_augmentation_with_bbox(min_visibility=0.9)
        result = transform(
            image=dummy_image,
            bboxes=dummy_bboxes,
            class_labels=dummy_labels
        )
        
        # With high min_visibility, some boxes may be filtered
        assert len(result['bboxes']) <= len(dummy_bboxes)


class TestValidationAugmentation:
    
    def test_validation_augmentation_no_resize(self, dummy_image):
        transform = get_validation_augmentation()
        result = transform(image=dummy_image)
        
        assert result['image'].shape == dummy_image.shape
    
    def test_validation_augmentation_with_resize(self, dummy_image):
        target_size = (384, 1280)
        transform = get_validation_augmentation(image_size=target_size)
        result = transform(image=dummy_image)
        
        assert result['image'].shape[:2] == target_size
    
    def test_validation_augmentation_with_bbox(self, dummy_image, dummy_bboxes, dummy_labels):
        transform = get_validation_augmentation_with_bbox()
        result = transform(
            image=dummy_image,
            bboxes=dummy_bboxes,
            class_labels=dummy_labels
        )
        
        assert 'image' in result
        assert 'bboxes' in result
        assert 'class_labels' in result
        # Validation should not remove boxes (no augmentation)
        assert len(result['bboxes']) == len(dummy_bboxes)


class TestInferenceAugmentation:
    
    def test_inference_augmentation(self, dummy_image):
        transform = get_inference_augmentation()
        result = transform(image=dummy_image)
        
        assert result['image'].shape == dummy_image.shape
    
    def test_inference_augmentation_with_resize(self, dummy_image):
        target_size = (384, 1280)
        transform = get_inference_augmentation(image_size=target_size)
        result = transform(image=dummy_image)
        
        assert result['image'].shape[:2] == target_size


class TestCustomAugmentation:
    
    def test_preset_light(self, dummy_image):
        transform = get_custom_augmentation(preset='light')
        result = transform(image=dummy_image)
        
        assert result['image'].shape == dummy_image.shape
    
    def test_preset_medium(self, dummy_image):
        transform = get_custom_augmentation(preset='medium')
        result = transform(image=dummy_image)
        
        assert result['image'].shape == dummy_image.shape
    
    def test_preset_heavy(self, dummy_image):
        transform = get_custom_augmentation(preset='heavy')
        result = transform(image=dummy_image)
        
        assert result['image'].shape == dummy_image.shape
    
    def test_invalid_preset(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            get_custom_augmentation(preset='invalid')
    
    def test_custom_with_bbox(self, dummy_image, dummy_bboxes, dummy_labels):
        transform = get_custom_augmentation(preset='medium', with_bbox=True)
        result = transform(
            image=dummy_image,
            bboxes=dummy_bboxes,
            class_labels=dummy_labels
        )
        
        assert 'bboxes' in result
        assert 'class_labels' in result
        assert len(result['bboxes']) == len(result['class_labels'])
    
    def test_custom_with_resize(self, dummy_image):
        target_size = (384, 1280)
        transform = get_custom_augmentation(preset='medium', image_size=target_size)
        result = transform(image=dummy_image)
        
        assert result['image'].shape[:2] == target_size


class TestAugmentationPresets:
    
    def test_presets_exist(self):
        assert 'light' in AUGMENTATION_PRESETS
        assert 'medium' in AUGMENTATION_PRESETS
        assert 'heavy' in AUGMENTATION_PRESETS
    
    def test_preset_structure(self):
        required_keys = [
            'horizontal_flip_p',
            'shift_scale_rotate_p',
            'brightness_contrast_p',
            'weather_p'
        ]
        
        for preset_name, preset_config in AUGMENTATION_PRESETS.items():
            for key in required_keys:
                assert key in preset_config, f"Missing {key} in {preset_name}"
                assert 0 <= preset_config[key] <= 1, f"Invalid probability in {preset_name}"
    
    def test_preset_probability_order(self):
        light = AUGMENTATION_PRESETS['light']
        medium = AUGMENTATION_PRESETS['medium']
        heavy = AUGMENTATION_PRESETS['heavy']
        
        assert light['horizontal_flip_p'] <= medium['horizontal_flip_p'] <= heavy['horizontal_flip_p']
        assert light['brightness_contrast_p'] <= medium['brightness_contrast_p'] <= heavy['brightness_contrast_p']


class TestAugmentationConsistency:
    
    def test_augmentation_produces_variation(self, dummy_image):
        transform = get_training_augmentation()
        
        results = []
        for _ in range(5):
            result = transform(image=dummy_image.copy())
            results.append(result['image'])
        
        # At least some results should be different (probabilistic test)
        differences = []
        for i in range(len(results) - 1):
            diff = np.sum(results[i] != results[i + 1])
            differences.append(diff)
        
        # Should have at least one difference (very likely with augmentations)
        assert any(d > 0 for d in differences)
    
    def test_validation_is_deterministic(self, dummy_image):
        transform = get_validation_augmentation()
        
        result1 = transform(image=dummy_image.copy())
        result2 = transform(image=dummy_image.copy())
        
        # Results should be identical (no augmentation)
        assert np.array_equal(result1['image'], result2['image'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
