"""
YOLO Segmentation Configuration

Configuration settings and utilities for YOLO-based image segmentation.
"""

from typing import Dict, List, Any

# YOLO Model Configuration
YOLO_CONFIG = {
    'default_model': 'yolo11s-seg',
    'default_conf_threshold': 0.25,
    'default_iou_threshold': 0.7,
    'default_image_size': 640,
    'save_confidence': True,
    'save_crops': False,
    'save_txt': False,
    'verbose': False
}

# Model Selection Presets
MODEL_PRESETS = {
    'fastest': {
        'model': 'yolo11n-seg',
        'description': 'Fastest inference, lowest accuracy',
        'use_case': 'Real-time applications, edge devices',
        'conf_threshold': 0.3,
        'iou_threshold': 0.7
    },
    'balanced': {
        'model': 'yolo11s-seg',
        'description': 'Good balance of speed and accuracy',
        'use_case': 'General purpose applications',
        'conf_threshold': 0.25,
        'iou_threshold': 0.7
    },
    'accurate': {
        'model': 'yolo11m-seg',
        'description': 'Higher accuracy, moderate speed',
        'use_case': 'Quality-focused applications',
        'conf_threshold': 0.2,
        'iou_threshold': 0.7
    },
    'high_accuracy': {
        'model': 'yolo11l-seg',
        'description': 'High accuracy, slower inference',
        'use_case': 'Offline processing, detailed analysis',
        'conf_threshold': 0.15,
        'iou_threshold': 0.7
    },
    'maximum_accuracy': {
        'model': 'yolo11x-seg',
        'description': 'Maximum accuracy, slowest inference',
        'use_case': 'Research, maximum quality requirements',
        'conf_threshold': 0.1,
        'iou_threshold': 0.7
    }
}

# Output Configuration
OUTPUT_CONFIG = {
    'save_formats': ['binary', 'two_color', 'colored'],
    'create_comparison': True,
    'save_statistics': True,
    'image_quality': 95,
    'dpi': 150
}

# Class Names (COCO dataset classes that YOLO can detect)
COCO_CLASS_NAMES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

# Color palette for visualization
VISUALIZATION_COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green  
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 0, 0),    # Maroon
    (0, 128, 0),    # Dark Green
    (0, 0, 128),    # Navy
    (128, 128, 0),  # Olive
    (128, 0, 128),  # Purple
    (0, 128, 128),  # Teal
    (192, 192, 192), # Silver
    (128, 128, 128), # Gray
    (255, 165, 0),  # Orange
    (255, 192, 203), # Pink
    (173, 216, 230), # Light Blue
    (144, 238, 144), # Light Green
    (255, 218, 185), # Peach
    (221, 160, 221)  # Plum
]


def get_preset_config(preset_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific preset.
    
    Args:
        preset_name: Name of the preset
        
    Returns:
        Configuration dictionary
    """
    if preset_name not in MODEL_PRESETS:
        preset_name = 'balanced'  # Default fallback
    
    preset = MODEL_PRESETS[preset_name]
    config = YOLO_CONFIG.copy()
    config.update({
        'model': preset['model'],
        'conf_threshold': preset['conf_threshold'],
        'iou_threshold': preset['iou_threshold']
    })
    
    return config


def get_recommended_preset(image_characteristics: Dict) -> str:
    """
    Recommend a preset based on image characteristics.
    
    Args:
        image_characteristics: Dictionary with image analysis results
        
    Returns:
        Recommended preset name
    """
    total_pixels = image_characteristics.get('total_pixels', 0)
    edge_density = image_characteristics.get('edge_density', 0)
    texture_complexity = image_characteristics.get('texture_complexity', 0)
    
    # High resolution images
    if total_pixels > 1920 * 1080:
        if edge_density > 0.1 or texture_complexity > 1000:
            return 'high_accuracy'
        else:
            return 'accurate'
    
    # Low resolution images
    elif total_pixels < 640 * 480:
        return 'fastest'
    
    # Medium resolution images
    else:
        if edge_density > 0.08 or texture_complexity > 500:
            return 'accurate'
        else:
            return 'balanced'


def print_available_presets():
    """Print information about available presets."""
    print("Available YOLO Model Presets:")
    print("=" * 60)
    
    for preset_name, preset_info in MODEL_PRESETS.items():
        print(f"{preset_name:18} - {preset_info['model']}")
        print(f"{'':18}   {preset_info['description']}")
        print(f"{'':18}   Use case: {preset_info['use_case']}")
        print(f"{'':18}   Confidence: {preset_info['conf_threshold']}")
        print()


def get_class_name(class_id: int) -> str:
    """
    Get class name from class ID.
    
    Args:
        class_id: COCO class ID
        
    Returns:
        Class name string
    """
    return COCO_CLASS_NAMES.get(class_id, f'unknown_{class_id}')


def get_visualization_color(index: int) -> tuple:
    """
    Get visualization color for a given index.
    
    Args:
        index: Color index
        
    Returns:
        RGB color tuple
    """
    return VISUALIZATION_COLORS[index % len(VISUALIZATION_COLORS)]


def validate_config(config: Dict) -> Dict:
    """
    Validate and sanitize configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validated configuration dictionary
    """
    validated = YOLO_CONFIG.copy()
    
    # Validate model name
    if 'model' in config:
        model = config['model']
        if not model.endswith('-seg'):
            if not model.startswith('yolo11'):
                model = f'yolo11{model}-seg'
            else:
                model = f'{model}-seg'
        validated['model'] = model
    
    # Validate thresholds
    if 'conf_threshold' in config:
        conf = float(config['conf_threshold'])
        validated['conf_threshold'] = max(0.01, min(1.0, conf))
    
    if 'iou_threshold' in config:
        iou = float(config['iou_threshold'])
        validated['iou_threshold'] = max(0.01, min(1.0, iou))
    
    # Validate image size
    if 'image_size' in config:
        size = int(config['image_size'])
        validated['image_size'] = max(320, min(1280, size))
    
    return validated
