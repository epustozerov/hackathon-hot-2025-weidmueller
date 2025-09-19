"""
YOLO Models Configuration

This module contains YOLO model configurations and wrapper classes.
"""

from typing import Dict, List, Optional
import torch
from pathlib import Path


class YOLOModels:
    """YOLO model configurations and management."""
    
    # Available YOLO11 segmentation models based on Ultralytics documentation
    AVAILABLE_MODELS = {
        'yolo11n-seg': {
            'name': 'YOLO11 Nano Segment',
            'description': 'Fastest and smallest YOLO11 segmentation model',
            'params': '2.9M',
            'flops': '10.4B',
            'map_box': 38.9,
            'map_mask': 32.0,
            'speed_cpu': '65.9ms',
            'speed_gpu': '1.8ms',
            'recommended_for': ['real_time', 'edge_devices', 'fast_inference']
        },
        'yolo11s-seg': {
            'name': 'YOLO11 Small Segment',
            'description': 'Small YOLO11 segmentation model with good balance',
            'params': '10.1M',
            'flops': '35.5B',
            'map_box': 46.6,
            'map_mask': 37.8,
            'speed_cpu': '117.6ms',
            'speed_gpu': '2.9ms',
            'recommended_for': ['balanced_performance', 'mobile_devices']
        },
        'yolo11m-seg': {
            'name': 'YOLO11 Medium Segment',
            'description': 'Medium YOLO11 segmentation model for better accuracy',
            'params': '22.4M',
            'flops': '123.3B',
            'map_box': 51.5,
            'map_mask': 41.5,
            'speed_cpu': '281.6ms',
            'speed_gpu': '6.3ms',
            'recommended_for': ['good_accuracy', 'general_purpose']
        },
        'yolo11l-seg': {
            'name': 'YOLO11 Large Segment',
            'description': 'Large YOLO11 segmentation model for high accuracy',
            'params': '27.6M',
            'flops': '142.2B',
            'map_box': 53.4,
            'map_mask': 42.9,
            'speed_cpu': '344.2ms',
            'speed_gpu': '7.8ms',
            'recommended_for': ['high_accuracy', 'server_deployment']
        },
        'yolo11x-seg': {
            'name': 'YOLO11 Extra Large Segment',
            'description': 'Largest YOLO11 segmentation model for maximum accuracy',
            'params': '62.1M',
            'flops': '319.0B',
            'map_box': 54.7,
            'map_mask': 43.8,
            'speed_cpu': '664.5ms',
            'speed_gpu': '15.8ms',
            'recommended_for': ['maximum_accuracy', 'research', 'offline_processing']
        }
    }
    
    # Model selection presets
    MODEL_PRESETS = {
        'fastest': 'yolo11n-seg',
        'balanced': 'yolo11s-seg',
        'accurate': 'yolo11m-seg',
        'high_accuracy': 'yolo11l-seg',
        'maximum_accuracy': 'yolo11x-seg'
    }
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Dict:
        """Get information about a specific model."""
        return cls.AVAILABLE_MODELS.get(model_name, {})
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available model names."""
        return list(cls.AVAILABLE_MODELS.keys())
    
    @classmethod
    def get_model_by_preset(cls, preset: str) -> str:
        """Get model name by preset."""
        return cls.MODEL_PRESETS.get(preset, 'yolo11s-seg')
    
    @classmethod
    def recommend_model(cls, requirements: Dict) -> str:
        """
        Recommend a model based on requirements.
        
        Args:
            requirements: Dict with keys like 'speed_priority', 'accuracy_priority', 'device_type'
        
        Returns:
            Recommended model name
        """
        speed_priority = requirements.get('speed_priority', False)
        accuracy_priority = requirements.get('accuracy_priority', False)
        device_type = requirements.get('device_type', 'cpu')
        
        if speed_priority and device_type in ['cpu', 'mobile']:
            return 'yolo11n-seg'
        elif accuracy_priority:
            return 'yolo11l-seg' if device_type == 'gpu' else 'yolo11m-seg'
        else:
            return 'yolo11s-seg'  # Balanced default
    
    @classmethod
    def print_models_info(cls):
        """Print information about all available models."""
        print("Available YOLO11 Segmentation Models:")
        print("=" * 80)
        
        for model_id, info in cls.AVAILABLE_MODELS.items():
            print(f"{model_id:15} - {info['name']}")
            print(f"{'':15}   {info['description']}")
            print(f"{'':15}   Parameters: {info['params']}, mAP(mask): {info['map_mask']}")
            print(f"{'':15}   Speed: CPU {info['speed_cpu']}, GPU {info['speed_gpu']}")
            print()


class YOLOModelWrapper:
    """Wrapper class for YOLO models to provide consistent interface."""
    
    def __init__(self, model_name: str = 'yolo11s-seg'):
        self.model_name = model_name
        self.model = None
        self.model_info = YOLOModels.get_model_info(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def load_model(self):
        """Load the YOLO model."""
        try:
            from ultralytics import YOLO
            print(f"Loading YOLO model: {self.model_name}")
            self.model = YOLO(f"{self.model_name}.pt")
            print(f"Model loaded successfully on {self.device}")
            return True
        except ImportError:
            print("Error: ultralytics package not installed. Install with: pip install ultralytics")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def get_model(self):
        """Get the loaded model."""
        if not self.is_loaded():
            self.load_model()
        return self.model
    
    def predict(self, source, **kwargs):
        """Run prediction on source."""
        if not self.is_loaded():
            if not self.load_model():
                raise RuntimeError("Failed to load YOLO model")
        
        return self.model(source, **kwargs)
    
    def get_info(self) -> Dict:
        """Get model information."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'loaded': self.is_loaded(),
            **self.model_info
        }
