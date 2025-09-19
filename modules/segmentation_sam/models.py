"""
SAM 2 Models Configuration

This module contains SAM 2 model configurations and wrapper classes based on Meta's 
Segment Anything Model 2.
"""

from typing import Dict, List, Optional, Tuple
import torch
import os
import numpy as np
from pathlib import Path


class SAMModels:
    """SAM 2 model configurations and management."""
    
    # Available SAM 2 models from Meta
    AVAILABLE_MODELS = {
        'sam2_hiera_tiny': {
            'name': 'SAM 2 Hiera Tiny',
            'description': 'Fastest SAM 2 model with Hiera backbone',
            'config': 'sam2_hiera_t.yaml',
            'checkpoint': 'sam2_hiera_tiny.pt',
            'size': '~38MB',
            'speed': 'Fastest',
            'accuracy': 'Good',
            'recommended_for': ['real_time', 'edge_devices', 'interactive_demos']
        },
        'sam2_hiera_small': {
            'name': 'SAM 2 Hiera Small',
            'description': 'Small SAM 2 model with good balance',
            'config': 'sam2_hiera_s.yaml',
            'checkpoint': 'sam2_hiera_small.pt',
            'size': '~159MB',
            'speed': 'Fast',
            'accuracy': 'Better',
            'recommended_for': ['balanced_performance', 'general_purpose']
        },
        'sam2_hiera_base_plus': {
            'name': 'SAM 2 Hiera Base Plus',
            'description': 'Base Plus SAM 2 model for high quality',
            'config': 'sam2_hiera_b+.yaml',
            'checkpoint': 'sam2_hiera_base_plus.pt',
            'size': '~80MB',
            'speed': 'Moderate',
            'accuracy': 'High',
            'recommended_for': ['high_quality', 'production_use']
        },
        'sam2_hiera_large': {
            'name': 'SAM 2 Hiera Large',
            'description': 'Large SAM 2 model for maximum accuracy',
            'config': 'sam2_hiera_l.yaml',
            'checkpoint': 'sam2_hiera_large.pt',
            'size': '~224MB',
            'speed': 'Slower',
            'accuracy': 'Highest',
            'recommended_for': ['maximum_accuracy', 'research', 'offline_processing']
        }
    }
    
    # Model selection presets
    MODEL_PRESETS = {
        'fastest': 'sam2_hiera_tiny',
        'balanced': 'sam2_hiera_small',
        'high_quality': 'sam2_hiera_base_plus',
        'maximum_accuracy': 'sam2_hiera_large'
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
        return cls.MODEL_PRESETS.get(preset, 'sam2_hiera_small')
    
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
        interactive = requirements.get('interactive', False)
        
        if speed_priority or interactive:
            return 'sam2_hiera_tiny'
        elif accuracy_priority:
            return 'sam2_hiera_large' if device_type == 'gpu' else 'sam2_hiera_base_plus'
        else:
            return 'sam2_hiera_small'  # Balanced default
    
    @classmethod
    def print_models_info(cls):
        """Print information about all available models."""
        print("Available SAM 2 Segmentation Models:")
        print("=" * 80)
        
        for model_id, info in cls.AVAILABLE_MODELS.items():
            print(f"{model_id:20} - {info['name']}")
            print(f"{'':20}   {info['description']}")
            print(f"{'':20}   Size: {info['size']}, Speed: {info['speed']}, Accuracy: {info['accuracy']}")
            print()


class SAMModelWrapper:
    """Wrapper class for SAM 2 models to provide consistent interface."""
    
    def __init__(self, model_name: str = 'sam2_hiera_small', weights_dir: str = None):
        self.model_name = model_name
        self.model = None
        self.predictor = None
        self.model_info = SAMModels.get_model_info(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Set up organized directory structure
        if weights_dir is None:
            # Get the project root directory (3 levels up from this file)
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            weights_dir = project_root / "models" / "segmentation_sam" / "weights"
        
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up cache and output directories
        self.cache_dir = self.weights_dir.parent / "cache"
        self.outputs_dir = self.weights_dir.parent / "outputs"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"SAM 2 directories initialized:")
        print(f"  Weights: {self.weights_dir}")
        print(f"  Cache: {self.cache_dir}")
        print(f"  Outputs: {self.outputs_dir}")
    
    def load_model(self):
        """Load the SAM 2 model."""
        try:
            # Try to import sam2
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
            except ImportError:
                print("Error: sam2 package not installed.")
                print("Please install SAM 2 from: https://github.com/facebookresearch/segment-anything-2")
                print("Or run: pip install git+https://github.com/facebookresearch/segment-anything-2.git")
                return False
            
            # Get model configuration and checkpoint paths
            config_name = self.model_info.get('config', 'sam2_hiera_s.yaml')
            checkpoint_name = self.model_info.get('checkpoint', 'sam2_hiera_small.pt')
            
            # Check if model exists in our weights directory
            checkpoint_path = self.weights_dir / checkpoint_name
            
            if not checkpoint_path.exists():
                print(f"Model checkpoint not found: {checkpoint_path}")
                print("Please download SAM 2 models from: https://ai.meta.com/sam2/")
                print("Available models:")
                for model_id, info in SAMModels.AVAILABLE_MODELS.items():
                    print(f"  - {info['checkpoint']} ({info['size']})")
                return False
            
            print(f"Loading SAM 2 model: {self.model_name}")
            print(f"Config: {config_name}")
            print(f"Checkpoint: {checkpoint_path}")
            
            # Build SAM 2 model
            self.model = build_sam2(config_name, str(checkpoint_path), device=self.device)
            
            # Create predictor
            self.predictor = SAM2ImagePredictor(self.model)
            
            print(f"SAM 2 model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"Error loading SAM 2 model: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None and self.predictor is not None
    
    def get_predictor(self):
        """Get the loaded predictor."""
        if not self.is_loaded():
            self.load_model()
        return self.predictor
    
    def predict_with_points(self, image: np.ndarray, points: np.ndarray, labels: np.ndarray):
        """
        Run prediction with point prompts.
        
        Args:
            image: Input image as numpy array
            points: Point coordinates as numpy array (N, 2)
            labels: Point labels as numpy array (N,) - 1 for foreground, 0 for background
            
        Returns:
            Prediction results
        """
        if not self.is_loaded():
            if not self.load_model():
                raise RuntimeError("Failed to load SAM 2 model")
        
        try:
            # Set image
            self.predictor.set_image(image)
            
            # Ensure points are in the correct format
            if points.ndim == 1:
                points = points.reshape(1, -1)
            if labels.ndim == 0:
                labels = labels.reshape(1)
            
            # Predict
            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True
            )
            
            return masks, scores, logits
            
        except Exception as e:
            print(f"Error in predict_with_points: {e}")
            print(f"Points shape: {points.shape}, dtype: {points.dtype}")
            print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")
            print(f"Image shape: {image.shape}, dtype: {image.dtype}")
            import traceback
            traceback.print_exc()
            raise
    
    def predict_with_box(self, image: np.ndarray, box: np.ndarray):
        """
        Run prediction with box prompt.
        
        Args:
            image: Input image as numpy array
            box: Bounding box as numpy array (4,) - [x1, y1, x2, y2]
            
        Returns:
            Prediction results
        """
        if not self.is_loaded():
            if not self.load_model():
                raise RuntimeError("Failed to load SAM 2 model")
        
        # Set image
        self.predictor.set_image(image)
        
        # Predict
        masks, scores, logits = self.predictor.predict(
            box=box,
            multimask_output=False
        )
        
        return masks, scores, logits
    
    def predict_with_mask(self, image: np.ndarray, mask_input: np.ndarray):
        """
        Run prediction with mask prompt.
        
        Args:
            image: Input image as numpy array
            mask_input: Input mask as numpy array
            
        Returns:
            Prediction results
        """
        if not self.is_loaded():
            if not self.load_model():
                raise RuntimeError("Failed to load SAM 2 model")
        
        # Set image
        self.predictor.set_image(image)
        
        # Predict
        masks, scores, logits = self.predictor.predict(
            mask_input=mask_input[None, :, :],
            multimask_output=False
        )
        
        return masks, scores, logits
    
    def get_info(self) -> Dict:
        """Get model information."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'loaded': self.is_loaded(),
            'weights_dir': str(self.weights_dir),
            'cache_dir': str(self.cache_dir),
            'outputs_dir': str(self.outputs_dir),
            'checkpoint_exists': (self.weights_dir / self.model_info.get('checkpoint', '')).exists(),
            **self.model_info
        }
