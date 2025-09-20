"""
SAM Unified Models Configuration

This module contains unified SAM 2 and SAM 2.1 model configurations and wrapper classes.
It combines both SAM 2 and SAM 2.1 models into a single interface.
"""

from typing import Dict, List
import torch
import numpy as np
from pathlib import Path


class SAMModels:
    """Unified SAM 2 and SAM 2.1 model configurations and management."""
    
    # Available SAM models from Meta (both SAM 2 and SAM 2.1)
    AVAILABLE_MODELS = {
        # SAM 2.0 models (legacy)
        'sam2_hiera_tiny': {
            'name': 'SAM 2 Hiera Tiny',
            'description': 'Fastest SAM 2 model with Hiera backbone',
            'config': 'sam2_hiera_t.yaml',
            'checkpoint': 'sam2_hiera_tiny.pt',
            'size': '~38MB',
            'speed': 'Fastest',
            'accuracy': 'Good',
            'version': '2.0',
            'family': 'SAM 2',
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
            'version': '2.0',
            'family': 'SAM 2',
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
            'version': '2.0',
            'family': 'SAM 2',
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
            'version': '2.0',
            'family': 'SAM 2',
            'recommended_for': ['maximum_accuracy', 'research', 'offline_processing']
        },
        # SAM 2.1 models (new and improved)
        'sam2.1_hiera_tiny': {
            'name': 'SAM 2.1 Hiera Tiny',
            'description': 'Fastest SAM 2.1 model with improved performance',
            'config': 'sam2.1_hiera_t.yaml',
            'checkpoint': 'sam2.1_hiera_tiny.pt',
            'size': '~38MB',
            'speed': 'Fastest',
            'accuracy': 'Better',
            'version': '2.1',
            'family': 'SAM 2.1',
            'recommended_for': ['real_time', 'edge_devices', 'interactive_demos']
        },
        'sam2.1_hiera_small': {
            'name': 'SAM 2.1 Hiera Small',
            'description': 'Small SAM 2.1 model with enhanced accuracy',
            'config': 'sam2.1_hiera_s.yaml',
            'checkpoint': 'sam2.1_hiera_small.pt',
            'size': '~159MB',
            'speed': 'Fast',
            'accuracy': 'High',
            'version': '2.1',
            'family': 'SAM 2.1',
            'recommended_for': ['balanced_performance', 'general_purpose', 'production']
        },
        'sam2.1_hiera_base_plus': {
            'name': 'SAM 2.1 Hiera Base Plus',
            'description': 'Base Plus SAM 2.1 model with superior quality',
            'config': 'sam2.1_hiera_b+.yaml',
            'checkpoint': 'sam2.1_hiera_base_plus.pt',
            'size': '~80MB',
            'speed': 'Moderate',
            'accuracy': 'Very High',
            'version': '2.1',
            'family': 'SAM 2.1',
            'recommended_for': ['high_quality', 'production_use', 'professional']
        },
        'sam2.1_hiera_large': {
            'name': 'SAM 2.1 Hiera Large',
            'description': 'Large SAM 2.1 model with maximum accuracy and improvements',
            'config': 'sam2.1_hiera_l.yaml',
            'checkpoint': 'sam2.1_hiera_large.pt',
            'size': '~224MB',
            'speed': 'Slower',
            'accuracy': 'Highest',
            'version': '2.1',
            'family': 'SAM 2.1',
            'recommended_for': ['maximum_accuracy', 'research', 'offline_processing', 'professional']
        }
    }
    
    # Model selection presets (default to SAM 2.1 for better performance)
    MODEL_PRESETS = {
        'fastest': 'sam2.1_hiera_tiny',
        'balanced': 'sam2.1_hiera_small',
        'high_quality': 'sam2.1_hiera_base_plus',
        'maximum_accuracy': 'sam2.1_hiera_large'
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
    def get_available_models_by_version(cls, version: str = None) -> List[str]:
        """Get list of available model names by version."""
        if version is None:
            return list(cls.AVAILABLE_MODELS.keys())
        return [name for name, info in cls.AVAILABLE_MODELS.items() if info.get('version') == version]
    
    @classmethod
    def get_model_by_preset(cls, preset: str) -> str:
        """Get model name by preset."""
        return cls.MODEL_PRESETS.get(preset, 'sam2.1_hiera_small')
    
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
            return 'sam2.1_hiera_tiny'
        elif accuracy_priority:
            return 'sam2.1_hiera_large'
        else:
            return 'sam2.1_hiera_small'
    
    @classmethod
    def print_models_info(cls):
        """Print information about all available models."""
        print("Available SAM Models:")
        print("=" * 80)
        
        # Group by family
        sam2_models = {k: v for k, v in cls.AVAILABLE_MODELS.items() if v.get('family') == 'SAM 2'}
        sam21_models = {k: v for k, v in cls.AVAILABLE_MODELS.items() if v.get('family') == 'SAM 2.1'}
        
        if sam2_models:
            print("\nSAM 2 Models:")
            print("-" * 40)
            for model_id, info in sam2_models.items():
                print(f"{model_id:<25} - {info['name']}")
                print(f"{'':25}   {info['description']}")
                print(f"{'':25}   Size: {info['size']}, Speed: {info['speed']}, Accuracy: {info['accuracy']}")
                print()
        
        if sam21_models:
            print("\nSAM 2.1 Models:")
            print("-" * 40)
            for model_id, info in sam21_models.items():
                print(f"{model_id:<25} - {info['name']}")
                print(f"{'':25}   {info['description']}")
                print(f"{'':25}   Size: {info['size']}, Speed: {info['speed']}, Accuracy: {info['accuracy']}")
                print()
        
        print("\nModel Presets:")
        print("-" * 40)
        print("fastest            - Fastest model for real-time applications")
        print("balanced           - Good balance of speed and accuracy")
        print("high_quality       - High quality segmentation")
        print("maximum_accuracy   - Maximum accuracy for research")
    
    @classmethod
    def get_sam2_models(cls) -> List[str]:
        """Get list of SAM 2.0 models."""
        return cls.get_available_models_by_version('2.0')
    
    @classmethod
    def get_sam21_models(cls) -> List[str]:
        """Get list of SAM 2.1 models."""
        return cls.get_available_models_by_version('2.1')
    
    @classmethod
    def is_sam2_model(cls, model_name: str) -> bool:
        """Check if model is SAM 2.0."""
        info = cls.get_model_info(model_name)
        return info.get('version') == '2.0'
    
    @classmethod
    def is_sam21_model(cls, model_name: str) -> bool:
        """Check if model is SAM 2.1."""
        info = cls.get_model_info(model_name)
        return info.get('version') == '2.1'


class SAMModelWrapper:
    """Unified wrapper class for SAM 2 and SAM 2.1 models to provide consistent interface."""
    
    def __init__(self, model_name: str = 'sam2.1_hiera_small', weights_dir: str = None, device: str = None):
        """
        Initialize unified SAM model wrapper.
        
        Args:
            model_name: Name of the SAM model to use (defaults to SAM 2.1 for better performance)
            weights_dir: Directory containing model weights
            device: Device to run the model on ('cpu' or 'cuda')
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model_name = model_name
        self.device = device
        self.model = None
        self.predictor = None
        
        # Get model information
        self.model_info = SAMModels.get_model_info(model_name)
        if not self.model_info:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Set up directories
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
        self.configs_dir = self.weights_dir.parent / "configs"
        
        # Create directories
        for dir_path in [self.cache_dir, self.outputs_dir, self.configs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"SAM unified directories initialized:")
        print(f"  Weights: {self.weights_dir}")
        print(f"  Cache: {self.cache_dir}")
        print(f"  Outputs: {self.outputs_dir}")
        print(f"  Configs: {self.configs_dir}")
    
    def load_model(self):
        """Load the SAM model (works for both SAM 2 and SAM 2.1)."""
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
                print("Please download SAM models from: https://ai.meta.com/sam2/")
                print("Available models:")
                for model_id, info in SAMModels.AVAILABLE_MODELS.items():
                    print(f"  - {info['checkpoint']} ({info['size']}) - {info['family']}")
                return False
            
            print(f"Loading SAM model: {self.model_name}")
            print(f"Version: {self.model_info.get('version', 'Unknown')}")
            print(f"Config: {config_name}")
            print(f"Checkpoint: {checkpoint_path}")
            
            # Build SAM model (works for both SAM 2 and SAM 2.1)
            self.model = build_sam2(config_name, str(checkpoint_path), device=self.device)
            
            # Create predictor
            self.predictor = SAM2ImagePredictor(self.model)
            
            print(f"SAM model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"Error loading SAM model: {e}")
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
                raise RuntimeError("Failed to load SAM model")
        
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
                raise RuntimeError("Failed to load SAM model")
        
        # Set image
        self.predictor.set_image(image)
        
        # Predict
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=True
        )
        
        return masks, scores, logits
    
    def predict_everything(self, image: np.ndarray, points_per_side: int = 32):
        """
        Run prediction to segment everything in the image.
        
        Args:
            image: Input image as numpy array
            points_per_side: Number of points per side for grid generation
            
        Returns:
            Prediction results
        """
        if not self.is_loaded():
            if not self.load_model():
                raise RuntimeError("Failed to load SAM model")
        
        # Set image
        self.predictor.set_image(image)
        
        # Generate grid of points
        h, w = image.shape[:2]
        points = []
        step_x = w // points_per_side
        step_y = h // points_per_side
        
        for y in range(step_y // 2, h, step_y):
            for x in range(step_x // 2, w, step_x):
                points.append([x, y])
        
        points = np.array(points, dtype=np.float32)
        labels = np.ones(len(points), dtype=np.int32)
        
        # Predict
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True
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