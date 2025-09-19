"""
SAM 2 Segmentation Configuration

Configuration settings and utilities for SAM 2-based image segmentation.
"""

from typing import Dict, List, Any
from pathlib import Path
import os

# SAM 2 Model Configuration
SAM_CONFIG = {
    'default_model': 'sam2_hiera_small',
    'multimask_output': True,
    'stability_score_thresh': 0.95,
    'stability_score_offset': 1.0,
    'box_nms_thresh': 0.7,
    'crop_n_layers': 0,
    'crop_nms_thresh': 0.7,
    'crop_overlap_ratio': 512 / 1500,
    'crop_n_points_downscale_factor': 1,
    'point_grids': None,
    'min_mask_region_area': 0,
    'output_mode': 'binary_mask'
}

# Directory Configuration
def get_sam_directories():
    """Get organized SAM 2 directory structure."""
    # Get the project root directory
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent
    
    base_dir = project_root / "models" / "segmentation_sam"
    
    directories = {
        'base': base_dir,
        'weights': base_dir / "weights",
        'cache': base_dir / "cache", 
        'outputs': base_dir / "outputs",
        'configs': base_dir / "configs"
    }
    
    # Create directories if they don't exist
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return directories

def setup_sam_environment():
    """Set up environment for SAM 2 to use organized directories."""
    dirs = get_sam_directories()
    
    # Set environment variables if needed
    os.environ['SAM_CACHE_DIR'] = str(dirs['cache'])
    
    return dirs

# Model Selection Presets
MODEL_PRESETS = {
    'fastest': {
        'model': 'sam2_hiera_tiny',
        'description': 'Fastest inference for real-time applications',
        'use_case': 'Interactive demos, real-time segmentation',
        'multimask_output': True
    },
    'balanced': {
        'model': 'sam2_hiera_small',
        'description': 'Good balance of speed and accuracy',
        'use_case': 'General purpose segmentation',
        'multimask_output': True
    },
    'high_quality': {
        'model': 'sam2_hiera_base_plus',
        'description': 'High quality segmentation',
        'use_case': 'Production applications requiring quality',
        'multimask_output': True
    },
    'maximum_accuracy': {
        'model': 'sam2_hiera_large',
        'description': 'Maximum accuracy for research',
        'use_case': 'Research, offline processing',
        'multimask_output': True
    }
}

# Prompt Type Configuration
PROMPT_CONFIGS = {
    'point': {
        'description': 'Single or multiple point prompts',
        'parameters': ['points', 'labels'],
        'multimask_output': True,
        'use_case': 'Interactive segmentation, object selection'
    },
    'box': {
        'description': 'Bounding box prompt',
        'parameters': ['box'],
        'multimask_output': False,
        'use_case': 'Object detection integration, region of interest'
    },
    'mask': {
        'description': 'Mask refinement prompt',
        'parameters': ['mask_input'],
        'multimask_output': False,
        'use_case': 'Mask refinement, iterative improvement'
    },
    'everything': {
        'description': 'Automatic everything segmentation',
        'parameters': ['points_per_side'],
        'multimask_output': True,
        'use_case': 'Scene understanding, automatic annotation'
    }
}

# Output Configuration
OUTPUT_CONFIG = {
    'save_formats': ['binary', 'two_color', 'colored'],
    'create_comparison': True,
    'save_all_masks': False,
    'image_quality': 95,
    'dpi': 150,
    'overlay_alpha': 0.6
}

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
    config = SAM_CONFIG.copy()
    config.update({
        'model': preset['model'],
        'multimask_output': preset['multimask_output']
    })
    
    return config

def get_recommended_preset(requirements: Dict) -> str:
    """
    Recommend a preset based on requirements.
    
    Args:
        requirements: Dictionary with requirement specifications
        
    Returns:
        Recommended preset name
    """
    speed_priority = requirements.get('speed_priority', False)
    accuracy_priority = requirements.get('accuracy_priority', False)
    interactive = requirements.get('interactive', False)
    device_type = requirements.get('device_type', 'cpu')
    
    if speed_priority or interactive:
        return 'fastest'
    elif accuracy_priority:
        return 'maximum_accuracy' if device_type == 'gpu' else 'high_quality'
    else:
        return 'balanced'

def print_available_presets():
    """Print information about available presets."""
    print("Available SAM 2 Model Presets:")
    print("=" * 60)
    
    for preset_name, preset_info in MODEL_PRESETS.items():
        print(f"{preset_name:18} - {preset_info['model']}")
        print(f"{'':18}   {preset_info['description']}")
        print(f"{'':18}   Use case: {preset_info['use_case']}")
        print()

def print_prompt_types():
    """Print information about available prompt types."""
    print("Available SAM 2 Prompt Types:")
    print("=" * 60)
    
    for prompt_type, prompt_info in PROMPT_CONFIGS.items():
        print(f"{prompt_type:12} - {prompt_info['description']}")
        print(f"{'':12}   Parameters: {', '.join(prompt_info['parameters'])}")
        print(f"{'':12}   Use case: {prompt_info['use_case']}")
        print()

def validate_config(config: Dict) -> Dict:
    """
    Validate and sanitize configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validated configuration dictionary
    """
    validated = SAM_CONFIG.copy()
    
    # Validate model name
    if 'model' in config:
        model = config['model']
        if model not in ['sam2_hiera_tiny', 'sam2_hiera_small', 'sam2_hiera_base_plus', 'sam2_hiera_large']:
            model = 'sam2_hiera_small'  # Default fallback
        validated['model'] = model
    
    # Validate boolean settings
    for bool_key in ['multimask_output']:
        if bool_key in config:
            validated[bool_key] = bool(config[bool_key])
    
    # Validate threshold values
    for thresh_key in ['stability_score_thresh', 'box_nms_thresh', 'crop_nms_thresh']:
        if thresh_key in config:
            thresh = float(config[thresh_key])
            validated[thresh_key] = max(0.0, min(1.0, thresh))
    
    return validated

# Download URLs for SAM 2 models (from Meta AI)
MODEL_URLS = {
    'sam2_hiera_tiny': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt',
    'sam2_hiera_small': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt',
    'sam2_hiera_base_plus': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt',
    'sam2_hiera_large': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt'
}

def get_model_download_info():
    """Get information about downloading SAM 2 models."""
    print("SAM 2 Model Download Information:")
    print("=" * 60)
    print("Models need to be downloaded manually from Meta AI:")
    print("https://ai.meta.com/sam2/")
    print()
    print("Available models:")
    for model_name, url in MODEL_URLS.items():
        size = {
            'sam2_hiera_tiny': '~38MB',
            'sam2_hiera_small': '~159MB', 
            'sam2_hiera_base_plus': '~80MB',
            'sam2_hiera_large': '~224MB'
        }.get(model_name, 'Unknown')
        print(f"  {model_name:20} - {size}")
        print(f"  {'':20}   {url}")
        print()
    
    dirs = get_sam_directories()
    print(f"Download models to: {dirs['weights']}")
    print()
    print("Installation instructions:")
    print("1. Install SAM 2: pip install git+https://github.com/facebookresearch/segment-anything-2.git")
    print("2. Download model checkpoints to the weights directory")
    print("3. Run: python app_sam.py --list-models to verify installation")
