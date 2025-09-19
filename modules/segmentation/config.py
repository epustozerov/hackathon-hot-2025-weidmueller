"""
Configuration for Segmentation Methods

This module contains configuration options for different segmentation methods.
"""

# Available segmentation methods
AVAILABLE_METHODS = {
    'deeplabv3': {
        'name': 'DeepLabV3',
        'description': 'State-of-the-art semantic segmentation with ResNet-50 backbone',
        'type': 'neural_network',
        'pretrained': True,
        'recommended_for': ['general_purpose', 'high_texture_complexity']
    },
    'unet': {
        'name': 'U-Net',
        'description': 'Classic encoder-decoder architecture for medical image segmentation',
        'type': 'neural_network',
        'pretrained': False,
        'recommended_for': ['medical_images', 'binary_segmentation']
    },
    'neural_clustering': {
        'name': 'Neural Clustering',
        'description': 'K-means clustering on ResNet-50 features',
        'type': 'hybrid',
        'pretrained': True,
        'recommended_for': ['high_contrast', 'feature_based_segmentation']
    },
    'grabcut_neural': {
        'name': 'Neural GrabCut',
        'description': 'Traditional GrabCut algorithm guided by neural network predictions',
        'type': 'hybrid',
        'pretrained': True,
        'recommended_for': ['high_edge_density', 'refinement']
    }
}

# Default method combinations for different scenarios
DEFAULT_METHOD_SETS = {
    'fast': ['deeplabv3'],
    'balanced': ['deeplabv3', 'neural_clustering'],
    'comprehensive': ['deeplabv3', 'neural_clustering', 'grabcut_neural'],
    'all': ['deeplabv3', 'unet', 'neural_clustering', 'grabcut_neural']
}

# Method selection criteria
SELECTION_CRITERIA = {
    'texture_complexity_threshold': 500,
    'edge_density_threshold': 0.08,
    'contrast_threshold': 50,
    'general_purpose_fallback': 'deeplabv3'
}

def get_recommended_methods(image_stats):
    """
    Get recommended methods based on image statistics.
    
    Args:
        image_stats (dict): Dictionary containing image statistics
            - mean_intensity: float
            - std_intensity: float
            - edge_density: float
            - texture_complexity: float
    
    Returns:
        list: Recommended method names
    """
    recommended = []
    
    # Always include DeepLabV3 as it's generally robust
    recommended.append('deeplabv3')
    
    # Add specific methods based on image characteristics
    if image_stats.get('texture_complexity', 0) > SELECTION_CRITERIA['texture_complexity_threshold']:
        if 'deeplabv3' not in recommended:
            recommended.append('deeplabv3')
    
    if image_stats.get('edge_density', 0) > SELECTION_CRITERIA['edge_density_threshold']:
        recommended.append('grabcut_neural')
    
    if image_stats.get('std_intensity', 0) > SELECTION_CRITERIA['contrast_threshold']:
        recommended.append('neural_clustering')
    
    # Ensure we have at least one method
    if not recommended:
        recommended.append(SELECTION_CRITERIA['general_purpose_fallback'])
    
    return recommended
