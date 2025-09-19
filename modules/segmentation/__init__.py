"""
Image Segmentation Module

This module provides neural network-based image segmentation capabilities.
"""

from .neural_segmenter import NeuralImageSegmenter
from .models import UNet, DeepLabV3Segmenter
from .methods import SegmentationMethods

__all__ = ['NeuralImageSegmenter', 'UNet', 'DeepLabV3Segmenter', 'SegmentationMethods']
