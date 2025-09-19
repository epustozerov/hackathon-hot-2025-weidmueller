"""
SAM 2 Segmentation Methods

This module contains SAM 2-specific segmentation methods and utilities.
SAM 2 provides interactive segmentation with point, box, and mask prompts.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
from pathlib import Path
import torch

from .models import SAMModelWrapper
from .config import setup_sam_environment


class SAMSegmentationMethods:
    """SAM 2-based segmentation methods."""
    
    def __init__(self, model_name: str = 'sam2_hiera_small'):
        # Set up organized directory structure
        self.sam_dirs = setup_sam_environment()
        
        # Initialize model wrapper with organized weights directory
        self.model_wrapper = SAMModelWrapper(model_name, str(self.sam_dirs['weights']))
        self.model_name = model_name
        self.current_image = None
        self.current_image_path = None
        
    def load_image(self, image_path: str) -> np.ndarray:
        """Load and prepare image for SAM 2."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Store for interactive use
        self.current_image = image_rgb
        self.current_image_path = image_path
        
        return image_rgb
    
    def segment_with_points(self, image_path: str, points: List[Tuple[int, int]], 
                          labels: List[int] = None) -> Dict:
        """
        Segment image using point prompts.
        
        Args:
            image_path: Path to input image
            points: List of (x, y) coordinates for prompts
            labels: List of labels (1 for foreground, 0 for background)
            
        Returns:
            Dictionary containing segmentation results
        """
        try:
            # Load image
            image = self.load_image(image_path)
            
            # Default labels (all foreground) if not provided
            if labels is None:
                labels = [1] * len(points)
            
            if len(points) != len(labels):
                raise ValueError("Number of points must match number of labels")
            
            # Convert to numpy arrays
            points_np = np.array(points, dtype=np.float32)
            labels_np = np.array(labels, dtype=np.int32)
            
            # Run SAM 2 prediction
            masks, scores, logits = self.model_wrapper.predict_with_points(
                image, points_np, labels_np
            )
            
            # Process results safely
            # Convert scores to list safely
            try:
                if isinstance(scores, np.ndarray):
                    scores_list = scores.tolist()
                else:
                    scores_list = list(scores)
            except Exception as e:
                scores_list = [0.0]
            
            # Get best mask index safely
            try:
                if len(scores_list) > 0:
                    best_mask_idx = int(np.argmax(scores_list))
                else:
                    best_mask_idx = 0
            except Exception as e:
                best_mask_idx = 0
            
            segmentation_data = {
                'success': True,
                'image_path': image_path,
                'model_name': self.model_name,
                'prompt_type': 'points',
                'points': points,
                'labels': labels,
                'masks': masks,
                'scores': scores_list,
                'best_mask_idx': best_mask_idx,
                'num_masks': len(masks) if hasattr(masks, '__len__') else 1
            }
            
            return segmentation_data
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'prompt_type': 'points'}
    
    def segment_with_box(self, image_path: str, box: Tuple[int, int, int, int]) -> Dict:
        """
        Segment image using box prompt.
        
        Args:
            image_path: Path to input image
            box: Bounding box as (x1, y1, x2, y2)
            
        Returns:
            Dictionary containing segmentation results
        """
        try:
            # Load image
            image = self.load_image(image_path)
            
            # Convert box to numpy array
            box_np = np.array(box, dtype=np.float32)
            
            # Run SAM 2 prediction
            masks, scores, logits = self.model_wrapper.predict_with_box(image, box_np)
            
            # Process results
            # Convert scores to list safely
            if hasattr(scores, 'tolist'):
                scores_list = scores.tolist()
            elif hasattr(scores, '__iter__'):
                scores_list = list(scores)
            else:
                scores_list = [float(scores)]
            
            segmentation_data = {
                'success': True,
                'image_path': image_path,
                'model_name': self.model_name,
                'prompt_type': 'box',
                'box': box,
                'masks': masks,
                'scores': scores_list,
                'best_mask_idx': 0,  # Single mask for box prompt
                'num_masks': len(masks) if hasattr(masks, '__len__') else 1
            }
            
            return segmentation_data
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'prompt_type': 'box'}
    
    def segment_with_mask(self, image_path: str, mask_input: np.ndarray) -> Dict:
        """
        Segment image using mask prompt.
        
        Args:
            image_path: Path to input image
            mask_input: Input mask for refinement
            
        Returns:
            Dictionary containing segmentation results
        """
        try:
            # Load image
            image = self.load_image(image_path)
            
            # Run SAM 2 prediction
            masks, scores, logits = self.model_wrapper.predict_with_mask(image, mask_input)
            
            # Process results
            segmentation_data = {
                'success': True,
                'image_path': image_path,
                'model_name': self.model_name,
                'prompt_type': 'mask',
                'masks': masks,
                'scores': scores.tolist() if hasattr(scores, 'tolist') else list(scores),
                'best_mask_idx': 0,  # Single mask for mask prompt
                'num_masks': len(masks)
            }
            
            return segmentation_data
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'prompt_type': 'mask'}
    
    def segment_everything(self, image_path: str, points_per_side: int = 32) -> Dict:
        """
        Segment everything in the image using automatic point generation.
        
        Args:
            image_path: Path to input image
            points_per_side: Number of points per side for grid generation
            
        Returns:
            Dictionary containing segmentation results
        """
        try:
            # Load image
            image = self.load_image(image_path)
            h, w = image.shape[:2]
            
            # Generate grid of points
            points = []
            step_x = w // points_per_side
            step_y = h // points_per_side
            
            for y in range(step_y // 2, h, step_y):
                for x in range(step_x // 2, w, step_x):
                    points.append((x, y))
            
            # All points are foreground
            labels = [1] * len(points)
            
            # Use point-based segmentation
            return self.segment_with_points(image_path, points, labels)
            
        except Exception as e:
            return {'success': False, 'error': str(e), 'prompt_type': 'everything'}
    
    def create_binary_mask(self, segmentation_data: Dict, mask_idx: int = None) -> Optional[np.ndarray]:
        """
        Create binary mask from SAM 2 segmentation results.
        
        Args:
            segmentation_data: Results from segmentation methods
            mask_idx: Index of mask to use (None for best mask)
            
        Returns:
            Binary mask as numpy array
        """
        if not segmentation_data.get('success', False):
            return None
        
        masks = segmentation_data.get('masks')
        if masks is None or len(masks) == 0:
            return None
        
        # Use specified mask or best mask
        if mask_idx is None:
            mask_idx = segmentation_data.get('best_mask_idx', 0)
        
        if mask_idx >= len(masks):
            mask_idx = 0
        
        # Convert to binary mask
        mask = masks[mask_idx]
        
        # Ensure mask is a numpy array
        if hasattr(mask, 'cpu'):
            mask = mask.cpu().numpy()
        elif not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        
        # Safely create binary mask
        try:
            mask_bool = mask > 0.5
            binary_mask = mask_bool.astype(np.uint8) * 255
        except Exception:
            # Fallback: treat mask as binary
            binary_mask = mask.astype(np.uint8) * 255
        
        return binary_mask
    
    def create_two_color_segmentation(self, image_path: str, segmentation_data: Dict, 
                                    mask_idx: int = None) -> Optional[np.ndarray]:
        """
        Create two-color (black/white) segmentation.
        
        Args:
            image_path: Path to original image
            segmentation_data: Results from segmentation methods
            mask_idx: Index of mask to use
            
        Returns:
            Two-color segmentation image
        """
        binary_mask = self.create_binary_mask(segmentation_data, mask_idx)
        if binary_mask is None:
            return None
        
        # Load original image to get dimensions
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Create two-color image
        two_color = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        try:
            mask_bool = binary_mask > 127
            two_color[mask_bool] = [255, 255, 255]  # White for foreground
        except Exception:
            # Fallback: treat mask as binary
            two_color[binary_mask > 0] = [255, 255, 255]
        # Background remains black (0, 0, 0)
        
        return two_color
    
    def create_colored_overlay(self, image_path: str, segmentation_data: Dict, 
                             alpha: float = 0.6) -> Optional[np.ndarray]:
        """
        Create colored overlay on original image.
        
        Args:
            image_path: Path to original image
            segmentation_data: Results from segmentation methods
            alpha: Transparency factor for overlay
            
        Returns:
            Image with colored overlay
        """
        if not segmentation_data.get('success', False):
            return None
        
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        overlay = image_rgb.copy()
        
        # Get masks
        masks = segmentation_data.get('masks', [])
        # Safely check if masks is empty
        if masks is None or (hasattr(masks, '__len__') and len(masks) == 0):
            return image_rgb
        
        # Define colors for different masks
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)
        ]
        
        # Apply colored overlay for each mask
        for i, mask in enumerate(masks):
            color = colors[i % len(colors)]
            try:
                mask_bool = mask > 0.5
                overlay[mask_bool] = color
            except Exception:
                # Fallback: treat mask as binary
                overlay[mask > 0] = color
        
        # Blend with original image
        result = cv2.addWeighted(image_rgb, 1-alpha, overlay, alpha, 0)
        
        return result
    
    def get_segmentation_stats(self, segmentation_data: Dict) -> Dict:
        """
        Get statistics from segmentation results.
        
        Args:
            segmentation_data: Results from segmentation methods
            
        Returns:
            Dictionary with segmentation statistics
        """
        if not segmentation_data.get('success', False):
            return {'success': False}
        
        masks = segmentation_data.get('masks', [])
        scores = segmentation_data.get('scores', [])
        
        # Safely check if masks is empty
        if masks is None or (hasattr(masks, '__len__') and len(masks) == 0):
            return {'success': True, 'num_masks': 0}
        
        stats = {
            'success': True,
            'num_masks': len(masks),
            'prompt_type': segmentation_data.get('prompt_type', 'unknown'),
            'model_name': segmentation_data.get('model_name', 'unknown'),
            'scores': scores,
            'best_score': float(max(scores)) if len(scores) > 0 else 0.0,
            'average_score': float(sum(scores) / len(scores)) if len(scores) > 0 else 0.0,
            'mask_areas': [],
            'total_area': 0,
            'coverage_percentage': 0.0
        }
        
        # Calculate mask areas
        total_pixels = 0
        image_pixels = 0
        
        for i, mask in enumerate(masks):
            # Ensure mask is a numpy array
            if hasattr(mask, 'cpu'):
                mask = mask.cpu().numpy()
            elif not isinstance(mask, np.ndarray):
                mask = np.array(mask)
            
            # Safely calculate mask pixels
            try:
                mask_bool = mask > 0.5
                mask_pixels = int(np.sum(mask_bool))
            except Exception:
                # Fallback: treat mask as binary
                mask_pixels = int(np.sum(mask))
            
            stats['mask_areas'].append(mask_pixels)
            total_pixels += mask_pixels
            if image_pixels == 0:
                image_pixels = mask.size
        
        stats['total_area'] = int(total_pixels)
        if image_pixels > 0:
            stats['coverage_percentage'] = float(total_pixels / image_pixels * 100)
        
        return stats
    
    def save_segmentation_results(self, image_path: str, segmentation_data: Dict, 
                                output_folder: Path, save_types: List[str] = None) -> Dict:
        """
        Save segmentation results in various formats.
        
        Args:
            image_path: Path to original image
            segmentation_data: Results from segmentation methods
            output_folder: Output directory
            save_types: List of types to save ['binary', 'colored', 'two_color', 'all_masks']
            
        Returns:
            Dictionary with saved file paths
        """
        if save_types is None:
            save_types = ['binary', 'two_color', 'colored']
        
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        image_name = Path(image_path).stem
        prompt_type = segmentation_data.get('prompt_type', 'unknown')
        saved_files = {}
        
        try:
            # Save binary mask
            if 'binary' in save_types:
                binary_mask = self.create_binary_mask(segmentation_data)
                if binary_mask is not None:
                    binary_path = output_folder / f"{image_name}_sam_{prompt_type}_binary.png"
                    cv2.imwrite(str(binary_path), binary_mask)
                    saved_files['binary'] = str(binary_path)
            
            # Save two-color segmentation
            if 'two_color' in save_types:
                two_color = self.create_two_color_segmentation(image_path, segmentation_data)
                if two_color is not None:
                    two_color_path = output_folder / f"{image_name}_sam_{prompt_type}_two_color.png"
                    cv2.imwrite(str(two_color_path), cv2.cvtColor(two_color, cv2.COLOR_RGB2BGR))
                    saved_files['two_color'] = str(two_color_path)
            
            # Save colored overlay
            if 'colored' in save_types:
                colored = self.create_colored_overlay(image_path, segmentation_data)
                if colored is not None:
                    colored_path = output_folder / f"{image_name}_sam_{prompt_type}_colored.png"
                    cv2.imwrite(str(colored_path), cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
                    saved_files['colored'] = str(colored_path)
            
            # Save all masks individually
            if 'all_masks' in save_types:
                masks = segmentation_data.get('masks', [])
                for i, mask in enumerate(masks):
                    try:
                        mask_bool = mask > 0.5
                        mask_binary = mask_bool.astype(np.uint8) * 255
                    except Exception:
                        # Fallback: treat mask as binary
                        mask_binary = mask.astype(np.uint8) * 255
                    
                    mask_path = output_folder / f"{image_name}_sam_{prompt_type}_mask_{i}.png"
                    cv2.imwrite(str(mask_path), mask_binary)
                    saved_files[f'mask_{i}'] = str(mask_path)
            
            return {'success': True, 'saved_files': saved_files}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def create_comparison_plot(self, image_path: str, segmentation_data: Dict, 
                             output_path: Optional[str] = None) -> bool:
        """
        Create comparison plot showing original image and segmentation results.
        
        Args:
            image_path: Path to original image
            segmentation_data: Results from segmentation methods
            output_path: Optional path to save the plot
            
        Returns:
            Success status
        """
        try:
            # Load original image
            original = cv2.imread(image_path)
            if original is None:
                return False
            
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            
            # Create different visualizations
            binary_mask = self.create_binary_mask(segmentation_data)
            two_color = self.create_two_color_segmentation(image_path, segmentation_data)
            colored = self.create_colored_overlay(image_path, segmentation_data)
            
            # Create subplot
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            # Original image
            axes[0].imshow(original_rgb)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Add prompt visualization if available
            prompt_type = segmentation_data.get('prompt_type', 'unknown')
            if prompt_type == 'points':
                points = segmentation_data.get('points', [])
                labels = segmentation_data.get('labels', [])
                for point, label in zip(points, labels):
                    color = 'red' if label == 1 else 'blue'
                    axes[0].plot(point[0], point[1], 'o', color=color, markersize=8)
            elif prompt_type == 'box':
                box = segmentation_data.get('box', [])
                if box:
                    x1, y1, x2, y2 = box
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2)
                    axes[0].add_patch(rect)
            
            # Binary mask
            if binary_mask is not None:
                axes[1].imshow(binary_mask, cmap='gray')
                axes[1].set_title('Binary Mask')
            else:
                axes[1].text(0.5, 0.5, 'No Mask', ha='center', va='center')
                axes[1].set_title('Binary Mask (Failed)')
            axes[1].axis('off')
            
            # Two-color segmentation
            if two_color is not None:
                axes[2].imshow(two_color)
                axes[2].set_title('Two-Color Segmentation')
            else:
                axes[2].text(0.5, 0.5, 'No Segmentation', ha='center', va='center')
                axes[2].set_title('Two-Color (Failed)')
            axes[2].axis('off')
            
            # Colored overlay
            if colored is not None:
                axes[3].imshow(colored)
                axes[3].set_title('Colored Overlay')
            else:
                axes[3].text(0.5, 0.5, 'No Overlay', ha='center', va='center')
                axes[3].set_title('Colored Overlay (Failed)')
            axes[3].axis('off')
            
            # Add title with stats
            stats = self.get_segmentation_stats(segmentation_data)
            if stats.get('success', False):
                fig.suptitle(f'SAM 2 Segmentation - {prompt_type.title()} Prompt | '
                           f'Masks: {stats.get("num_masks", 0)} | '
                           f'Best Score: {stats.get("best_score", 0):.3f}', 
                           fontsize=14)
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
            
            return True
            
        except Exception as e:
            print(f"Error creating comparison plot: {e}")
            return False
