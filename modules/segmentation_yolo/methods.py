"""
YOLO Segmentation Methods

This module contains YOLO-specific segmentation methods and utilities.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import torch

from .models import YOLOModelWrapper


class YOLOSegmentationMethods:
    """YOLO-based segmentation methods."""
    
    def __init__(self, model_name: str = 'yolo11s-seg'):
        self.model_wrapper = YOLOModelWrapper(model_name)
        self.model_name = model_name
        
    def segment_image(self, image_path: str, conf_threshold: float = 0.25, 
                     iou_threshold: float = 0.7) -> Dict:
        """
        Segment image using YOLO model.
        
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Dictionary containing segmentation results
        """
        try:
            # Run YOLO prediction
            results = self.model_wrapper.predict(
                image_path,
                conf=conf_threshold,
                iou=iou_threshold,
                save=False,
                verbose=False
            )
            
            if not results:
                return {'success': False, 'error': 'No results from YOLO model'}
            
            result = results[0]  # Get first result
            
            # Extract information
            segmentation_data = {
                'success': True,
                'image_path': image_path,
                'model_name': self.model_name,
                'conf_threshold': conf_threshold,
                'iou_threshold': iou_threshold,
                'detections': []
            }
            
            # Process detections if masks are available
            if result.masks is not None:
                masks_data = result.masks.data  # Tensor of shape (N, H, W)
                boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
                confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
                classes = result.boxes.cls.cpu().numpy() if result.boxes is not None else []
                class_names = result.names if hasattr(result, 'names') else {}
                
                # Convert masks to numpy
                masks_np = masks_data.cpu().numpy()
                
                for i in range(len(masks_np)):
                    detection = {
                        'mask': masks_np[i],
                        'confidence': float(confidences[i]) if i < len(confidences) else 0.0,
                        'class_id': int(classes[i]) if i < len(classes) else -1,
                        'class_name': class_names.get(int(classes[i]), 'unknown') if i < len(classes) else 'unknown',
                        'bbox': boxes[i].tolist() if i < len(boxes) else []
                    }
                    segmentation_data['detections'].append(detection)
            
            return segmentation_data
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def create_binary_mask(self, segmentation_data: Dict, combine_all: bool = True) -> Optional[np.ndarray]:
        """
        Create binary mask from YOLO segmentation results.
        
        Args:
            segmentation_data: Results from segment_image
            combine_all: If True, combine all detected objects into one mask
            
        Returns:
            Binary mask as numpy array
        """
        if not segmentation_data.get('success', False):
            return None
        
        detections = segmentation_data.get('detections', [])
        if not detections:
            return None
        
        # Get image dimensions from first mask
        first_mask = detections[0]['mask']
        h, w = first_mask.shape
        
        if combine_all:
            # Combine all masks
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            for detection in detections:
                mask = detection['mask']
                combined_mask = np.logical_or(combined_mask, mask > 0.5)
            return combined_mask.astype(np.uint8) * 255
        else:
            # Return individual masks
            masks = []
            for detection in detections:
                mask = (detection['mask'] > 0.5).astype(np.uint8) * 255
                masks.append(mask)
            return masks
    
    def create_colored_segmentation(self, image_path: str, segmentation_data: Dict) -> Optional[np.ndarray]:
        """
        Create colored segmentation overlay on original image.
        
        Args:
            image_path: Path to original image
            segmentation_data: Results from segment_image
            
        Returns:
            Image with colored segmentation overlay
        """
        if not segmentation_data.get('success', False):
            return None
        
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        overlay = image_rgb.copy()
        
        # Define colors for different classes
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]
        
        detections = segmentation_data.get('detections', [])
        for i, detection in enumerate(detections):
            mask = detection['mask']
            color = colors[i % len(colors)]
            
            # Resize mask to match image size
            mask_resized = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]))
            mask_bool = mask_resized > 0.5
            
            # Apply colored overlay
            overlay[mask_bool] = color
        
        # Blend with original image
        alpha = 0.6
        result = cv2.addWeighted(image_rgb, 1-alpha, overlay, alpha, 0)
        
        return result
    
    def create_two_color_segmentation(self, image_path: str, segmentation_data: Dict) -> Optional[np.ndarray]:
        """
        Create two-color (black/white) segmentation.
        
        Args:
            image_path: Path to original image
            segmentation_data: Results from segment_image
            
        Returns:
            Two-color segmentation image
        """
        binary_mask = self.create_binary_mask(segmentation_data, combine_all=True)
        if binary_mask is None:
            return None
        
        # Load original image to get dimensions
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Resize mask to match original image dimensions
        mask_resized = cv2.resize(binary_mask, (image.shape[1], image.shape[0]))
        
        # Create two-color image
        two_color = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        two_color[mask_resized > 127] = [255, 255, 255]  # White for foreground
        # Background remains black (0, 0, 0)
        
        return two_color
    
    def get_segmentation_stats(self, segmentation_data: Dict) -> Dict:
        """
        Get statistics from segmentation results.
        
        Args:
            segmentation_data: Results from segment_image
            
        Returns:
            Dictionary with segmentation statistics
        """
        if not segmentation_data.get('success', False):
            return {'success': False}
        
        detections = segmentation_data.get('detections', [])
        
        stats = {
            'success': True,
            'total_objects': len(detections),
            'classes_detected': {},
            'average_confidence': 0.0,
            'max_confidence': 0.0,
            'min_confidence': 1.0,
            'total_mask_area': 0,
            'mask_coverage_percentage': 0.0
        }
        
        if not detections:
            return stats
        
        confidences = []
        total_pixels = 0
        image_pixels = 0
        
        for detection in detections:
            # Confidence stats
            conf = detection.get('confidence', 0.0)
            confidences.append(conf)
            
            # Class stats
            class_name = detection.get('class_name', 'unknown')
            stats['classes_detected'][class_name] = stats['classes_detected'].get(class_name, 0) + 1
            
            # Mask area stats
            mask = detection.get('mask', np.array([]))
            if mask.size > 0:
                mask_pixels = np.sum(mask > 0.5)
                total_pixels += mask_pixels
                if image_pixels == 0:
                    image_pixels = mask.size
        
        # Calculate statistics
        if confidences:
            stats['average_confidence'] = float(np.mean(confidences))
            stats['max_confidence'] = float(np.max(confidences))
            stats['min_confidence'] = float(np.min(confidences))
        
        stats['total_mask_area'] = int(total_pixels)
        if image_pixels > 0:
            stats['mask_coverage_percentage'] = float(total_pixels / image_pixels * 100)
        
        return stats
    
    def save_segmentation_results(self, image_path: str, segmentation_data: Dict, 
                                output_folder: Path, save_types: List[str] = None) -> Dict:
        """
        Save segmentation results in various formats.
        
        Args:
            image_path: Path to original image
            segmentation_data: Results from segment_image
            output_folder: Output directory
            save_types: List of types to save ['binary', 'colored', 'two_color']
            
        Returns:
            Dictionary with saved file paths
        """
        if save_types is None:
            save_types = ['binary', 'two_color', 'colored']
        
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        image_name = Path(image_path).stem
        saved_files = {}
        
        try:
            # Save binary mask
            if 'binary' in save_types:
                binary_mask = self.create_binary_mask(segmentation_data)
                if binary_mask is not None:
                    binary_path = output_folder / f"{image_name}_yolo_binary_mask.png"
                    cv2.imwrite(str(binary_path), binary_mask)
                    saved_files['binary'] = str(binary_path)
            
            # Save two-color segmentation
            if 'two_color' in save_types:
                two_color = self.create_two_color_segmentation(image_path, segmentation_data)
                if two_color is not None:
                    two_color_path = output_folder / f"{image_name}_yolo_two_color.png"
                    cv2.imwrite(str(two_color_path), cv2.cvtColor(two_color, cv2.COLOR_RGB2BGR))
                    saved_files['two_color'] = str(two_color_path)
            
            # Save colored overlay
            if 'colored' in save_types:
                colored = self.create_colored_segmentation(image_path, segmentation_data)
                if colored is not None:
                    colored_path = output_folder / f"{image_name}_yolo_colored.png"
                    cv2.imwrite(str(colored_path), cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))
                    saved_files['colored'] = str(colored_path)
            
            return {'success': True, 'saved_files': saved_files}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def create_comparison_plot(self, image_path: str, segmentation_data: Dict, 
                             output_path: Optional[str] = None) -> bool:
        """
        Create comparison plot showing original image and segmentation results.
        
        Args:
            image_path: Path to original image
            segmentation_data: Results from segment_image
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
            colored = self.create_colored_segmentation(image_path, segmentation_data)
            
            # Create subplot
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            # Original image
            axes[0].imshow(original_rgb)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
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
