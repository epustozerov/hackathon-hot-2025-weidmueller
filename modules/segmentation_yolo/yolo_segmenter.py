"""
YOLO Image Segmenter

Main class for YOLO-based image segmentation using Ultralytics YOLO models.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt

from .models import YOLOModels, YOLOModelWrapper
from .methods import YOLOSegmentationMethods


class YOLOImageSegmenter:
    """YOLO-based image segmentation class."""
    
    def __init__(self, model_name: str = 'yolo11s-seg', 
                 input_folder: str = "data/datasets/dataset_1", 
                 output_folder: str = "data/output/output_yolo"):
        self.model_name = model_name
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Initialize YOLO methods
        self.methods = YOLOSegmentationMethods(model_name)
        
        # Configuration
        self.conf_threshold = 0.25
        self.iou_threshold = 0.7
        
        print(f"YOLO Image Segmenter initialized with model: {model_name}")
        model_info = YOLOModels.get_model_info(model_name)
        if model_info:
            print(f"Model: {model_info['name']} - {model_info['description']}")
    
    def set_thresholds(self, conf_threshold: float = 0.25, iou_threshold: float = 0.7):
        """Set confidence and IoU thresholds."""
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        print(f"Thresholds updated - Confidence: {conf_threshold}, IoU: {iou_threshold}")
    
    def process_single_image(self, image_path: Path, save_results: bool = True) -> Dict:
        """
        Process a single image with YOLO segmentation.
        
        Args:
            image_path: Path to the image
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary with processing results
        """
        print(f"Processing image: {image_path.name}")
        
        try:
            # Run YOLO segmentation
            segmentation_data = self.methods.segment_image(
                str(image_path),
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold
            )
            
            if not segmentation_data.get('success', False):
                print(f"  ❌ Segmentation failed: {segmentation_data.get('error', 'Unknown error')}")
                return segmentation_data
            
            # Get statistics
            stats = self.methods.get_segmentation_stats(segmentation_data)
            
            print(f"  ✅ Detected {stats.get('total_objects', 0)} objects")
            print(f"  📊 Coverage: {stats.get('mask_coverage_percentage', 0):.1f}% of image")
            
            # Print detected classes
            classes = stats.get('classes_detected', {})
            if classes:
                class_info = ", ".join([f"{name}({count})" for name, count in classes.items()])
                print(f"  🏷️  Classes: {class_info}")
            
            # Save results if requested
            if save_results:
                image_name = image_path.stem
                save_result = self.methods.save_segmentation_results(
                    str(image_path),
                    segmentation_data,
                    self.output_folder,
                    save_types=['binary', 'two_color', 'colored']
                )
                
                if save_result.get('success', False):
                    saved_files = save_result.get('saved_files', {})
                    for file_type, file_path in saved_files.items():
                        print(f"  💾 Saved {file_type}: {Path(file_path).name}")
                    
                    # Create comparison plot
                    comparison_path = self.output_folder / f"{image_name}_yolo_comparison.png"
                    if self.methods.create_comparison_plot(str(image_path), segmentation_data, str(comparison_path)):
                        print(f"  📊 Saved comparison: {comparison_path.name}")
                else:
                    print(f"  ❌ Failed to save results: {save_result.get('error', 'Unknown error')}")
            
            # Add stats to results
            segmentation_data['stats'] = stats
            return segmentation_data
            
        except Exception as e:
            error_msg = f"Error processing {image_path.name}: {str(e)}"
            print(f"  ❌ {error_msg}")
            return {'success': False, 'error': error_msg}
    
    def process_all_images(self, save_results: bool = True) -> List[Dict]:
        """
        Process all images in the input folder.
        
        Args:
            save_results: Whether to save results to disk
            
        Returns:
            List of processing results for each image
        """
        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(self.input_folder.glob(ext))
            image_files.extend(self.input_folder.glob(ext.upper()))
        
        if not image_files:
            print(f"❌ No image files found in {self.input_folder}")
            return []
        
        print(f"🔍 Found {len(image_files)} images to process")
        print(f"📂 Output folder: {self.output_folder}")
        print("-" * 60)
        
        results = []
        successful_count = 0
        
        for image_path in image_files:
            result = self.process_single_image(image_path, save_results)
            results.append(result)
            
            if result.get('success', False):
                successful_count += 1
            
            print("-" * 60)
        
        print(f"🎉 Processing complete!")
        print(f"✅ Successfully processed: {successful_count}/{len(image_files)} images")
        
        if save_results:
            output_files = list(self.output_folder.glob("*"))
            print(f"💾 Generated {len(output_files)} output files")
        
        return results
    
    def process_multiple_datasets(self, base_data_folder: str = "data/datasets", 
                                save_results: bool = True) -> Dict:
        """
        Process all datasets in the datasets folder.
        
        Args:
            base_data_folder: Base folder containing datasets
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary with results for each dataset
        """
        base_path = Path(base_data_folder)
        
        if not base_path.exists():
            print(f"❌ Base data folder not found: {base_path}")
            return {}
        
        # Find all dataset folders
        dataset_folders = [folder for folder in base_path.iterdir() if folder.is_dir()]
        
        if not dataset_folders:
            print(f"❌ No dataset folders found in {base_path}")
            return {}
        
        print(f"🗂️  Found {len(dataset_folders)} datasets to process:")
        for folder in dataset_folders:
            print(f"   - {folder.name}")
        print()
        
        all_results = {}
        
        # Process each dataset
        for dataset_folder in dataset_folders:
            print(f"📁 Processing dataset: {dataset_folder.name}")
            print("=" * 70)
            
            # Update paths for this dataset
            original_input = self.input_folder
            original_output = self.output_folder
            
            self.input_folder = dataset_folder
            self.output_folder = Path(f"data/output/output_yolo_{dataset_folder.name}")
            self.output_folder.mkdir(parents=True, exist_ok=True)
            
            # Process all images in this dataset
            dataset_results = self.process_all_images(save_results)
            all_results[dataset_folder.name] = dataset_results
            
            print(f"✅ Completed dataset: {dataset_folder.name}")
            print("=" * 70)
            print()
            
            # Restore original paths
            self.input_folder = original_input
            self.output_folder = original_output
        
        return all_results
    
    def analyze_image_for_yolo(self, image_path: Path) -> Dict:
        """
        Analyze an image to determine optimal YOLO model and settings.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dictionary with analysis results and recommendations
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return {'success': False, 'error': 'Could not load image'}
            
            # Basic image analysis
            height, width = image.shape[:2]
            total_pixels = height * width
            
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Calculate statistics
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # Edge analysis
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / total_pixels
            
            # Texture analysis
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Color analysis
            color_channels = cv2.split(image)
            color_std = np.mean([np.std(channel) for channel in color_channels])
            
            analysis = {
                'success': True,
                'image_path': str(image_path),
                'dimensions': {'width': width, 'height': height},
                'total_pixels': total_pixels,
                'mean_intensity': float(mean_intensity),
                'std_intensity': float(std_intensity),
                'edge_density': float(edge_density),
                'texture_complexity': float(laplacian_var),
                'color_variation': float(color_std)
            }
            
            # Model recommendation based on image characteristics
            if total_pixels > 1920 * 1080:  # High resolution
                if edge_density > 0.1 or laplacian_var > 1000:
                    recommended_model = 'yolo11l-seg'  # High accuracy for complex high-res images
                else:
                    recommended_model = 'yolo11m-seg'  # Balanced for simple high-res images
            elif total_pixels < 640 * 480:  # Low resolution
                recommended_model = 'yolo11n-seg'  # Fast for small images
            else:  # Medium resolution
                if edge_density > 0.08 or laplacian_var > 500:
                    recommended_model = 'yolo11m-seg'  # Good accuracy for complex images
                else:
                    recommended_model = 'yolo11s-seg'  # Balanced for simple images
            
            # Confidence threshold recommendation
            if std_intensity < 30:  # Low contrast
                recommended_conf = 0.15
            elif edge_density > 0.1:  # High detail
                recommended_conf = 0.35
            else:
                recommended_conf = 0.25
            
            analysis.update({
                'recommended_model': recommended_model,
                'recommended_conf_threshold': recommended_conf,
                'recommended_iou_threshold': 0.7,
                'reasoning': self._get_recommendation_reasoning(analysis, recommended_model)
            })
            
            return analysis
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _get_recommendation_reasoning(self, analysis: Dict, model: str) -> str:
        """Generate reasoning for model recommendation."""
        reasons = []
        
        total_pixels = analysis['total_pixels']
        edge_density = analysis['edge_density']
        texture_complexity = analysis['texture_complexity']
        
        # Resolution reasoning
        if total_pixels > 1920 * 1080:
            reasons.append("High resolution image detected")
        elif total_pixels < 640 * 480:
            reasons.append("Low resolution image detected")
        
        # Complexity reasoning
        if edge_density > 0.1:
            reasons.append("High edge density suggests complex objects")
        if texture_complexity > 1000:
            reasons.append("High texture complexity detected")
        elif texture_complexity < 100:
            reasons.append("Simple image with low texture complexity")
        
        # Model reasoning
        model_reasoning = {
            'yolo11n-seg': "Fastest model for real-time processing",
            'yolo11s-seg': "Balanced performance and accuracy",
            'yolo11m-seg': "Good accuracy for complex scenes",
            'yolo11l-seg': "High accuracy for detailed segmentation",
            'yolo11x-seg': "Maximum accuracy for research applications"
        }
        
        reasons.append(model_reasoning.get(model, "General purpose model"))
        
        return "; ".join(reasons)
    
    def print_model_info(self):
        """Print information about available YOLO models."""
        YOLOModels.print_models_info()
    
    def get_current_model_info(self) -> Dict:
        """Get information about the currently selected model."""
        info = YOLOModels.get_model_info(self.model_name)
        info.update({
            'current_model': self.model_name,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'input_folder': str(self.input_folder),
            'output_folder': str(self.output_folder)
        })
        return info
