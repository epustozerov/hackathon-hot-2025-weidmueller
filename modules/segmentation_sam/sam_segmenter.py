"""
SAM 2 Image Segmenter

Main class for SAM 2-based image segmentation using Meta's Segment Anything Model 2.
Provides interactive segmentation with point, box, and mask prompts.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt

from .models import SAMModels, SAMModelWrapper
from .methods import SAMSegmentationMethods
from .config import setup_sam_environment, get_sam_directories


class SAMImageSegmenter:
    """SAM 2-based image segmentation class."""
    
    def __init__(self, model_name: str = 'sam2_hiera_small', 
                 input_folder: str = "data/datasets/dataset_1", 
                 output_folder: str = "data/output/output_sam"):
        self.model_name = model_name
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Set up organized SAM 2 directory structure
        self.sam_dirs = setup_sam_environment()
        print(f"SAM 2 organized directories:")
        for name, path in self.sam_dirs.items():
            print(f"  {name.capitalize()}: {path}")
        
        # Initialize SAM 2 methods
        self.methods = SAMSegmentationMethods(model_name)
        
        print(f"SAM 2 Image Segmenter initialized with model: {model_name}")
        model_info = SAMModels.get_model_info(model_name)
        if model_info:
            print(f"Model: {model_info['name']} - {model_info['description']}")
    
    def segment_with_points(self, image_path: Path, points: List[Tuple[int, int]], 
                          labels: List[int] = None, save_results: bool = True) -> Dict:
        """
        Segment image using point prompts.
        
        Args:
            image_path: Path to the image
            points: List of (x, y) coordinates
            labels: List of labels (1 for foreground, 0 for background)
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary with segmentation results
        """
        print(f"Processing image with point prompts: {image_path.name}")
        print(f"  Points: {points}")
        print(f"  Labels: {labels if labels else [1] * len(points)}")
        
        try:
            # Run SAM 2 segmentation
            segmentation_data = self.methods.segment_with_points(
                str(image_path), points, labels
            )
            
            if not segmentation_data.get('success', False):
                print(f"  ❌ Segmentation failed: {segmentation_data.get('error', 'Unknown error')}")
                return segmentation_data
            
            # Get statistics
            stats = self.methods.get_segmentation_stats(segmentation_data)
            
            print(f"  ✅ Generated {stats.get('num_masks', 0)} masks")
            print(f"  📊 Best score: {stats.get('best_score', 0):.3f}")
            print(f"  📈 Coverage: {stats.get('coverage_percentage', 0):.1f}% of image")
            
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
                    comparison_path = self.output_folder / f"{image_name}_sam_points_comparison.png"
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
    
    def segment_with_box(self, image_path: Path, box: Tuple[int, int, int, int], 
                        save_results: bool = True) -> Dict:
        """
        Segment image using box prompt.
        
        Args:
            image_path: Path to the image
            box: Bounding box as (x1, y1, x2, y2)
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary with segmentation results
        """
        print(f"Processing image with box prompt: {image_path.name}")
        print(f"  Box: {box}")
        
        try:
            # Run SAM 2 segmentation
            segmentation_data = self.methods.segment_with_box(str(image_path), box)
            
            if not segmentation_data.get('success', False):
                print(f"  ❌ Segmentation failed: {segmentation_data.get('error', 'Unknown error')}")
                return segmentation_data
            
            # Get statistics
            stats = self.methods.get_segmentation_stats(segmentation_data)
            
            print(f"  ✅ Generated {stats.get('num_masks', 0)} masks")
            print(f"  📊 Score: {stats.get('best_score', 0):.3f}")
            print(f"  📈 Coverage: {stats.get('coverage_percentage', 0):.1f}% of image")
            
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
                    comparison_path = self.output_folder / f"{image_name}_sam_box_comparison.png"
                    if self.methods.create_comparison_plot(str(image_path), segmentation_data, str(comparison_path)):
                        print(f"  📊 Saved comparison: {comparison_path.name}")
            
            # Add stats to results
            segmentation_data['stats'] = stats
            return segmentation_data
            
        except Exception as e:
            error_msg = f"Error processing {image_path.name}: {str(e)}"
            print(f"  ❌ {error_msg}")
            return {'success': False, 'error': error_msg}
    
    def segment_everything(self, image_path: Path, points_per_side: int = 32, 
                         save_results: bool = True) -> Dict:
        """
        Segment everything in the image automatically.
        
        Args:
            image_path: Path to the image
            points_per_side: Number of points per side for grid generation
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary with segmentation results
        """
        print(f"Processing image with everything segmentation: {image_path.name}")
        print(f"  Points per side: {points_per_side}")
        
        try:
            # Run SAM 2 everything segmentation
            segmentation_data = self.methods.segment_everything(str(image_path), points_per_side)
            
            if not segmentation_data.get('success', False):
                print(f"  ❌ Segmentation failed: {segmentation_data.get('error', 'Unknown error')}")
                return segmentation_data
            
            # Get statistics
            stats = self.methods.get_segmentation_stats(segmentation_data)
            
            print(f"  ✅ Generated {stats.get('num_masks', 0)} masks")
            print(f"  📊 Average score: {stats.get('average_score', 0):.3f}")
            print(f"  📈 Coverage: {stats.get('coverage_percentage', 0):.1f}% of image")
            
            # Save results if requested
            if save_results:
                image_name = image_path.stem
                save_result = self.methods.save_segmentation_results(
                    str(image_path),
                    segmentation_data,
                    self.output_folder,
                    save_types=['binary', 'two_color', 'colored', 'all_masks']
                )
                
                if save_result.get('success', False):
                    saved_files = save_result.get('saved_files', {})
                    mask_count = sum(1 for key in saved_files.keys() if key.startswith('mask_'))
                    print(f"  💾 Saved {len(saved_files) - mask_count} composite files")
                    if mask_count > 0:
                        print(f"  💾 Saved {mask_count} individual masks")
                    
                    # Create comparison plot
                    comparison_path = self.output_folder / f"{image_name}_sam_everything_comparison.png"
                    if self.methods.create_comparison_plot(str(image_path), segmentation_data, str(comparison_path)):
                        print(f"  📊 Saved comparison: {comparison_path.name}")
            
            # Add stats to results
            segmentation_data['stats'] = stats
            return segmentation_data
            
        except Exception as e:
            error_msg = f"Error processing {image_path.name}: {str(e)}"
            print(f"  ❌ {error_msg}")
            return {'success': False, 'error': error_msg}
    
    def process_all_images_with_points(self, center_points: bool = True, 
                                     save_results: bool = True) -> List[Dict]:
        """
        Process all images using center point prompts.
        
        Args:
            center_points: Whether to use center points as prompts
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
        
        print(f"🔍 Found {len(image_files)} images to process with point prompts")
        print(f"📂 Output folder: {self.output_folder}")
        print("-" * 60)
        
        results = []
        successful_count = 0
        
        for image_path in image_files:
            # Generate center point for each image
            if center_points:
                # Load image to get dimensions
                img = cv2.imread(str(image_path))
                if img is not None:
                    h, w = img.shape[:2]
                    points = [(w // 2, h // 2)]  # Center point
                    labels = [1]  # Foreground
                else:
                    points = [(400, 300)]  # Default center
                    labels = [1]
            else:
                points = [(400, 300)]  # Default point
                labels = [1]
            
            result = self.segment_with_points(image_path, points, labels, save_results)
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
                                prompt_type: str = "points", save_results: bool = True) -> Dict:
        """
        Process all datasets in the datasets folder.
        
        Args:
            base_data_folder: Base folder containing datasets
            prompt_type: Type of prompt to use ('points', 'everything')
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
            self.output_folder = Path(f"data/output/output_sam_{dataset_folder.name}")
            self.output_folder.mkdir(parents=True, exist_ok=True)
            
            # Process based on prompt type
            if prompt_type == "everything":
                # Process with everything segmentation
                image_files = list(dataset_folder.glob("*.jpg")) + list(dataset_folder.glob("*.png"))
                dataset_results = []
                for image_path in image_files:
                    result = self.segment_everything(image_path, save_results=save_results)
                    dataset_results.append(result)
            else:
                # Process with point prompts
                dataset_results = self.process_all_images_with_points(save_results=save_results)
            
            all_results[dataset_folder.name] = dataset_results
            
            print(f"✅ Completed dataset: {dataset_folder.name}")
            print("=" * 70)
            print()
            
            # Restore original paths
            self.input_folder = original_input
            self.output_folder = original_output
        
        return all_results
    
    def analyze_image_for_sam(self, image_path: Path) -> Dict:
        """
        Analyze an image to provide SAM 2 processing recommendations.
        
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
            
            # Calculate statistics
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # Edge analysis
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / total_pixels
            
            # Texture analysis
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Object detection (simple blob detection)
            detector = cv2.SimpleBlobDetector_create()
            keypoints = detector.detect(gray)
            num_objects = len(keypoints)
            
            analysis = {
                'success': True,
                'image_path': str(image_path),
                'dimensions': {'width': width, 'height': height},
                'total_pixels': total_pixels,
                'mean_intensity': float(mean_intensity),
                'std_intensity': float(std_intensity),
                'edge_density': float(edge_density),
                'texture_complexity': float(laplacian_var),
                'estimated_objects': num_objects
            }
            
            # Model recommendation based on image characteristics
            if total_pixels > 1920 * 1080:  # High resolution
                recommended_model = 'sam2_hiera_large'
            elif total_pixels < 640 * 480:  # Low resolution
                recommended_model = 'sam2_hiera_tiny'
            elif edge_density > 0.1 or laplacian_var > 1000:  # Complex image
                recommended_model = 'sam2_hiera_base_plus'
            else:
                recommended_model = 'sam2_hiera_small'
            
            # Prompt recommendation
            if num_objects <= 2:
                recommended_prompt = 'points'
                prompt_suggestion = f"Use center point or click on main object"
            elif num_objects <= 5:
                recommended_prompt = 'box'
                prompt_suggestion = "Draw bounding boxes around objects of interest"
            else:
                recommended_prompt = 'everything'
                prompt_suggestion = "Use automatic everything segmentation"
            
            analysis.update({
                'recommended_model': recommended_model,
                'recommended_prompt': recommended_prompt,
                'prompt_suggestion': prompt_suggestion,
                'reasoning': self._get_sam_recommendation_reasoning(analysis, recommended_model, recommended_prompt)
            })
            
            return analysis
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _get_sam_recommendation_reasoning(self, analysis: Dict, model: str, prompt: str) -> str:
        """Generate reasoning for SAM 2 recommendations."""
        reasons = []
        
        total_pixels = analysis['total_pixels']
        edge_density = analysis['edge_density']
        num_objects = analysis['estimated_objects']
        
        # Resolution reasoning
        if total_pixels > 1920 * 1080:
            reasons.append("High resolution image needs powerful model")
        elif total_pixels < 640 * 480:
            reasons.append("Low resolution image suitable for fast model")
        
        # Complexity reasoning
        if edge_density > 0.1:
            reasons.append("High edge density suggests complex scene")
        
        # Object count reasoning
        if num_objects <= 2:
            reasons.append("Few objects detected - point prompts work well")
        elif num_objects <= 5:
            reasons.append("Multiple objects - box prompts provide good control")
        else:
            reasons.append("Many objects detected - everything mode captures all")
        
        # Model reasoning
        model_reasoning = {
            'sam2_hiera_tiny': "Fastest model for real-time interaction",
            'sam2_hiera_small': "Balanced performance for general use",
            'sam2_hiera_base_plus': "High quality for complex scenes",
            'sam2_hiera_large': "Maximum accuracy for detailed work"
        }
        
        reasons.append(model_reasoning.get(model, "General purpose model"))
        
        return "; ".join(reasons)
    
    def print_model_info(self):
        """Print information about available SAM 2 models."""
        SAMModels.print_models_info()
    
    def get_current_model_info(self) -> Dict:
        """Get information about the currently selected model."""
        info = SAMModels.get_model_info(self.model_name)
        info.update({
            'current_model': self.model_name,
            'input_folder': str(self.input_folder),
            'output_folder': str(self.output_folder),
            'sam_directories': {name: str(path) for name, path in self.sam_dirs.items()},
            'model_checkpoint_path': str(self.sam_dirs['weights'] / info.get('checkpoint', '')),
            'model_exists_locally': (self.sam_dirs['weights'] / info.get('checkpoint', '')).exists()
        })
        return info
