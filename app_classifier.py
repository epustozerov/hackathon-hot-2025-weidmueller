#!/usr/bin/env python3
"""
Object Classification and Similarity Evaluation Application

This application uses neural networks to:
1. Extract object borders using edge detection
2. Compare cut objects with templates using similarity metrics
3. Evaluate alignment and similarity between objects and templates

Features:
- Neural network-based edge detection for object borders
- Multiple similarity metrics (SSIM, structural similarity, feature matching)
- Template matching and comparison
- Automated evaluation pipeline

Configuration:
- Modify parameters in the main() function
- Set paths for cut objects and templates
- Configure neural network parameters
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime

# Import comparison methods
from modules.comparison.comparison import (
    create_point_cloud, 
    edge_segmentation, 
    compare_images
)

# Import sophisticated alignment from app_matching
from app_matching import ImageMatcher


class ObjectClassifier:
    """Class for neural network-based object classification and similarity evaluation with sophisticated alignment."""
    
    def __init__(self, output_dir: str = "data/output/classification_results", masks_dir: Path = None):
        """
        Initialize the ObjectClassifier class.
        
        Args:
            output_dir: Directory to save classification results
            masks_dir: Directory containing object masks for better alignment
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir = masks_dir
        
        # Create subdirectories for different outputs
        self.edges_dir = self.output_dir / "edges"
        self.similarity_dir = self.output_dir / "similarity"
        self.results_dir = self.output_dir / "results"
        self.aligned_dir = self.output_dir / "aligned_comparisons"
        
        for dir_path in [self.edges_dir, self.similarity_dir, self.results_dir, self.aligned_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize sophisticated alignment matcher
        self.matcher = ImageMatcher(str(self.output_dir / "alignment_temp"))
    
    def detect_object_edges(self, image: np.ndarray, method: str = "canny") -> np.ndarray:
        """
        Detect edges of objects using neural network-based edge detection.
        
        Args:
            image: Input image as numpy array
            method: Edge detection method ("canny", "sobel", "laplacian", "neural")
            
        Returns:
            Edge-detected image
        """
        if image is None or image.size == 0:
            return image
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if method == "canny":
            # Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
        elif method == "sobel":
            # Sobel edge detection
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = np.uint8(edges)
        elif method == "laplacian":
            # Laplacian edge detection
            edges = cv2.Laplacian(gray, cv2.CV_64F)
            edges = np.uint8(np.absolute(edges))
        elif method == "neural":
            # Neural network-based edge detection (placeholder for future implementation)
            # For now, use advanced Canny with adaptive thresholds
            edges = self._neural_edge_detection(gray)
        else:
            edges = gray
        
        return edges
    
    def _neural_edge_detection(self, gray: np.ndarray) -> np.ndarray:
        """
        Neural network-based edge detection (placeholder implementation).
        
        Args:
            gray: Grayscale image
            
        Returns:
            Edge-detected image
        """
        # Placeholder for neural network implementation
        # For now, use advanced Canny with adaptive thresholds
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive threshold calculation
        mean_val = np.mean(blur)
        lower_thresh = max(0, mean_val - 30)
        upper_thresh = min(255, mean_val + 30)
        
        edges = cv2.Canny(blur, int(lower_thresh), int(upper_thresh))
        
        # Morphological operations to clean up edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return edges
    
    def calculate_similarity_metrics(self, image1: np.ndarray, image2: np.ndarray) -> Dict[str, float]:
        """
        Calculate multiple similarity metrics between two images.
        
        Args:
            image1: First image
            image2: Second image
            
        Returns:
            Dictionary of similarity metrics
        """
        metrics = {}
        
        # Ensure images are the same size
        if image1.shape != image2.shape:
            # Resize image2 to match image1
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        
        # Convert to grayscale for some metrics
        if len(image1.shape) == 3:
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = image1
            
        if len(image2.shape) == 3:
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = image2
        
        # 1. Structural Similarity Index (SSIM)
        try:
            from skimage.metrics import structural_similarity as ssim
            ssim_score = ssim(gray1, gray2)
            metrics['ssim'] = ssim_score
        except ImportError:
            metrics['ssim'] = self._calculate_ssim(gray1, gray2)
        
        # 2. Mean Squared Error (MSE)
        mse = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)
        metrics['mse'] = mse
        
        # 3. Peak Signal-to-Noise Ratio (PSNR)
        if mse > 0:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        else:
            psnr = float('inf')
        metrics['psnr'] = psnr
        
        # 4. Normalized Cross Correlation (NCC)
        ncc = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)[0][0]
        metrics['ncc'] = ncc
        
        # 5. Histogram Correlation
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        hist_corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        metrics['histogram_correlation'] = hist_corr
        
        # 6. Edge-based similarity
        edges1 = self.detect_object_edges(gray1, "canny")
        edges2 = self.detect_object_edges(gray2, "canny")
        edge_similarity = self._calculate_edge_similarity(edges1, edges2)
        metrics['edge_similarity'] = edge_similarity
        
        return metrics
    
    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate SSIM manually (fallback when scikit-image is not available).
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            SSIM score
        """
        # Simplified SSIM calculation
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1 = np.var(img1)
        sigma2 = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
        
        return ssim
    
    def _calculate_edge_similarity(self, edges1: np.ndarray, edges2: np.ndarray) -> float:
        """
        Calculate similarity based on edge structures.
        
        Args:
            edges1: First edge image
            edges2: Second edge image
            
        Returns:
            Edge similarity score
        """
        # Ensure same size
        if edges1.shape != edges2.shape:
            edges2 = cv2.resize(edges2, (edges1.shape[1], edges1.shape[0]))
        
        # Calculate intersection over union of edges
        intersection = cv2.bitwise_and(edges1, edges2)
        union = cv2.bitwise_or(edges1, edges2)
        
        if np.sum(union) > 0:
            iou = np.sum(intersection) / np.sum(union)
        else:
            iou = 0.0
        
        return iou
    
    def _sophisticated_alignment(self, template_img: np.ndarray, object_img: np.ndarray,
                                template_path: Path, object_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Perform sophisticated alignment between template and object using systematic rotation optimization.
        
        Args:
            template_img: Template image
            object_img: Object image to align with template
            template_path: Path to template image (for mask loading)
            object_path: Path to object image (for mask loading)
            
        Returns:
            Tuple of (aligned_template, aligned_object, alignment_info)
        """
        print("   🎯 Performing sophisticated alignment...")
        
        # Convert images to binary/load masks for alignment
        template_binary = None
        object_binary = None
        
        if self.masks_dir and self.masks_dir.exists():
            # Try to load corresponding masks
            template_mask_path = self.masks_dir / f"{template_path.stem}_mask.png"
            object_mask_path = self.masks_dir / f"{object_path.stem}_mask.png"
            
            if template_mask_path.exists():
                template_binary = cv2.imread(str(template_mask_path), cv2.IMREAD_GRAYSCALE)
                print(f"   🎭 Loaded template mask: {template_mask_path.name}")
            else:
                template_binary = self.matcher.create_binary_image(template_img, "adaptive")
                print(f"   🔄 Created binary template from image")
            
            if object_mask_path.exists():
                object_binary = cv2.imread(str(object_mask_path), cv2.IMREAD_GRAYSCALE)
                print(f"   🎭 Loaded object mask: {object_mask_path.name}")
            else:
                object_binary = self.matcher.create_binary_image(object_img, "adaptive")
                print(f"   🔄 Created binary object from image")
        else:
            # Fallback to binary conversion
            template_binary = self.matcher.create_binary_image(template_img, "adaptive")
            object_binary = self.matcher.create_binary_image(object_img, "adaptive")
            print("   🔄 Using binary conversion for alignment")
        
        # Perform sophisticated rotation and scale optimization
        best_position, overlap_score, transformed_object_binary = self.matcher.find_optimal_overlap(
            template_binary, object_binary, "contour_matching"
        )
        
        # Apply the same transformation to the original color image
        # Extract transformation parameters from the optimization
        transformed_object_img = self._apply_optimal_transformation(object_img, object_binary, transformed_object_binary)
        
        alignment_info = {
            'overlap_score': float(overlap_score),
            'best_position': best_position,
            'transformation_applied': True
        }
        
        print(f"   ✅ Alignment complete: Overlap={overlap_score:.3f}, Position={best_position}")
        
        return template_img, transformed_object_img, alignment_info
    
    def _apply_optimal_transformation(self, original_img: np.ndarray, original_binary: np.ndarray, 
                                    transformed_binary: np.ndarray) -> np.ndarray:
        """
        Apply the same transformation that was applied to binary image to the original color image.
        
        Args:
            original_img: Original color image
            original_binary: Original binary image
            transformed_binary: Transformed binary image
            
        Returns:
            Transformed color image
        """
        # For now, we'll use a simplified approach by detecting the transformation
        # In a more sophisticated implementation, we would store the transformation matrix
        
        # Find contours to detect transformation
        orig_contours, _ = cv2.findContours(original_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        trans_contours, _ = cv2.findContours(transformed_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not orig_contours or not trans_contours:
            return original_img
        
        # Get the largest contours
        orig_contour = max(orig_contours, key=cv2.contourArea)
        trans_contour = max(trans_contours, key=cv2.contourArea)
        
        # Calculate transformation matrix using contour matching
        try:
            # Get minimum area rectangles
            orig_rect = cv2.minAreaRect(orig_contour)
            trans_rect = cv2.minAreaRect(trans_contour)
            
            # Extract rotation and scale
            orig_angle = orig_rect[2]
            trans_angle = trans_rect[2]
            
            rotation_angle = trans_angle - orig_angle
            
            orig_size = max(orig_rect[1])
            trans_size = max(trans_rect[1])
            scale = trans_size / orig_size if orig_size > 0 else 1.0
            
            # Apply transformation to original image
            h, w = original_img.shape[:2]
            center = (w // 2, h // 2)
            
            M = cv2.getRotationMatrix2D(center, rotation_angle, scale)
            
            # Calculate new dimensions
            cos_angle = abs(M[0, 0])
            sin_angle = abs(M[0, 1])
            new_w = int((h * sin_angle) + (w * cos_angle))
            new_h = int((h * cos_angle) + (w * sin_angle))
            
            # Adjust translation
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            
            # Apply transformation
            transformed_img = cv2.warpAffine(original_img, M, (new_w, new_h), 
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            return transformed_img
            
        except Exception as e:
            print(f"   ⚠️  Transformation application failed: {e}, returning original")
            return original_img
    
    def compare_with_templates(self, object_path: Path, template_dir: Path) -> Dict[str, Dict]:
        """
        Compare an object with all templates in the template directory.
        Applies rotation alignment before comparison.
        
        Args:
            object_path: Path to the object image
            template_dir: Directory containing template images
            
        Returns:
            Dictionary of comparison results
        """
        print(f"🔍 Comparing object: {object_path.name}")
        
        # Load object image
        object_img = cv2.imread(str(object_path))
        if object_img is None:
            print(f"❌ Could not load object image: {object_path}")
            return {}
        
        # Note: We'll apply sophisticated alignment for each template comparison
        
        # Find template images
        template_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        template_files = []
        for ext in template_extensions:
            template_files.extend(template_dir.glob(ext))
            template_files.extend(template_dir.glob(ext.upper()))
        
        if not template_files:
            print(f"❌ No template images found in {template_dir}")
            return {}
        
        print(f"📊 Found {len(template_files)} templates to compare")
        
        comparison_results = {}
        
        for template_path in template_files:
            print(f"   🔄 Comparing with template: {template_path.name}")
            
            # Load template image
            template_img = cv2.imread(str(template_path))
            if template_img is None:
                print(f"   ❌ Could not load template: {template_path}")
                continue
            
            # Apply sophisticated alignment between template and object
            template_img_aligned, object_img_aligned, alignment_info = self._sophisticated_alignment(
                template_img, object_img, template_path, object_path
            )
            
            # Calculate similarity metrics using aligned images
            metrics = self.calculate_similarity_metrics(object_img_aligned, template_img_aligned)
            
            # Save edge images for visualization (using aligned images)
            object_edges = self.detect_object_edges(object_img_aligned, "neural")
            template_edges = self.detect_object_edges(template_img_aligned, "neural")
            
            edge_output_path = self.edges_dir / f"{object_path.stem}_vs_{template_path.stem}_edges.png"
            self._save_edge_comparison(object_edges, template_edges, edge_output_path)
            
            # Save back-to-back aligned comparison image
            aligned_comparison_path = self.similarity_dir / f"{object_path.stem}_vs_{template_path.stem}_aligned.png"
            self._save_aligned_comparison(object_img_aligned, template_img_aligned, aligned_comparison_path)
            
            # Extract keypoints from aligned images for point-based comparison
            object_binary = self.matcher.create_binary_image(object_img_aligned, "adaptive")
            template_binary = self.matcher.create_binary_image(template_img_aligned, "adaptive")
            
            object_points = self.matcher._extract_object_keypoints(object_binary)
            template_points = self.matcher._extract_object_keypoints(template_binary)
            
            # Calculate point-based similarity
            point_similarity = self._calculate_point_similarity(object_points, template_points)
            
            # Store results with alignment and point information
            comparison_results[template_path.name] = {
                'metrics': metrics,
                'alignment_info': alignment_info,
                'point_similarity': point_similarity,
                'keypoints_count': {'object': len(object_points), 'template': len(template_points)},
                'edge_image': str(edge_output_path),
                'aligned_comparison': str(aligned_comparison_path),
                'object_path': str(object_path),
                'template_path': str(template_path),
                'object_aligned_path': str(self.aligned_dir / f"{object_path.stem}_aligned.png"),
                'template_aligned_path': str(self.aligned_dir / f"{template_path.stem}_aligned.png")
            }
            
            # Save individual aligned images
            cv2.imwrite(str(self.aligned_dir / f"{object_path.stem}_aligned.png"), object_img_aligned)
            cv2.imwrite(str(self.aligned_dir / f"{template_path.stem}_aligned.png"), template_img_aligned)
            
            print(f"   ✅ SSIM: {metrics['ssim']:.3f}, PSNR: {metrics['psnr']:.1f}, Edge: {metrics['edge_similarity']:.3f}")
            print(f"   🎯 Alignment: Overlap={alignment_info['overlap_score']:.3f}, Points: {point_similarity:.3f}")
        
        return comparison_results
    
    def _calculate_point_similarity(self, points1: np.ndarray, points2: np.ndarray) -> float:
        """
        Calculate similarity between two sets of keypoints.
        
        Args:
            points1: First set of keypoints
            points2: Second set of keypoints
            
        Returns:
            Point similarity score (0.0 to 1.0)
        """
        if len(points1) == 0 or len(points2) == 0:
            return 0.0
        
        # Method 1: Hausdorff distance-based similarity
        try:
            # Calculate distances from each point in set 1 to closest point in set 2
            distances_1_to_2 = []
            for p1 in points1:
                distances = np.linalg.norm(points2 - p1, axis=1)
                distances_1_to_2.append(np.min(distances))
            
            # Calculate distances from each point in set 2 to closest point in set 1
            distances_2_to_1 = []
            for p2 in points2:
                distances = np.linalg.norm(points1 - p2, axis=1)
                distances_2_to_1.append(np.min(distances))
            
            # Hausdorff distance
            hausdorff_dist = max(np.max(distances_1_to_2), np.max(distances_2_to_1))
            
            # Convert to similarity (lower distance = higher similarity)
            hausdorff_similarity = 1.0 / (1.0 + hausdorff_dist / 100.0)
            
        except Exception:
            hausdorff_similarity = 0.0
        
        # Method 2: Average closest point distance
        try:
            avg_dist_1_to_2 = np.mean(distances_1_to_2)
            avg_dist_2_to_1 = np.mean(distances_2_to_1)
            avg_distance = (avg_dist_1_to_2 + avg_dist_2_to_1) / 2.0
            
            avg_distance_similarity = 1.0 / (1.0 + avg_distance / 50.0)
            
        except Exception:
            avg_distance_similarity = 0.0
        
        # Method 3: Point count similarity
        count_similarity = min(len(points1), len(points2)) / max(len(points1), len(points2))
        
        # Combine metrics
        final_similarity = (0.4 * hausdorff_similarity + 
                           0.4 * avg_distance_similarity + 
                           0.2 * count_similarity)
        
        return final_similarity
    
    def _save_edge_comparison(self, edges1: np.ndarray, edges2: np.ndarray, output_path: Path):
        """
        Save edge comparison visualization.
        
        Args:
            edges1: First edge image
            edges2: Second edge image
            output_path: Path to save the comparison image
        """
        # Create side-by-side comparison
        if edges1.shape != edges2.shape:
            edges2 = cv2.resize(edges2, (edges1.shape[1], edges1.shape[0]))
        
        # Convert to 3-channel for color visualization
        edges1_color = cv2.cvtColor(edges1, cv2.COLOR_GRAY2BGR)
        edges2_color = cv2.cvtColor(edges2, cv2.COLOR_GRAY2BGR)
        
        # Create comparison image
        comparison = np.hstack([edges1_color, edges2_color])
        
        # Add labels
        cv2.putText(comparison, "Object Edges", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "Template Edges", (edges1.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imwrite(str(output_path), comparison)
    
    def _save_aligned_comparison(self, img1: np.ndarray, img2: np.ndarray, output_path: Path):
        """
        Save back-to-back aligned comparison image.
        
        Args:
            img1: First aligned image
            img2: Second aligned image
            output_path: Path to save the comparison image
        """
        # Ensure images are the same height for side-by-side comparison
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Use the maximum height
        max_height = max(h1, h2)
        
        # Resize images to same height
        if h1 != max_height:
            scale1 = max_height / h1
            new_w1 = int(w1 * scale1)
            img1_resized = cv2.resize(img1, (new_w1, max_height))
        else:
            img1_resized = img1
            
        if h2 != max_height:
            scale2 = max_height / h2
            new_w2 = int(w2 * scale2)
            img2_resized = cv2.resize(img2, (new_w2, max_height))
        else:
            img2_resized = img2
        
        # Create side-by-side comparison
        comparison = np.hstack([img1_resized, img2_resized])
        
        # Add labels
        cv2.putText(comparison, "Object (Aligned)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "Template (Aligned)", (img1_resized.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add separator line
        cv2.line(comparison, (img1_resized.shape[1], 0), (img1_resized.shape[1], max_height), (255, 255, 255), 2)
        
        cv2.imwrite(str(output_path), comparison)
    
    def process_objects_directory(self, objects_dir: Path, templates_dir: Path) -> Dict:
        """
        Process all objects in a directory and compare them with templates.
        
        Args:
            objects_dir: Directory containing object images
            templates_dir: Directory containing template images
            
        Returns:
            Dictionary of all comparison results
        """
        print(f"📁 Processing objects directory: {objects_dir}")
        print(f"📁 Using templates directory: {templates_dir}")
        print("=" * 60)
        
        # Find object images
        object_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        object_files = []
        for ext in object_extensions:
            object_files.extend(objects_dir.glob(ext))
            object_files.extend(objects_dir.glob(ext.upper()))
        
        if not object_files:
            print(f"❌ No object images found in {objects_dir}")
            return {}
        
        print(f"📊 Found {len(object_files)} objects to process")
        
        all_results = {}
        
        for object_path in object_files:
            print(f"\n🔍 Processing object: {object_path.name}")
            print("-" * 40)
            
            # Compare with templates
            object_results = self.compare_with_templates(object_path, templates_dir)
            all_results[object_path.name] = object_results
        
        # Save comprehensive results
        self._save_comprehensive_results(all_results)
        
        return all_results
    
    def _save_comprehensive_results(self, results: Dict):
        """
        Save comprehensive comparison results to JSON file.
        
        Args:
            results: Dictionary of all comparison results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"classification_results_{timestamp}.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Clean results for JSON serialization
        clean_results = {}
        for obj_name, obj_results in results.items():
            clean_results[obj_name] = {}
            for template_name, template_data in obj_results.items():
                clean_results[obj_name][template_name] = {
                    'metrics': {k: convert_numpy(v) for k, v in template_data['metrics'].items()},
                    'alignment_info': {k: convert_numpy(v) for k, v in template_data['alignment_info'].items()},
                    'point_similarity': convert_numpy(template_data['point_similarity']),
                    'keypoints_count': {k: convert_numpy(v) for k, v in template_data['keypoints_count'].items()},
                    'edge_image': template_data['edge_image'],
                    'aligned_comparison': template_data['aligned_comparison'],
                    'object_aligned_path': template_data['object_aligned_path'],
                    'template_aligned_path': template_data['template_aligned_path'],
                    'object_path': template_data['object_path'],
                    'template_path': template_data['template_path']
                }
        
        with open(results_file, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        print(f"\n💾 Comprehensive results saved to: {results_file}")
        
        # Generate summary report
        self._generate_summary_report(clean_results, results_file.parent / f"summary_report_{timestamp}.txt")
    
    def _generate_summary_report(self, results: Dict, report_path: Path):
        """
        Generate a summary report of the classification results.
        
        Args:
            results: Cleaned results dictionary
            report_path: Path to save the summary report
        """
        with open(report_path, 'w') as f:
            f.write("Object Classification and Similarity Evaluation Report\n")
            f.write("=" * 60 + "\n\n")
            
            total_objects = len(results)
            f.write(f"Total Objects Processed: {total_objects}\n\n")
            
            for obj_name, obj_results in results.items():
                f.write(f"Object: {obj_name}\n")
                f.write("-" * 30 + "\n")
                
                if not obj_results:
                    f.write("No comparison results available.\n\n")
                    continue
                
                # Find best match for each object
                best_matches = []
                for template_name, template_data in obj_results.items():
                    metrics = template_data['metrics']
                    best_matches.append({
                        'template': template_name,
                        'ssim': metrics['ssim'],
                        'psnr': metrics['psnr'],
                        'edge_similarity': metrics['edge_similarity']
                    })
                
                # Sort by SSIM score
                best_matches.sort(key=lambda x: x['ssim'], reverse=True)
                
                f.write("Best Matches (sorted by SSIM):\n")
                for i, match in enumerate(best_matches[:3]):  # Top 3 matches
                    f.write(f"  {i+1}. {match['template']}: SSIM={match['ssim']:.3f}, "
                           f"PSNR={match['psnr']:.1f}, Edge={match['edge_similarity']:.3f}\n")
                
                f.write("\n")
        
        print(f"📊 Summary report saved to: {report_path}")


def main():
    """Main function with hardcoded parameters."""
    # Configuration parameters - modify these as needed
    objects_dir = Path('data/datasets/cut_objects')  # Directory containing cut objects
    templates_dir = Path('data/datasets/templates')  # Directory containing template images
    masks_dir = Path('data/datasets/cut_objects/masks')  # Directory containing object masks
    output_dir = Path('data/output/classification_results')  # Output directory for results
    
    print("🧠 Object Classification and Similarity Evaluation Application")
    print("=" * 70)
    print(f"📁 Objects directory: {objects_dir}")
    print(f"📁 Templates directory: {templates_dir}")
    print(f"🎭 Masks directory: {masks_dir}")
    print(f"📂 Output directory: {output_dir}")
    print()
    
    # Check if directories exist
    if not objects_dir.exists():
        print(f"❌ Objects directory not found: {objects_dir}")
        print("Please run the alignment application first to generate cut objects.")
        sys.exit(1)
    
    if not templates_dir.exists():
        print(f"❌ Templates directory not found: {templates_dir}")
        print("Please create a templates directory with template images.")
        sys.exit(1)
    
    try:
        # Initialize classifier with masks directory
        classifier = ObjectClassifier(str(output_dir), masks_dir)
        
        # Process objects and compare with templates
        results = classifier.process_objects_directory(objects_dir, templates_dir)
        
        if results:
            print("\n" + "=" * 70)
            print("🎉 Classification Complete!")
            print(f"📂 Results saved in: {output_dir}")
            print("📊 Check the JSON results file and summary report for detailed analysis")
            print("🖼️  Edge comparison images saved for visualization")
            print("=" * 70)
        else:
            print("\n❌ No objects were processed")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Error during classification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
