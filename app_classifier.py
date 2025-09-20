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


class ObjectClassifier:
    """Class for neural network-based object classification and similarity evaluation."""
    
    def __init__(self, output_dir: str = "data/output/classification_results"):
        """
        Initialize the ObjectClassifier class.
        
        Args:
            output_dir: Directory to save classification results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different outputs
        self.edges_dir = self.output_dir / "edges"
        self.similarity_dir = self.output_dir / "similarity"
        self.results_dir = self.output_dir / "results"
        
        for dir_path in [self.edges_dir, self.similarity_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
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
    
    def compare_with_templates(self, object_path: Path, template_dir: Path) -> Dict[str, Dict]:
        """
        Compare an object with all templates in the template directory.
        
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
            
            # Calculate similarity metrics
            metrics = self.calculate_similarity_metrics(object_img, template_img)
            
            # Save edge images for visualization
            object_edges = self.detect_object_edges(object_img, "neural")
            template_edges = self.detect_object_edges(template_img, "neural")
            
            edge_output_path = self.edges_dir / f"{object_path.stem}_vs_{template_path.stem}_edges.png"
            self._save_edge_comparison(object_edges, template_edges, edge_output_path)
            
            # Store results
            comparison_results[template_path.name] = {
                'metrics': metrics,
                'edge_image': str(edge_output_path),
                'object_path': str(object_path),
                'template_path': str(template_path)
            }
            
            print(f"   ✅ SSIM: {metrics['ssim']:.3f}, PSNR: {metrics['psnr']:.1f}, Edge: {metrics['edge_similarity']:.3f}")
        
        return comparison_results
    
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
                    'edge_image': template_data['edge_image'],
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
    objects_dir = Path('data/output/cut_objects')  # Directory containing cut objects
    templates_dir = Path('data/datasets/templates')  # Directory containing template images
    output_dir = Path('data/output/classification_results')  # Output directory for results
    
    print("🧠 Object Classification and Similarity Evaluation Application")
    print("=" * 70)
    print(f"📁 Objects directory: {objects_dir}")
    print(f"📁 Templates directory: {templates_dir}")
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
        # Initialize classifier
        classifier = ObjectClassifier(str(output_dir))
        
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
