"""
Neural Image Segmenter

Main class for neural network-based image segmentation.
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

from .methods import SegmentationMethods


class NeuralImageSegmenter:
    """Neural network-based image segmentation class."""
    
    def __init__(self, input_folder="data/datasets/dataset_1", output_folder="data/output/output_neural"):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize segmentation methods
        self.methods = SegmentationMethods(self.device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_image(self, image_path):
        """Load and preprocess image."""
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # For display
        image_np = np.array(image)
        
        # For model input
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        return image_np, image_tensor, original_size
    
    def process_image(self, image_path, methods=['deeplabv3', 'unet', 'neural_clustering', 'grabcut_neural']):
        """
        Process a single image using neural network-based segmentation methods.
        """
        print(f"Processing: {image_path}")
        
        # Load image
        image_np, image_tensor, original_size = self.load_image(image_path)
        
        results = {}
        
        if 'deeplabv3' in methods:
            print("  - Running DeepLabV3 segmentation...")
            results['deeplabv3'] = self.methods.segment_with_deeplabv3(image_tensor, original_size)
        
        if 'unet' in methods:
            print("  - Running U-Net segmentation...")
            results['unet'] = self.methods.segment_with_unet(image_tensor, original_size)
        
        if 'neural_clustering' in methods:
            print("  - Running neural network-enhanced clustering...")
            results['neural_clustering'] = self.methods.segment_with_advanced_clustering(image_np)
        
        if 'grabcut_neural' in methods:
            print("  - Running neural network-guided GrabCut...")
            results['grabcut_neural'] = self.methods.segment_with_grabcut_neural(image_np)
        
        return image_np, results
    
    def save_results(self, original, results, image_name):
        """Save the segmentation results."""
        # Create a figure to display all results
        num_methods = len(results)
        fig, axes = plt.subplots(1, num_methods + 1, figsize=(5 * (num_methods + 1), 5))
        
        if num_methods == 0:
            axes = [axes]
        elif num_methods == 1:
            axes = [axes[0], axes[1]] if hasattr(axes, '__len__') else [axes, axes]
        
        # Show original
        axes[0].imshow(original)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Show segmented results
        for i, (method, segmented) in enumerate(results.items()):
            axes[i + 1].imshow(segmented)
            axes[i + 1].set_title(f'{method.replace("_", " ").title()}')
            axes[i + 1].axis('off')
            
            # Save individual segmented image
            output_path = self.output_folder / f"{image_name}_{method}_segmented.png"
            plt.imsave(output_path, segmented)
            print(f"    Saved: {output_path}")
        
        # Save comparison plot
        comparison_path = self.output_folder / f"{image_name}_neural_comparison.png"
        plt.tight_layout()
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved comparison: {comparison_path}")
    
    def process_all_images(self, methods=['deeplabv3', 'neural_clustering', 'grabcut_neural']):
        """Process all images in the input folder."""
        image_files = list(self.input_folder.glob("*.jpg")) + list(self.input_folder.glob("*.png"))
        
        if not image_files:
            print(f"No image files found in {self.input_folder}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        for image_path in image_files:
            try:
                original, results = self.process_image(image_path, methods)
                image_name = image_path.stem
                self.save_results(original, results, image_name)
                print(f"Successfully processed: {image_path.name}")
                print("-" * 50)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                import traceback
                traceback.print_exc()
    
    def process_multiple_datasets(self, base_data_folder="data/datasets", methods=['deeplabv3', 'neural_clustering', 'grabcut_neural']):
        """Process all datasets in the data/datasets folder."""
        base_path = Path(base_data_folder)
        
        if not base_path.exists():
            print(f"Base data folder not found: {base_path}")
            return
        
        # Find all dataset folders
        dataset_folders = [folder for folder in base_path.iterdir() if folder.is_dir()]
        
        if not dataset_folders:
            print(f"No dataset folders found in {base_path}")
            return
        
        print(f"Found {len(dataset_folders)} datasets to process:")
        for folder in dataset_folders:
            print(f"  - {folder.name}")
        print()
        
        # Process each dataset
        for dataset_folder in dataset_folders:
            print(f"Processing dataset: {dataset_folder.name}")
            print("=" * 60)
            
            # Update paths for this dataset
            self.input_folder = dataset_folder
            self.output_folder = Path(f"data/output/output_neural_{dataset_folder.name}")
            self.output_folder.mkdir(parents=True, exist_ok=True)
            
            # Process all images in this dataset
            self.process_all_images(methods)
            print(f"Completed dataset: {dataset_folder.name}")
            print("=" * 60)
            print()
    
    def analyze_image_complexity(self, image_path):
        """
        Analyze an image and suggest the best neural network method.
        """
        image_np, image_tensor, original_size = self.load_image(image_path)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Calculate image statistics
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # Calculate edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Calculate texture complexity using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        print(f"Neural analysis for {image_path.name}:")
        print(f"  Mean intensity: {mean_intensity:.2f}")
        print(f"  Std deviation: {std_intensity:.2f}")
        print(f"  Edge density: {edge_density:.4f}")
        print(f"  Texture complexity: {laplacian_var:.2f}")
        
        # Heuristic method selection for neural networks
        if laplacian_var > 500 and edge_density > 0.05:
            recommended = "deeplabv3"
            reason = "High texture complexity - best for detailed segmentation"
        elif edge_density > 0.08:
            recommended = "grabcut_neural"
            reason = "High edge density - neural-guided refinement works well"
        elif std_intensity > 50:
            recommended = "neural_clustering"
            reason = "High contrast - neural feature clustering effective"
        else:
            recommended = "deeplabv3"
            reason = "General purpose - pre-trained model handles diverse scenes"
        
        print(f"  Recommended method: {recommended} ({reason})")
        return recommended
