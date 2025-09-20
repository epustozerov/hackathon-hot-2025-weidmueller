#!/usr/bin/env python3
"""
Neural Network-Based Image Segmentation Tool

Main application entry point for image segmentation using neural networks.
"""

import sys
from pathlib import Path

# Import our modular segmentation system
from modules.segmentation_classic import NeuralImageSegmenter
from modules.segmentation_classic.config import AVAILABLE_METHODS, DEFAULT_METHOD_SETS


def get_selected_methods(methods=None, method_set='balanced'):
    """Determine which methods to use based on arguments."""
    if methods:
        return methods
    else:
        return DEFAULT_METHOD_SETS[method_set]


def process_single_dataset(segmenter, methods, analyze_only=False):
    """Process a single dataset."""
    if not segmenter.input_folder.exists():
        print(f"Input folder not found: {segmenter.input_folder}")
        return False
    
    image_files = list(segmenter.input_folder.glob("*.jpg")) + list(segmenter.input_folder.glob("*.png"))
    
    if not image_files:
        print(f"No images found in {segmenter.input_folder}")
        return False
    
    print(f"Found {len(image_files)} images:")
    for img in image_files:
        print(f"  - {img.name}")
    print()
    
    # Analyze images
    print("Analyzing images for best neural network method:")
    print("-" * 60)
    for image_path in image_files:
        segmenter.analyze_image_complexity(image_path)
        print()
    
    if analyze_only:
        print("Analysis complete. Skipping processing as requested.")
        return True
    
    # Process images
    print("Processing all images with neural network segmentation methods:")
    print("-" * 60)
    segmenter.process_all_images(methods=methods)
    
    print(f"\nProcessing complete!")
    print(f"Results saved in: {segmenter.output_folder}")
    
    # Show summary
    if segmenter.output_folder.exists():
        output_files = list(segmenter.output_folder.glob("*"))
        print(f"Generated {len(output_files)} output files")
    
    return True


def main():
    """Main function."""
    # Configuration - modify these values as needed
    input_folder = 'data/datasets/dataset_1'
    output_folder = 'data/output/output_neural'
    methods = None  # Use specific methods, or None to use method_set
    method_set = 'balanced'  # 'fastest', 'balanced', 'accurate', 'high_quality'
    multi_dataset = False  # Set to True to process all datasets
    analyze_only = False  # Set to True to only analyze images
    
    print("Neural Network-Based Image Segmentation Tool")
    print("=" * 60)
    
    selected_methods = get_selected_methods(methods, method_set)
    print(f"Selected methods: {', '.join(selected_methods)}")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print()
    
    segmenter = NeuralImageSegmenter(input_folder=input_folder, output_folder=output_folder)
    
    try:
        if multi_dataset:
            print("Multi-dataset processing mode")
            print("-" * 60)
            segmenter.process_multiple_datasets(methods=selected_methods)
            print("\nMulti-dataset processing complete!")
        else:
            print("Single dataset processing mode")
            print("-" * 60)
            success = process_single_dataset(segmenter, selected_methods, analyze_only)
            if not success:
                sys.exit(1)
        
        print("\n" + "=" * 60)
        print("Neural Network Methods Used:")
        for method in selected_methods:
            if method in AVAILABLE_METHODS:
                info = AVAILABLE_METHODS[method]
                print(f"  {info['name']} - {info['description']}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
