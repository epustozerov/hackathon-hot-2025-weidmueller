#!/usr/bin/env python3
"""
Neural Network-Based Image Segmentation Tool

Main application entry point for image segmentation using neural networks.
"""

import argparse
import sys
from pathlib import Path

# Import our modular segmentation system
from modules.segmentation import NeuralImageSegmenter
from modules.segmentation.config import AVAILABLE_METHODS, DEFAULT_METHOD_SETS


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Neural Network-Based Image Segmentation Tool")
    
    parser.add_argument('--methods', nargs='+', choices=list(AVAILABLE_METHODS.keys()),
                       help='Specific segmentation methods to use')
    
    parser.add_argument('--method-set', choices=list(DEFAULT_METHOD_SETS.keys()), default='balanced',
                       help='Predefined set of methods (default: balanced)')
    
    parser.add_argument('--input', type=str, default='data/datasets/dataset_1',
                       help='Input folder containing images')
    
    parser.add_argument('--output', type=str, default='data/output/output_neural',
                       help='Output folder for results')
    
    parser.add_argument('--multi-dataset', action='store_true',
                       help='Process all datasets in data/datasets/ folder')
    
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze images without processing')
    
    return parser.parse_args()


def get_selected_methods(args):
    """Determine which methods to use based on arguments."""
    if args.methods:
        return args.methods
    else:
        return DEFAULT_METHOD_SETS[args.method_set]


def process_single_dataset(segmenter, methods, args):
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
    
    if args.analyze_only:
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
    args = parse_arguments()
    
    print("Neural Network-Based Image Segmentation Tool")
    print("=" * 60)
    
    methods = get_selected_methods(args)
    print(f"Selected methods: {', '.join(methods)}")
    print(f"Input folder: {args.input}")
    print(f"Output folder: {args.output}")
    print()
    
    segmenter = NeuralImageSegmenter(input_folder=args.input, output_folder=args.output)
    
    try:
        if args.multi_dataset:
            print("Multi-dataset processing mode")
            print("-" * 60)
            segmenter.process_multiple_datasets(methods=methods)
            print("\nMulti-dataset processing complete!")
        else:
            print("Single dataset processing mode")
            print("-" * 60)
            success = process_single_dataset(segmenter, methods, args)
            if not success:
                sys.exit(1)
        
        print("\n" + "=" * 60)
        print("Neural Network Methods Used:")
        for method in methods:
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
