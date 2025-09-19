#!/usr/bin/env python3
"""
YOLO-based Image Segmentation Tool

Main application entry point for YOLO-based image segmentation using Ultralytics YOLO models.
"""

import argparse
import sys
from pathlib import Path

# Import YOLO segmentation system
from modules.segmentation_yolo import YOLOImageSegmenter
from modules.segmentation_yolo.config import MODEL_PRESETS, get_preset_config, print_available_presets
from modules.segmentation_yolo.models import YOLOModels


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="YOLO-based Image Segmentation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available YOLO Models:
  yolo11n-seg    - Nano (fastest, 2.9M params)
  yolo11s-seg    - Small (balanced, 10.1M params)  
  yolo11m-seg    - Medium (good accuracy, 22.4M params)
  yolo11l-seg    - Large (high accuracy, 27.6M params)
  yolo11x-seg    - Extra Large (maximum accuracy, 62.1M params)

Model Presets:
  fastest        - yolo11n-seg with optimized settings
  balanced       - yolo11s-seg with balanced settings (default)
  accurate       - yolo11m-seg with quality settings
  high_accuracy  - yolo11l-seg with high-quality settings
  maximum_accuracy - yolo11x-seg with maximum quality settings

Examples:
  python app_yolo.py                                    # Use balanced preset
  python app_yolo.py --model yolo11n-seg               # Use specific model
  python app_yolo.py --preset fastest                  # Use fastest preset
  python app_yolo.py --conf 0.3 --iou 0.6             # Custom thresholds
  python app_yolo.py --input custom/path               # Custom input folder
  python app_yolo.py --multi-dataset --preset accurate # Process all datasets
  python app_yolo.py --analyze-only                    # Analyze images only
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=YOLOModels.get_available_models(),
        help='Specific YOLO model to use'
    )
    
    parser.add_argument(
        '--preset',
        choices=list(MODEL_PRESETS.keys()),
        default='balanced',
        help='Model preset with optimized settings (default: balanced)'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        help='Confidence threshold (0.0-1.0)'
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        help='IoU threshold for NMS (0.0-1.0)'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='data/datasets/dataset_1',
        help='Input folder containing images (default: data/datasets/dataset_1)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/output/output_yolo',
        help='Output folder for results (default: data/output/output_yolo)'
    )
    
    parser.add_argument(
        '--multi-dataset',
        action='store_true',
        help='Process all datasets in data/datasets/ folder'
    )
    
    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='Only analyze images without processing'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List all available YOLO models and exit'
    )
    
    parser.add_argument(
        '--list-presets',
        action='store_true',
        help='List all available presets and exit'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def determine_model_and_settings(args):
    """Determine which model and settings to use based on arguments."""
    if args.model:
        # Use specific model
        model = args.model
        conf = args.conf if args.conf is not None else 0.25
        iou = args.iou if args.iou is not None else 0.7
    else:
        # Use preset
        preset_config = get_preset_config(args.preset)
        model = preset_config['model']
        conf = args.conf if args.conf is not None else preset_config['conf_threshold']
        iou = args.iou if args.iou is not None else preset_config['iou_threshold']
    
    return model, conf, iou


def process_single_dataset(segmenter, args):
    """Process a single dataset."""
    if not segmenter.input_folder.exists():
        print(f"❌ Input folder not found: {segmenter.input_folder}")
        return False
    
    # Find image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(segmenter.input_folder.glob(ext))
        image_files.extend(segmenter.input_folder.glob(ext.upper()))
    
    if not image_files:
        print(f"❌ No images found in {segmenter.input_folder}")
        return False
    
    print(f"📁 Found {len(image_files)} images:")
    for img in image_files:
        print(f"   - {img.name}")
    print()
    
    # Analyze images if requested
    if args.analyze_only:
        print("🔍 Analyzing images for YOLO processing:")
        print("-" * 60)
        
        for image_path in image_files:
            analysis = segmenter.analyze_image_for_yolo(image_path)
            if analysis.get('success', False):
                print(f"📊 Analysis for {image_path.name}:")
                print(f"   Dimensions: {analysis['dimensions']['width']}x{analysis['dimensions']['height']}")
                print(f"   Recommended model: {analysis['recommended_model']}")
                print(f"   Recommended confidence: {analysis['recommended_conf_threshold']}")
                print(f"   Reasoning: {analysis['reasoning']}")
            else:
                print(f"❌ Failed to analyze {image_path.name}: {analysis.get('error', 'Unknown error')}")
            print()
        
        print("✅ Analysis complete. Skipping processing as requested.")
        return True
    
    # Process images
    print("🚀 Processing images with YOLO segmentation:")
    print("-" * 60)
    
    results = segmenter.process_all_images(save_results=True)
    
    # Summary
    successful = sum(1 for r in results if r.get('success', False))
    print(f"\n📊 Processing Summary:")
    print(f"   ✅ Successfully processed: {successful}/{len(results)} images")
    
    if successful > 0:
        print(f"   💾 Results saved in: {segmenter.output_folder}")
        
        # Show detected classes summary
        all_classes = {}
        total_objects = 0
        for result in results:
            if result.get('success', False) and 'stats' in result:
                stats = result['stats']
                total_objects += stats.get('total_objects', 0)
                classes = stats.get('classes_detected', {})
                for class_name, count in classes.items():
                    all_classes[class_name] = all_classes.get(class_name, 0) + count
        
        if all_classes:
            print(f"   🏷️  Total objects detected: {total_objects}")
            print(f"   📋 Classes found: {', '.join(all_classes.keys())}")
    
    return True


def main():
    """Main function."""
    args = parse_arguments()
    
    # Handle list options
    if args.list_models:
        YOLOModels.print_models_info()
        return
    
    if args.list_presets:
        print_available_presets()
        return
    
    print("🎯 YOLO-based Image Segmentation Tool")
    print("=" * 60)
    
    # Determine model and settings
    model, conf, iou = determine_model_and_settings(args)
    
    print(f"🤖 Selected model: {model}")
    print(f"⚙️  Settings: Confidence={conf}, IoU={iou}")
    print(f"📁 Input folder: {args.input}")
    print(f"📂 Output folder: {args.output}")
    
    if args.verbose:
        model_info = YOLOModels.get_model_info(model)
        if model_info:
            print(f"📋 Model info: {model_info['name']} - {model_info['description']}")
            print(f"   Parameters: {model_info['params']}, mAP(mask): {model_info['map_mask']}")
    
    print()
    
    # Initialize YOLO segmenter
    try:
        segmenter = YOLOImageSegmenter(
            model_name=model,
            input_folder=args.input,
            output_folder=args.output
        )
        
        # Set thresholds
        segmenter.set_thresholds(conf_threshold=conf, iou_threshold=iou)
        
    except Exception as e:
        print(f"❌ Error initializing YOLO segmenter: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    try:
        if args.multi_dataset:
            print("🔄 Multi-dataset processing mode")
            print("-" * 60)
            results = segmenter.process_multiple_datasets(save_results=True)
            
            # Summary for multi-dataset
            total_datasets = len(results)
            successful_datasets = sum(1 for dataset_results in results.values() 
                                    if any(r.get('success', False) for r in dataset_results))
            
            print(f"\n🎉 Multi-dataset processing complete!")
            print(f"✅ Successfully processed: {successful_datasets}/{total_datasets} datasets")
            print("📂 Results saved in: data/output/output_yolo_[dataset_name]/")
            
        else:
            print("📝 Single dataset processing mode")
            print("-" * 60)
            success = process_single_dataset(segmenter, args)
            if not success:
                sys.exit(1)
        
        print("\n" + "=" * 60)
        print("🎊 YOLO Segmentation Complete!")
        model_info = YOLOModels.get_model_info(model)
        if model_info:
            print(f"Used model: {model_info['name']} - {model_info['description']}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
