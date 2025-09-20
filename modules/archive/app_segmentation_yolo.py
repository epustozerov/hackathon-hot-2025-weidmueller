#!/usr/bin/env python3
"""
YOLO-based Image Segmentation Tool

Main application entry point for YOLO-based image segmentation using Ultralytics YOLO models.
"""

import sys
from pathlib import Path

# Import YOLO segmentation system
from modules.segmentation_yolo import YOLOImageSegmenter
from modules.segmentation_yolo.config import MODEL_PRESETS, get_preset_config, print_available_presets
from modules.segmentation_yolo.models import YOLOModels


def determine_model_and_settings(model=None, preset='balanced', conf=None, iou=None):
    """Determine which model and settings to use based on arguments."""
    if model:
        # Use specific model
        conf = conf if conf is not None else 0.25
        iou = iou if iou is not None else 0.7
    else:
        # Use preset
        preset_config = get_preset_config(preset)
        model = preset_config['model']
        conf = conf if conf is not None else preset_config['conf_threshold']
        iou = iou if iou is not None else preset_config['iou_threshold']
    
    return model, conf, iou


def process_single_dataset(segmenter, analyze_only=False):
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
    if analyze_only:
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
    # Configuration - modify these values as needed
    model = None  # Use specific model, or None to use preset
    preset = 'balanced'  # 'fastest', 'balanced', 'accurate', 'high_accuracy', 'maximum_accuracy'
    conf = None  # Confidence threshold (0.0-1.0), or None to use preset default
    iou = None  # IoU threshold for NMS (0.0-1.0), or None to use preset default
    input_folder = 'data/datasets/dataset_1'
    output_folder = 'data/output/output_yolo'
    multi_dataset = False  # Set to True to process all datasets
    analyze_only = False  # Set to True to only analyze images
    verbose = False  # Set to True for verbose output
    
    print("🎯 YOLO-based Image Segmentation Tool")
    print("=" * 60)
    
    # Determine model and settings
    selected_model, conf_threshold, iou_threshold = determine_model_and_settings(model, preset, conf, iou)
    
    print(f"🤖 Selected model: {selected_model}")
    print(f"⚙️  Settings: Confidence={conf_threshold}, IoU={iou_threshold}")
    print(f"📁 Input folder: {input_folder}")
    print(f"📂 Output folder: {output_folder}")
    
    if verbose:
        model_info = YOLOModels.get_model_info(selected_model)
        if model_info:
            print(f"📋 Model info: {model_info['name']} - {model_info['description']}")
            print(f"   Parameters: {model_info['params']}, mAP(mask): {model_info['map_mask']}")
    
    print()
    
    # Initialize YOLO segmenter
    try:
        segmenter = YOLOImageSegmenter(
            model_name=selected_model,
            input_folder=input_folder,
            output_folder=output_folder
        )
        
        # Set thresholds
        segmenter.set_thresholds(conf_threshold=conf_threshold, iou_threshold=iou_threshold)
        
    except Exception as e:
        print(f"❌ Error initializing YOLO segmenter: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    try:
        if multi_dataset:
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
            success = process_single_dataset(segmenter, analyze_only)
            if not success:
                sys.exit(1)
        
        print("\n" + "=" * 60)
        print("🎊 YOLO Segmentation Complete!")
        model_info = YOLOModels.get_model_info(selected_model)
        if model_info:
            print(f"Used model: {model_info['name']} - {model_info['description']}")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
