#!/usr/bin/env python3
"""
SAM 2-based Image Segmentation Tool

Main application entry point for SAM 2-based image segmentation using Meta's 
Segment Anything Model 2, now with support for both SAM 2 and SAM 2.1 models.
"""

import sys
from typing import List, Tuple

# Import SAM 2 segmentation system
from modules.segmentation_sam import SAMImageSegmenter
from modules.segmentation_sam.config import get_preset_config
from modules.segmentation_sam.models import SAMModels


def determine_model_and_settings(model=None, preset='balanced'):
    """Determine which model to use based on arguments."""
    if model:
        return model
    else:
        preset_config = get_preset_config(preset)
        return preset_config['model']


def parse_points(points_str: List[str]) -> List[Tuple[int, int]]:
    """Parse point coordinates from arguments."""
    points = []
    for point_str in points_str:
        try:
            x, y = map(int, point_str.split(','))
            points.append((x, y))
        except ValueError:
            print(f"⚠️  Invalid point format: {point_str}. Expected: x,y")
    return points


def parse_box(box_str: str) -> Tuple[int, int, int, int]:
    """Parse box coordinates from arguments."""
    try:
        x1, y1, x2, y2 = map(int, box_str.split(','))
        return x1, y1, x2, y2
    except ValueError:
        print(f"⚠️  Invalid box format: {box_str}. Expected: x1,y1,x2,y2")
        return None


def process_single_dataset(segmenter, prompt_type='points', points=None, box=None, points_per_side=32, analyze_only=False):
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
        print("🔍 Analyzing images for SAM processing:")
        print("-" * 60)
        
        for image_path in image_files:
            analysis = segmenter.analyze_image_for_sam(image_path)
            if analysis.get('success', False):
                print(f"📊 Analysis for {image_path.name}:")
                print(f"   Dimensions: {analysis['dimensions']['width']}x{analysis['dimensions']['height']}")
                print(f"   Estimated objects: {analysis['estimated_objects']}")
                print(f"   Recommended model: {analysis['recommended_model']}")
                print(f"   Recommended prompt: {analysis['recommended_prompt']}")
                print(f"   Suggestion: {analysis['prompt_suggestion']}")
                print(f"   Reasoning: {analysis['reasoning']}")
            else:
                print(f"❌ Failed to analyze {image_path.name}: {analysis.get('error', 'Unknown error')}")
            print()
        
        print("✅ Analysis complete. Skipping processing as requested.")
        return True
    
    # Process images based on prompt type
    print(f"🚀 Processing images with SAM segmentation:")
    print(f"   Prompt type: {prompt_type}")
    print("-" * 60)
    
    results = []
    successful = 0
    
    for image_path in image_files:
        try:
            if prompt_type == 'points':
                # Use custom points or default center point
                if points:
                    points_list = parse_points(points)
                    labels = [1] * len(points_list)  # All foreground
                else:
                    # Use center point as default
                    import cv2
                    img = cv2.imread(str(image_path))
                    if img is not None:
                        h, w = img.shape[:2]
                        points_list = [(w // 2, h // 2)]
                    else:
                        points_list = [(400, 300)]  # Fallback
                    labels = [1]
                
                result = segmenter.segment_with_points(image_path, points_list, labels, save_results=True)
                
            elif prompt_type == 'box':
                # Use custom box or default center box
                if box:
                    box_coords = parse_box(box)
                    if box_coords is None:
                        continue
                else:
                    # Use center box as default
                    import cv2
                    img = cv2.imread(str(image_path))
                    if img is not None:
                        h, w = img.shape[:2]
                        box_coords = (w//4, h//4, 3*w//4, 3*h//4)
                    else:
                        box_coords = (100, 100, 300, 200)  # Fallback
                
                result = segmenter.segment_with_box(image_path, box_coords, save_results=True)
                
            elif prompt_type == 'everything':
                result = segmenter.segment_everything(image_path, points_per_side, save_results=True)
            
            results.append(result)
            if result.get('success', False):
                successful += 1
                
        except Exception as e:
            print(f"❌ Error processing {image_path.name}: {e}")
            results.append({'success': False, 'error': str(e)})
        
        print("-" * 60)
    
    # Summary
    print(f"📊 Processing Summary:")
    print(f"   ✅ Successfully processed: {successful}/{len(results)} images")
    
    if successful > 0:
        print(f"   💾 Results saved in: {segmenter.output_folder}")
        
        # Show segmentation summary
        total_masks = 0
        for result in results:
            if result.get('success', False) and 'stats' in result:
                stats = result['stats']
                total_masks += stats.get('num_masks', 0)
        
        if total_masks > 0:
            print(f"   🎯 Total masks generated: {total_masks}")
    
    return True


def main():
    """Main function."""
    # Configuration - modify these values as needed
    model = None  # Use specific model, or None to use preset
    preset = 'balanced'  # 'fastest', 'balanced', 'high_quality', 'maximum_accuracy'
    prompt_type = 'points'  # 'points', 'box', 'everything'
    points = None  # List of point coordinates as strings, e.g., ['400,300', '500,400']
    box = None  # Box coordinates as string, e.g., '100,100,300,200'
    points_per_side = 32  # For 'everything' prompt type
    input_folder = 'data/datasets/dataset_1_selection'
    output_folder = 'data/output/output_sam_selection'
    multi_dataset = False  # Set to True to process all datasets
    analyze_only = False  # Set to True to only analyze images
    verbose = False  # Set to True for verbose output
    
    print("🎯 SAM 2-based Image Segmentation Tool")
    print("=" * 60)
    
    # Determine model
    selected_model = determine_model_and_settings(model, preset)
    
    print(f"🤖 Selected model: {selected_model}")
    print(f"🎯 Prompt type: {prompt_type}")
    print(f"📁 Input folder: {input_folder}")
    print(f"📂 Output folder: {output_folder}")
    
    if verbose:
        model_info = SAMModels.get_model_info(selected_model)
        if model_info:
            print(f"📋 Model info: {model_info['name']} - {model_info['description']}")
            print(f"   Version: {model_info.get('version', '2.0')}")
            print(f"   Size: {model_info['size']}, Speed: {model_info['speed']}")
    
    print()
    
    # Initialize SAM segmenter
    try:
        segmenter = SAMImageSegmenter(
            model_name=selected_model,
            input_folder=input_folder,
            output_folder=output_folder
        )
        
    except Exception as e:
        print(f"❌ Error initializing SAM segmenter: {e}")
        print("\n💡 Make sure you have:")
        print("   1. Installed SAM 2: pip install git+https://github.com/facebookresearch/segment-anything-2.git")
        print("   2. Downloaded model checkpoints to models/segmentation_sam/weights/")
        print("   3. Run: python app_sam.py --download-info for more details")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    try:
        if multi_dataset:
            print("🔄 Multi-dataset processing mode")
            print("-" * 60)
            results = segmenter.process_multiple_datasets(
                prompt_type=prompt_type, 
                save_results=True
            )
            
            # Summary for multi-dataset
            total_datasets = len(results)
            successful_datasets = sum(1 for dataset_results in results.values() 
                                    if any(r.get('success', False) for r in dataset_results))
            
            print(f"\n🎉 Multi-dataset processing complete!")
            print(f"✅ Successfully processed: {successful_datasets}/{total_datasets} datasets")
            print("📂 Results saved in: data/output/output_sam_[dataset_name]/")
            
        else:
            print("📝 Single dataset processing mode")
            print("-" * 60)
            success = process_single_dataset(segmenter, prompt_type, points, box, points_per_side, analyze_only)
            if not success:
                sys.exit(1)
        
        print("\n" + "=" * 60)
        print("🎊 SAM Segmentation Complete!")
        model_info = SAMModels.get_model_info(selected_model)
        if model_info:
            print(f"Used model: {model_info['name']} - {model_info['description']}")
            print(f"Version: {model_info.get('version', '2.0')}")
        print("📚 Learn more: https://ai.meta.com/sam2/")
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