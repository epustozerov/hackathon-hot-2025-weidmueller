#!/usr/bin/env python3
"""
Object Cutting Application

This application cuts objects from original images based on binary masks.
It locates original images and corresponding masks, then cuts out individual
objects and saves them as separate image files.

Features:
- Locate original images in specified folder
- Locate corresponding masks in masks folder
- Cut objects from original image based on mask
- Save cut objects as separate files
- No alignment processing (just cutting)

Configuration:
- Modify parameters in the main() function
- No command line arguments needed
- Hardcoded parameters for easy customization
"""

import sys
from pathlib import Path
from typing import List, Dict

# Import alignment methods
from modules.alignment.methods import ObjectAlignment, align_image, align_images


def process_single_image(input_path: Path, output_dir: Path, image_type: str = "auto", original_path: Path = None) -> bool:
    """
    Process a single segmented image to extract objects.
    
    Args:
        input_path: Path to the segmented image (mask)
        output_dir: Directory to save aligned objects
        image_type: Type of image ("binary", "colored", or "auto")
        original_path: Path to the original image (for binary masks)
        
    Returns:
        True if successful, False otherwise
    """
    print(f"🎯 Processing single image: {input_path.name}")
    print("=" * 60)
    
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return False
    
    try:
        # Initialize alignment processor
        aligner = ObjectAlignment(str(output_dir))
        
        # For binary masks, try to find corresponding original image
        if original_path is None and 'binary' in input_path.name.lower():
            original_name = input_path.stem.replace('_sam_points_binary', '') + '.jpg'
            original_path = input_path.parent / original_name
            
            if not original_path.exists():
                print(f"❌ Original image not found: {original_path}")
                return False
        
        # Process the image
        saved_paths = aligner.process_segmented_image(input_path, image_type, original_path)
        
        if saved_paths:
            print(f"\n✅ Successfully processed {len(saved_paths)} objects")
            print(f"📂 Objects saved in: {output_dir}")
            return True
        else:
            print(f"❌ No objects found in {input_path.name}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return False


def process_directory(input_dir: Path, output_dir: Path, image_type: str = "auto") -> bool:
    """
    Process all segmented images in a directory.
    
    Args:
        input_dir: Directory containing segmented images
        output_dir: Directory to save aligned objects
        image_type: Type of image ("binary", "colored", or "auto")
        
    Returns:
        True if successful, False otherwise
    """
    print(f"📁 Processing directory: {input_dir}")
    print("=" * 60)
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return False
    
    try:
        # Initialize alignment processor
        aligner = ObjectAlignment(str(output_dir))
        
        # Process directory
        results = aligner.process_directory(input_dir)
        
        if results:
            total_objects = sum(len(paths) for paths in results.values())
            print(f"\n✅ Successfully processed {len(results)} images")
            print(f"📊 Total objects extracted: {total_objects}")
            print(f"📂 Objects saved in: {output_dir}")
            return True
        else:
            print(f"❌ No images processed")
            return False
            
    except Exception as e:
        print(f"❌ Error processing directory: {e}")
        return False


def cut_objects_from_image(original_folder: Path, masks_folder: Path, output_path: Path, image_name: str) -> bool:
    """
    Cut objects from original image based on mask.
    
    Args:
        original_folder: Folder containing original images
        masks_folder: Folder containing mask images
        output_path: Output folder for cut objects
        image_name: Base name of the image (without extension)
        
    Returns:
        True if successful, False otherwise
    """
    print(f"🔍 Processing image: {image_name}")
    print("=" * 60)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find original image
    original_image_path = original_folder / f"{image_name}.jpg"
    if not original_image_path.exists():
        print(f"❌ Original image not found: {original_image_path}")
        return False
    
    # Find mask image
    mask_image_path = masks_folder / f"{image_name}_sam_points_binary.png"
    if not mask_image_path.exists():
        print(f"❌ Mask image not found: {mask_image_path}")
        return False
    
    print(f"📸 Original image: {original_image_path.name}")
    print(f"🎭 Mask image: {mask_image_path.name}")
    
    try:
        import cv2
        import numpy as np
        
        # Load original image
        original_img = cv2.imread(str(original_image_path))
        if original_img is None:
            print(f"❌ Could not load original image: {original_image_path}")
            return False
        
        # Load mask image
        mask_img = cv2.imread(str(mask_image_path))
        if mask_img is None:
            print(f"❌ Could not load mask image: {mask_image_path}")
            return False
        
        print(f"📐 Original image shape: {original_img.shape}")
        print(f"📐 Mask image shape: {mask_img.shape}")
        
        # Convert mask to grayscale and create binary mask
        # In the mask: black = objects, white = background
        mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        # Invert the mask so black objects become white (255) and white background becomes black (0)
        binary_mask = cv2.bitwise_not(mask_gray)
        
        # Find connected components in the mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        print(f"📊 Found {num_labels - 1} objects (excluding background)")
        
        objects_cut = 0
        
        # Process each object (skip background label 0)
        for i in range(1, num_labels):
            # Get object mask
            object_mask = (labels == i).astype(np.uint8) * 255
            
            # Get object statistics
            x, y, w, h, area = stats[i]
            centroid = centroids[i]
            
            # Skip very small objects and edge artifacts
            if area < 10000:  # Increased threshold to filter out edge artifacts
                print(f"   ⏭️  Skipping small object {i} (area: {area})")
                continue
            
            # Skip edge artifacts (very thin objects at borders)
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
            if aspect_ratio > 50:  # Very thin objects (likely edge artifacts)
                print(f"   ⏭️  Skipping edge artifact {i} (aspect ratio: {aspect_ratio:.1f})")
                continue
            
            print(f"   ✂️  Cutting object {i}: area={area}, bbox=({x},{y},{w},{h})")
            
            # Extract object from original image using the mask
            object_img = original_img[y:y+h, x:x+w].copy()
            object_mask_cropped = object_mask[y:y+h, x:x+w]
            
            # Apply mask to object image (set background to black)
            object_img[object_mask_cropped == 0] = [0, 0, 0]
            
            # Save cut object
            output_filename = f"{image_name}_object_{i:02d}.png"
            output_file_path = output_path / output_filename
            
            cv2.imwrite(str(output_file_path), object_img)
            print(f"   💾 Saved: {output_filename}")
            objects_cut += 1
        
        print(f"✅ Successfully cut {objects_cut} objects from {image_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error cutting objects: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_folder(original_folder: Path, masks_folder: Path, output_path: Path) -> bool:
    """
    Process all images in the original folder and their corresponding masks.
    
    Args:
        original_folder: Folder containing original images
        masks_folder: Folder containing mask images
        output_path: Output folder for cut objects
        
    Returns:
        True if successful, False otherwise
    """
    print(f"📁 Processing folder: {original_folder}")
    print("=" * 60)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all original images
    original_images = list(original_folder.glob('*.jpg')) + list(original_folder.glob('*.jpeg'))
    
    if not original_images:
        print(f"❌ No original images found in {original_folder}")
        return False
    
    print(f"📊 Found {len(original_images)} original images to process")
    print()
    
    total_objects_cut = 0
    processed_images = 0
    
    for original_image_path in original_images:
        # Extract base name without extension
        image_name = original_image_path.stem
        
        # Find corresponding mask
        mask_image_path = masks_folder / f"{image_name}_sam_points_binary.png"
        
        if not mask_image_path.exists():
            print(f"⚠️  No mask found for {image_name}, skipping...")
            continue
        
        print(f"🔍 Processing: {image_name}")
        print("-" * 40)
        
        try:
            import cv2
            import numpy as np
            
            # Load original image
            original_img = cv2.imread(str(original_image_path))
            if original_img is None:
                print(f"❌ Could not load original image: {original_image_path}")
                continue
            
            # Load mask image
            mask_img = cv2.imread(str(mask_image_path))
            if mask_img is None:
                print(f"❌ Could not load mask image: {mask_image_path}")
                continue
            
            print(f"📐 Original: {original_img.shape}, Mask: {mask_img.shape}")
            
            # Convert mask to grayscale and create binary mask
            # In the mask: black = objects, white = background
            mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
            # Invert the mask so black objects become white (255) and white background becomes black (0)
            binary_mask = cv2.bitwise_not(mask_gray)
            
            # Find connected components in the mask
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            
            print(f"📊 Found {num_labels - 1} objects (excluding background)")
            
            objects_cut = 0
            
            # Process each object (skip background label 0)
            for i in range(1, num_labels):
                # Get object mask
                object_mask = (labels == i).astype(np.uint8) * 255
                
                # Get object statistics
                x, y, w, h, area = stats[i]
                centroid = centroids[i]
                
                # Skip very small objects and edge artifacts
                if area < 10000:  # Increased threshold to filter out edge artifacts
                    continue
                
                # Skip edge artifacts (very thin objects at borders)
                aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
                if aspect_ratio > 50:  # Very thin objects (likely edge artifacts)
                    continue
                
                print(f"   ✂️  Cutting object {i}: area={area}, bbox=({x},{y},{w},{h})")
                
                # Extract object from original image using the mask
                object_img = original_img[y:y+h, x:x+w].copy()
                object_mask_cropped = object_mask[y:y+h, x:x+w]
                
                # Apply mask to object image (set background to black)
                object_img[object_mask_cropped == 0] = [0, 0, 0]
                
                # Save cut object
                output_filename = f"{image_name}_object_{i:02d}.png"
                output_file_path = output_path / output_filename
                
                cv2.imwrite(str(output_file_path), object_img)
                print(f"   💾 Saved: {output_filename}")
                objects_cut += 1
            
            print(f"✅ Cut {objects_cut} objects from {image_name}")
            total_objects_cut += objects_cut
            processed_images += 1
            
        except Exception as e:
            print(f"❌ Error processing {image_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        print()
    
    print(f"🎉 Folder processing complete!")
    print(f"📊 Processed {processed_images} images")
    print(f"📊 Total objects cut: {total_objects_cut}")
    print(f"📂 Results saved in: {output_path}")
    
    return processed_images > 0


def analyze_segmented_images(input_dir: Path) -> None:
    """
    Analyze segmented images to understand their structure.
    
    Args:
        input_dir: Directory containing segmented images
    """
    print(f"🔍 Analyzing segmented images in: {input_dir}")
    print("=" * 60)
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    import cv2
    import numpy as np
    
    # Find all image files
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(ext))
        image_files.extend(input_dir.glob(ext.upper()))
    
    if not image_files:
        print(f"❌ No images found in {input_dir}")
        return
    
    print(f"📊 Found {len(image_files)} images to analyze")
    print()
    
    for img_path in image_files:
        print(f"📸 Analyzing: {img_path.name}")
        
        try:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"   ❌ Could not load image")
                continue
            
            print(f"   📐 Shape: {img.shape}")
            
            # Analyze image type
            unique_colors = len(np.unique(img.reshape(-1, img.shape[2]), axis=0))
            if unique_colors <= 2:
                print(f"   🎨 Type: Binary image")
            else:
                print(f"   🎨 Type: Colored image ({unique_colors} unique colors)")
            
            # Count objects
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            num_labels, _ = cv2.connectedComponents(binary)
            print(f"   🔢 Objects: {num_labels - 1} (excluding background)")
            
            print()
            
        except Exception as e:
            print(f"   ❌ Error analyzing {img_path.name}: {e}")
            print()


def main():
    """Main function with hardcoded parameters."""
    # Configuration parameters - modify these as needed
    original_folder = Path('data/datasets/segmented_images/originals')  # Folder containing original images
    masks_folder = Path('data/datasets/segmented_images/masks')  # Folder containing mask images
    output_path = Path('data/output/cut_objects')  # Output folder for cut objects
    process_single_image = False  # Set to True to process a single image, False to process whole folder
    single_image_name = '20250919_222417'  # Base name without extension (used only if process_single_image is True)
    
    print("🎯 Object Cutting Application")
    print("=" * 60)
    print(f"📁 Original folder: {original_folder}")
    print(f"🎭 Masks folder: {masks_folder}")
    print(f"📂 Output: {output_path}")
    if process_single_image:
        print(f"🖼️  Processing single image: {single_image_name}")
    else:
        print(f"📁 Processing whole folder")
    print()
    
    if process_single_image:
        # Process single image
        success = cut_objects_from_image(original_folder, masks_folder, output_path, single_image_name)
        
        if success:
            print("\n" + "=" * 60)
            print("🎉 Object Cutting Complete!")
            print(f"📂 Cut objects saved in: {output_path}")
            print("📚 Each object is saved as a separate image file")
            print("=" * 60)
        else:
            print("\n❌ Object cutting failed")
            sys.exit(1)
    else:
        # Process whole folder
        success = process_folder(original_folder, masks_folder, output_path)
        
        if success:
            print("\n" + "=" * 60)
            print("🎉 Folder Processing Complete!")
            print(f"📂 Cut objects saved in: {output_path}")
            print("📚 Each object is saved as a separate image file")
            print("=" * 60)
        else:
            print("\n❌ Folder processing failed")
            sys.exit(1)


if __name__ == "__main__":
    main()
